from pdb import set_trace as T
from gymnasium import Env, spaces
import numpy as np
import os

from pokegym.pyboy_binding import (ACTIONS, make_env, open_state_file,
    load_pyboy_state, run_action_on_emulator)
from pokegym import ram_map, game_map
from rich import print


def play():
    '''Creates an environment and plays it'''
    env = Environment(rom_path='pokemon_red.gb', state_path=None, headless=False,
        disable_input=False, sound=False, sound_emulated=False, verbose=True,
    )

    env.reset()
    env.game.set_emulation_speed(1)

    # Display available actions
    print("Available actions:")
    for idx, action in enumerate(ACTIONS):
        print(f"{idx}: {action}")

    # Create a mapping from WindowEvent to action index
    window_event_to_action = {
        'PRESS_ARROW_DOWN': 0,
        'PRESS_ARROW_LEFT': 1,
        'PRESS_ARROW_RIGHT': 2,
        'PRESS_ARROW_UP': 3,
        'PRESS_BUTTON_A': 4,
        'PRESS_BUTTON_B': 5,
        'PRESS_BUTTON_START': 6,
        'PRESS_BUTTON_SELECT': 7,
        # Add more mappings if necessary
    }

    while True:
        # Get input from pyboy's get_input method
        input_events = env.game.get_input()
        env.game.tick()
        env.render()
        if len(input_events) == 0:
            continue

        for event in input_events:
            event_str = str(event)
            if event_str in window_event_to_action:
                action_index = window_event_to_action[event_str]
                observation, reward, done, _, info = env.step(
                    action_index, fast_video=False)

                # Check for game over
                if done:
                    print(f"{done}")
                    break

                # Additional game logic or information display can go here
                print(f"new Reward: {reward}\n")
                

class Base:
    def __init__(self, rom_path='pokemon_red.gb',
            state_path=None, headless=True, quiet=False, **kwargs):
        '''Creates a PokemonRed environment'''
        if state_path is None:
            state_path = __file__.rstrip('environment.py') + 'has_pokedex_nballs.state'

        self.game, self.screen = make_env(
            rom_path, headless, quiet, **kwargs)

        self.initial_state = open_state_file(state_path)
        self.headless = headless
        R, C = self.screen.raw_screen_buffer_dims()
        self.observation_space = spaces.Dict({
            'screen': spaces.Box(
                low=0, high=255, dtype=np.uint8,
                shape=(R // 2, C // 2, 3),
            ),
            "party_size": spaces.Discrete(6),
            "player_row": spaces.Box(low=0, high=444, shape=(1,), dtype=np.uint16),
            "player_column": spaces.Box(low=0, high=436, shape=(1,), dtype=np.uint16),
        })
        self.action_space = spaces.Discrete(len(ACTIONS))

    def reset(self, seed=None, options=None):
        '''Resets the game. Seeding is NOT supported'''
        load_pyboy_state(self.game, self.initial_state)
        return self.screen.screen_ndarray(), {}
        
    '''
    You can view this where the update of observation is done because in every step 
    the render is called which display the observation 
    '''
    def render(self):
        return self.screen.screen_ndarray()

    def step(self, action):
        run_action_on_emulator(self.game, self.screen, ACTIONS[action], self.headless)
        return self.render(), 0, False, False, {}

    def close(self):
        self.game.stop(False)


class Environment(Base):
    def __init__(self, rom_path='pokemon_red.gb',
            state_path=None, headless=True, quiet=False, verbose=False, 
            reward_the_agent_for_completing_the_pokedex=True,
            reward_the_agent_for_the_normalize_gain_of_new_money = True,
            **kwargs):
        super().__init__(rom_path, state_path, headless, quiet, **kwargs)
        # https://github.com/xinpw8/pokegym/blob/d44ee5048d597d7eefda06a42326220dd9b6295f/pokegym/environment.py#L233
        self.counts_map = np.zeros((444, 436)) # to solve the map
        self.verbose = verbose
        self.reward_the_agent_for_completing_the_pokedex: bool = reward_the_agent_for_completing_the_pokedex
        self.reward_the_agent_for_the_normalize_gain_of_new_money = reward_the_agent_for_the_normalize_gain_of_new_money
        self.last_map = -1

    def reset(self, seed=None, options=None,  max_episode_steps = 2^15, reward_scale=1):
        '''Resets the game. Seeding is NOT supported'''
        load_pyboy_state(self.game, self.initial_state)

        self.time = 0
        self.reward_scale = reward_scale
        self.max_episode_steps: int = max_episode_steps  # 32768
         
        self.max_events = 0
        self.max_level_sum = 0
        self.max_opponent_level = 0

        self.seen_coords = set()
        self.seen_maps = set()

        self.death_count = 0
        self.total_healing = 0
        self.last_hp = 1.0
        self.last_party_size = 1
        self.last_reward = None
        self.seen_coords_no_reward = set()
        
        self.number_of_wild_battle = 0
        self.number_of_trainer_battle = 0

        #return self.render()[::2, ::2], {}
        return {"screen": self.render()[::2, ::2], "party_size": 0, "player_row":0, "player_column":0}, {}

    def step(self, action, fast_video=True):
        # Reward the agent for seeing new pokemon that it never had seen 
        current_state_pokemon_seen = ram_map.pokemon_seen(self.game) # this is new pokemon you have seen
        # Reward teh agent for catching new pokemon to copmlete the pokedex
        current_state_completing_the_pokedex = ram_map.pokemon_caught(self.game) # this is new pokemon you have seen
        # Normalize the increase reward base the opportuniyes on having learniIugn rate 1 / ( 100 - current level of pokemon)
        # Check what is the current value of the money
        current_state_money = ram_map.money(self.game)
        # Current x
        r, c, map_n = ram_map.position(self.game)
        # Check if you are in a battle 
        current_state_is_in_battle:ram_map.BattleState = ram_map.is_in_battle(self.game)
        # Reward the increaseing the team pokemon levels
        prev_party, prev_party_size, prev_party_levels = ram_map.party(self.game)
        
        
        
        run_action_on_emulator(self.game, self.screen, ACTIONS[action],
            self.headless, fast_video=fast_video)
        self.time += 1

        # Exploration reward reward the agent travel a new state in the game 
        row, column, map_n = ram_map.position(self.game)
        try:
            global_row, global_column = game_map.local_to_global(row, column, map_n)
        except IndexError:
            print(f'IndexError: index {global_row} or {global_column} for {map_n} is out of bounds for axis 0 with size 444.')
            global_row = -1
            global_column = -1
        exploration_reward = 0
        if (row, column, map_n) not in self.seen_coords:
            prev_size = len(self.seen_coords)
            self.seen_coords.add((row, column, map_n))
            exploration_reward = normalize_gaine_exploration = 1.0 - normalize_value(len(self.seen_coords), 0, 444*436, 0, 1)
            assert normalize_gaine_exploration >= 0.0 and normalize_gaine_exploration <= 1.0, f"normalize_gaine_exploration: {normalize_gaine_exploration}"
            assert len(self.seen_coords) > prev_size, f"len(self.seen_coords): {len(self.seen_coords)} prev_size: {prev_size}"
        
        
        self.update_heat_map(row, column, map_n)


        # Level reward
        #party, party_size, party_levels = ram_map.party(self.game)
        next_state_party, next_state_party_size, next_state_party_levels = ram_map.party(self.game)
        self.max_level_sum = sum(next_state_party_levels)
        reward_the_agent_increase_the_level_of_the_pokemon: float =  ( sum(next_state_party_levels) - sum(prev_party_levels) )  / (600- sum(prev_party_levels))





        # Set rewards
        healing_reward = self.total_healing
        death_reward = -0.05 * self.death_count

        # Opponent level reward
        max_opponent_level = max(ram_map.opponent(self.game))
        self.max_opponent_level = max(self.max_opponent_level, max_opponent_level)
        opponent_level_reward = 0

        # Badge reward
        badges = ram_map.badges(self.game)
        badges_reward = 5 * badges

        # Event reward
        events = ram_map.events(self.game)
        self.max_events = max(self.max_events, events)
        event_reward = self.max_events
        # Money Reward
        next_state_money = money = ram_map.money(self.game)
        assert next_state_money >= 0 and next_state_money <= 999999, f"next_state_money: {next_state_money}"
        normalize_gain_of_new_money_reward = normalize_value(next_state_money - current_state_money, -999999.0, 999999.0, -1, 1)
        assert normalize_gain_of_new_money_reward >=  ( -1.0 - 1e5) and normalize_gain_of_new_money_reward <= 1.0, f"normalize_gain_of_new_money_reward: {normalize_gain_of_new_money_reward} the current state money is {current_state_money} and the next state money is {next_state_money}"
        
        
        
        # Seen Pokemon
        next_state_pokemon_seen = ram_map.pokemon_seen(self.game)
        reward_the_agent_seing_new_pokemon = next_state_pokemon_seen - current_state_pokemon_seen
        assert ( reward_the_agent_seing_new_pokemon >= 0 and reward_the_agent_seing_new_pokemon <= 1) or reward_the_agent_seing_new_pokemon==3, f"reward_the_agent_seing_new_pokemon: {reward_the_agent_seing_new_pokemon}"
        
        # Completing the pokedex
        next_state_completing_the_pokedex = ram_map.pokemon_caught(self.game)
        reward_for_completing_the_pokedex = next_state_completing_the_pokedex - current_state_completing_the_pokedex
        assert reward_for_completing_the_pokedex >= 0 and reward_for_completing_the_pokedex <= 1
        
        # Is in a trainer battle
        next_state_is_in_battle = ram_map.is_in_battle(self.game)
        if current_state_is_in_battle == ram_map.BattleState.NOT_IN_BATTLE and next_state_is_in_battle == ram_map.BattleState.WILD_BATTLE:
            self.number_of_wild_battle += 1
        
        
        # Total item count
        item_count = ram_map.total_items(self.game)
        
        # total hm count
        hm_count = ram_map.total_hm_party_has(self.game)
        
        # number of hm moves my pokemon party has
        total_number_hm_moves_that_my_pokemon_party_has = ram_map.total_hm_party_has(self.game)
        
        reward_for_battle = 0
        # Reward the Agent for choosing to be in a trainer battle and not losing
        if current_state_is_in_battle == ram_map.BattleState.NOT_IN_BATTLE and next_state_is_in_battle == ram_map.BattleState.TRAINER_BATTLE:
            reward_for_battle += 1
            self.number_of_trainer_battle += 1
        elif current_state_is_in_battle == ram_map.BattleState.TRAINER_BATTLE and next_state_is_in_battle == ram_map.BattleState.LOST_BATTLE:
            reward_for_battle -= 1
        
        reward: float =  (
                event_reward 
                + reward_the_agent_seing_new_pokemon 
                + opponent_level_reward 
                + death_reward 
                + badges_reward 
                + healing_reward 
                + exploration_reward
                +  reward_for_completing_the_pokedex if self.reward_the_agent_for_completing_the_pokedex else 0
                + normalize_gain_of_new_money_reward if self.reward_the_agent_for_the_normalize_gain_of_new_money else 0
                + reward_for_battle
        )


        info = {}
        done = self.time >= self.max_episode_steps
        #done = True
        if done:
            info = {
                'reward': {
                    'reward': reward,
                    'event': event_reward,
                    'level':reward_the_agent_increase_the_level_of_the_pokemon,
                    'opponent_level': opponent_level_reward,
                    'death': death_reward,
                    'badges': badges_reward,
                    'healing': healing_reward,
                    'exploration': exploration_reward,
                    "seeing_new_pokemon": reward_the_agent_seing_new_pokemon,
                    "completing_the_pokedex": reward_for_completing_the_pokedex,
                    "normalize_gain_of_new_money": normalize_gain_of_new_money_reward,
                    "reward_for_battle": reward_for_battle, # Reward the Agent for choosing to be in a trainer battle and not losing
                },
                'time': self.time,
                'maps_explored': len(self.seen_maps),
                'party_size': next_state_party_size,
                'highest_pokemon_level': max(next_state_party_levels),
                'total_party_level': sum(next_state_party_levels),
                'deaths': self.death_count,
                'badge_1': float(badges == 1),
                'badge_2': float(badges > 1),
                'event': events,
                'money': money,
                'pokemon_exploration_map': self.counts_map,
                "pokemon_seen": next_state_pokemon_seen,
                "total_items": item_count,
                "hm_item_counts": hm_count,
                "hm_moves": total_number_hm_moves_that_my_pokemon_party_has,
                "max_opponent_level": self.max_opponent_level,
                "money": next_state_money,
                "pokemon_seen": next_state_pokemon_seen,
                "pokedex": next_state_completing_the_pokedex,
                "number_of_wild_battle": self.number_of_wild_battle,
                "number_of_trainer_battle": self.number_of_trainer_battle,
                #"current_state_is_in_battle": current_state_is_in_battle, enum
                #"next_state_is_in_battle": next_state_is_in_battle, #enum
                "player_row_position": row,
                "player_column_position": column,
                "player_global_row_position": global_row,
                "player_global_column_position": global_column,
                "next_state_row": row,
                "next_state_column": column,
                "current_state_money": current_state_money,
                "next_state_money": next_state_money,
                "current_state_pokemon_seen": current_state_pokemon_seen,
                "next_state_pokemon_seen": next_state_pokemon_seen,
                "current_state_completing_the_pokedex": current_state_completing_the_pokedex,
                "next_state_completing_the_pokedex": next_state_completing_the_pokedex,
            }

        if self.verbose:
            print(
                f'time: {self.time}',
                f'exploration reward: {exploration_reward}',
                f'healing: {healing_reward}',
                f'death: {death_reward}',
                f'op_level: {opponent_level_reward}',
                f'badges reward: {badges_reward}',
                f'event reward: {event_reward}',
                f'money: {money}',
                f'ai reward: {reward}',
                f"In a trainer battle: {current_state_is_in_battle}",
                f'Info: {info}',
            )
        # Observation , reward, done, info
        observation = {
            'screen': self.render()[::2, ::2],
            "party_size": next_state_party_size / 6,
            "player_row": row,
            "player_column": column,
        }
        return observation, reward, done, done, info
    def update_heat_map(self, r, c, current_map):
        '''
        Updates the heat map based on the agent's current position.

        Args:
            r (int): global y coordinate of the agent's position.
            c (int): global x coordinate of the agent's position.
            current_map (int): ID of the current map (map_n)

        Updates the counts_map to track the frequency of visits to each position on the map.
        '''
        # Convert local position to global position
        try:
            glob_r, glob_c = game_map.local_to_global(r, c, current_map)
        except IndexError:
            print(f'IndexError: index {glob_r} or {glob_c} for {current_map} is out of bounds for axis 0 with size 444.')
            glob_r = 0
            glob_c = 0

        # Update heat map based on current map
        if self.last_map == current_map or self.last_map == -1:
            # Increment count for current global position
                try:
                    self.counts_map[glob_r, glob_c] += 1
                except:
                    pass
        else:
            # Reset count for current global position if it's a new map for warp artifacts
            self.counts_map[(glob_r, glob_c)] = -1

        # Update last_map for the next iteration
        self.last_map = current_map
def normalize_value(x: float, min_x: float, max_x: float, a: float, b: float) -> float:
    """Normalize a value from its original range to a new specified range.
    
    Args:
        x (float): The value to be normalized.
        min_x (float): The minimum value of the original range. 
        max_x (float): The maximum value of the original range.
        a (float): The lower bound of the target range.
        b (float): The upper bound of the target range.
        
    Returns:
        float: The normalized value.
    """
    
    return a + ((x - min_x) * (b - a)) / (max_x - min_x)
