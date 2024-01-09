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
        disable_input=False, sound=False, sound_emulated=False, verbose=True
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
        self.counts_map = np.zeros((444, 336)) # to solve the map
        self.verbose = verbose
        self.reward_the_agent_for_completing_the_pokedex: bool = reward_the_agent_for_completing_the_pokedex
        self.reward_the_agent_for_the_normalize_gain_of_new_money = reward_the_agent_for_the_normalize_gain_of_new_money
        self.time = 0

    def reset(self, seed=None, options=None, max_episode_steps=20480, reward_scale=4.0):
        '''Resets the game. Seeding is NOT supported'''
        load_pyboy_state(self.game, self.initial_state)

        self.time = 0
        self.max_episode_steps = max_episode_steps
        self.reward_scale = reward_scale
         
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

        return self.render()[::2, ::2], {}

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
        
        
        run_action_on_emulator(self.game, self.screen, ACTIONS[action],
            self.headless, fast_video=fast_video)
        self.time += 1

        # Exploration reward
        r, c, map_n = ram_map.position(self.game)
        self.seen_coords.add((r, c, map_n))
        self.seen_maps.add(map_n)
        exploration_reward = 0.01 * len(self.seen_coords)
        glob_r, glob_c = game_map.local_to_global(r, c, map_n)
        try:
            self.counts_map[glob_r, glob_c] += 1
        except:
            pass

        # Level reward
        party, party_size, party_levels = ram_map.party(self.game)
        self.max_level_sum = max(self.max_level_sum, sum(party_levels))
        if self.max_level_sum < 30:
            level_reward = 1 * self.max_level_sum
        else:
            level_reward = 30 + (self.max_level_sum - 30)/4

        # Healing and death rewards
        hp = ram_map.hp(self.game)
        hp_delta = hp - self.last_hp
        party_size_constant = party_size == self.last_party_size

        # Only reward if not reviving at pokecenter
        if hp_delta > 0 and party_size_constant and not self.is_dead:
            self.total_healing += hp_delta

        # Dead if hp is zero
        if hp <= 0 and self.last_hp > 0:
            self.death_count += 1
            self.is_dead = True
        elif hp > 0.01: # TODO: Check if this matters
            self.is_dead = False

        # Update last known values for next iteration
        self.last_hp = hp
        self.last_party_size = party_size

        # Set rewards
        healing_reward = self.total_healing
        death_reward = -0.05 * self.death_count

        # Opponent level reward
        max_opponent_level = max(ram_map.opponent(self.game))
        self.max_opponent_level = max(self.max_opponent_level, max_opponent_level)
        opponent_level_reward = 0.2 * self.max_opponent_level

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
        assert reward_the_agent_seing_new_pokemon >= 0 and reward_the_agent_seing_new_pokemon <= 1
        
        # Is in a trainer battle
        next_state_is_in_battle = ram_map.is_in_battle(self.game)
        
        
        # Total item count
        item_count = ram_map.total_items(self.game)
        
        # total hm count
        hm_count = ram_map.total_hm_party_has(self.game)
        
        # number of hm moves my pokemon party has
        total_number_hm_moves_that_my_pokemon_party_has = ram_map.total_hm_party_has(self.game)
        
        reward = self.reward_scale * (event_reward + level_reward + 
            opponent_level_reward + death_reward + badges_reward +
            healing_reward + exploration_reward)
        if self.reward_the_agent_for_completing_the_pokedex:
            reward += reward_for_completing_the_pokedex
        if self.reward_the_agent_for_the_normalize_gain_of_new_money:
            reward += normalize_gain_of_new_money_reward
        reward_for_battle = 0
        # Reward the Agent for choosing to be in a trainer battle and not losing
        if current_state_is_in_battle == ram_map.BattleState.NOT_IN_BATTLE and next_state_is_in_battle == ram_map.BattleState.TRAINER_BATTLE:
            reward_for_battle += 1
        elif current_state_is_in_battle == ram_map.BattleState.TRAINER_BATTLE and next_state_is_in_battle == ram_map.BattleState.LOST_BATTLE:
            reward_for_battle -= 1
        reward += reward_for_battle

        # Subtract previous reward
        # TODO: Don't record large cumulative rewards in the first place
        if self.last_reward is None:
            reward = 0
            self.last_reward = 0
        else:
            nxt_reward = reward
            reward -= self.last_reward
            self.last_reward = nxt_reward

        info = {}
        done = self.time >= self.max_episode_steps
        #done = True
        if done:
            info = {
                'reward': {
                    'delta': reward,
                    'event': event_reward,
                    'level': level_reward,
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
                'maps_explored': len(self.seen_maps),
                'party_size': party_size,
                'highest_pokemon_level': max(party_levels),
                'total_party_level': sum(party_levels),
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
                #"current_state_is_in_battle": current_state_is_in_battle, enum
                #"next_state_is_in_battle": next_state_is_in_battle, #enum
                "current_state_money": current_state_money,
                "next_state_money": next_state_money,
                "current_state_pokemon_seen": current_state_pokemon_seen,
                "next_state_pokemon_seen": next_state_pokemon_seen,
                "current_state_completing_the_pokedex": current_state_completing_the_pokedex,
                "next_state_completing_the_pokedex": next_state_completing_the_pokedex,
            }

        if self.verbose:
            print(
                f'steps: {self.time}',
                f'exploration reward: {exploration_reward}',
                f'level_Reward: {level_reward}',
                f'healing: {healing_reward}',
                f'death: {death_reward}',
                f'op_level: {opponent_level_reward}',
                f'badges reward: {badges_reward}',
                f'event reward: {event_reward}',
                f'money: {money}',
                f'ai reward: {reward}',
                f'party size: {party_size}',
                f'party levels: {party_levels}',
                f'party hp: {hp}',
                f"In a trainer battle: {current_state_is_in_battle}",
                f'Info: {info}',
            )

        return self.render()[::2, ::2], reward, done, done, info
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
