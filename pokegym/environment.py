from collections import deque
from pdb import set_trace as T
from typing import Literal
from gymnasium import Env, spaces
import numpy as np
import os
import io, os

from pokegym.pyboy_binding import (ACTIONS, make_env, open_state_file,
    load_pyboy_state, run_action_on_emulator)
from pokegym import ram_map, game_map
from rich import print

EVENT_FLAGS_START = 0xD747
EVENT_FLAGS_END = (
    0xD7F6  # 0xD761 # 0xD886 temporarily lower event flag range for obs input
)

PARTY_SIZE = 0xD163
CUT_SEQ = [
    ((0x3D, 1, 1, 0, 4, 1), (0x3D, 1, 1, 0, 1, 1)),
    ((0x50, 1, 1, 0, 4, 1), (0x50, 1, 1, 0, 1, 1)),
]

CUT_GRASS_SEQ = deque([(0x52, 255, 1, 0, 1, 1), (0x52, 255, 1, 0, 1, 1), (0x52, 1, 1, 0, 1, 1)])
CUT_FAIL_SEQ = deque([(-1, 255, 0, 0, 4, 1), (-1, 255, 0, 0, 1, 1), (-1, 255, 0, 0, 1, 1)])
TM_HM_MOVES = set(
    [
        5,  # Mega punch
        0xD,  # Razor wind
        0xE,  # Swords dance
        0x12,  # Whirlwind
        0x19,  # Mega kick
        0x5C,  # Toxic
        0x20,  # Horn drill
        0x22,  # Body slam
        0x24,  # Take down
        0x26,  # Double edge
        0x3D,  # Bubble beam
        0x37,  # Water gun
        0x3A,  # Ice beam
        0x3B,  # Blizzard
        0x3F,  # Hyper beam
        0x06,  # Pay day
        0x42,  # Submission
        0x44,  # Counter
        0x45,  # Seismic toss
        0x63,  # Rage
        0x48,  # Mega drain
        0x4C,  # Solar beam
        0x52,  # Dragon rage
        0x55,  # Thunderbolt
        0x57,  # Thunder
        0x59,  # Earthquake
        0x5A,  # Fissure
        0x5B,  # Dig
        0x5E,  # Psychic
        0x64,  # Teleport
        0x66,  # Mimic
        0x68,  # Double team
        0x73,  # Reflect
        0x75,  # Bide
        0x76,  # Metronome
        0x78,  # Selfdestruct
        0x79,  # Egg bomb
        0x7E,  # Fire blast
        0x81,  # Swift
        0x82,  # Skull bash
        0x87,  # Softboiled
        0x8A,  # Dream eater
        0x8F,  # Sky attack
        0x9C,  # Rest
        0x56,  # Thunder wave
        0x95,  # Psywave
        0x99,  # Explosion
        0x9D,  # Rock slide
        0xA1,  # Tri attack
        0xA4,  # Substitute
        0x0F,  # Cut
        0x13,  # Fly
        0x39,  # Surf
        0x46,  # Strength
        0x94,  # Flash
    ]
)

RESET_MAP_IDS = set(
    [
        0x0,  # Pallet Town
        0x1,  # Viridian City
        0x2,  # Pewter City
        0x3,  # Cerulean City
        0x4,  # Lavender Town
        0x5,  # Vermilion City
        0x6,  # Celadon City
        0x7,  # Fuchsia City
        0x8,  # Cinnabar Island
        0x9,  # Indigo Plateau
        0xA,  # Saffron City
        0xF,  # Route 4 (Mt Moon)
        0x10,  # Route 10 (Rock Tunnel)
        0xE9,  # Silph Co 9F (Heal station)
    ]
)

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
            state_path = __file__.rstrip('environment.py') + 'Bulbasaur_fast_text_no_battle_animations_fixed_battle.state

        # Make the environment
        self.game, self.screen = make_env(rom_path, headless, quiet,
                                          save_video=False, **kwargs)
        self.initial_states = open_state_file(state_path)
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
            "total_party_hit_point" : spaces.Box(low=0, high=999, shape=(1,), dtype=np.uint16),
            "total_party_max_hit_point" : spaces.Box(low=0, high=999, shape=(1,), dtype=np.uint16),
            "party_health_ratio": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "total_party_level": spaces.Box(low=0, high=600, shape=(1,), dtype=np.uint16),
            "each_pokemon_level": spaces.Box(low=0, high=100, shape=(6,), dtype=np.uint8),
            "type_of_battle" :spaces.Box(low=-1, high=2, shape=(1,), dtype=np.int8),
            "player_pokemon_party_id": spaces.Box(low=0, high=255, shape=(6,), dtype=np.uint8),
            "opponent_pokemon_party_id": spaces.Box(low=0, high=255, shape=(6,), dtype=np.uint8),
        
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
    def save_state(self):
        state = io.BytesIO()
        state.seek(0)
        self.game.save_state(state)
        self.initial_states = state
    def load_last_state(self):
        return self.initial_states

class Environment(Base):
    def __init__(self, rom_path='pokemon_red.gb',
            state_path=None, headless=True, quiet=False, verbose=False, 
            reward_the_agent_for_completing_the_pokedex=True,
            reward_the_agent_for_the_normalize_gain_of_new_money = True,
            punish_wipe_out:bool = True,
            perfect_ivs:bool = True,
            **kwargs):
        super().__init__(rom_path, state_path, headless, quiet, **kwargs)
        # https://github.com/xinpw8/pokegym/blob/d44ee5048d597d7eefda06a42326220dd9b6295f/pokegym/environment.py#L233
        self.counts_map = np.zeros((444, 436)) # to solve the map
        self.verbose = verbose
        self.reward_the_agent_for_completing_the_pokedex: bool = reward_the_agent_for_completing_the_pokedex
        self.reward_the_agent_for_the_normalize_gain_of_new_money = reward_the_agent_for_the_normalize_gain_of_new_money
        self.last_map = -1
        self.punish_wipe_out: bool = punish_wipe_out
        self.reset_count = 0
        self.seen_maps_no_reward = set()
        self.max_episode_steps: int = 100_000_000
        self.perfect_ivs = perfect_ivs
        self.pokecenter_ids: list[int] = [0x01, 0x02, 0x03, 0x0F, 0x15, 0x05, 0x06, 0x04, 0x07, 0x08, 0x0A, 0x09]
        
        self.first = True # The reset method will be called first before nay step is occured
    


    def reset(self, seed=None, options=None,  max_episode_steps = 100_000_000, reward_scale=1):
        '''Resets the game to the previous save steps. Seeding is NOT supported'''
        if self.first:
            self.recent_screen = deque()
            self.init_mem()
            self.moves_obtained = np.zeros(0xA5, dtype=np.uint8)
        else:
            self.moves_obtained.fill(0)
        #load_pyboy_state(self.game, self.initial_state)
        """Resets the game. Seeding is NOT supported"""
        # https://github.com/xinpw8/pokegym/blob/baseline_0.6/pokegym/environment.py
        load_pyboy_state(self.game, self.load_last_state()) # load a saved state

        self.time = 0
        self.reward_scale = reward_scale
        self.max_episode_steps: int = max_episode_steps  # 65536 or 2^16
        print(f"self.max_episode_steps: {self.max_episode_steps}")
         
        self.max_events = 0
        self.max_level_sum = 0
        self.max_opponent_level = 0

        self.seen_maps = set()

        self.death_count = 0
        self.total_healing = 0
        self.last_hp = 1.0
        self.last_party_size = 1
        self.last_reward = None
        self.seen_coords_no_reward = set()
        
        self.number_of_wild_battle = 0
        self.number_of_trainer_battle = 0
        self.number_of_gym_leader_music_is_playing = 0
        self.total_wipe_out = 0
        self.total_numebr_attempted_to_run = 0
        self.total_number_of_opponent_pokemon_fainted = 0
        
        self.taught_cut = self.check_if_party_has_cut()
        self.reset_count += 1
        
        self.max_map_progress = 0 
        self.first = False

        #return self.render()[::2, ::2], {}
        assert isinstance( np.array(ram_map.party(self.game)[2]), np.ndarray)
        return {"screen": self.render()[::2, ::2], 
                "party_size": ram_map.party(self.game)[1],
                "player_row": ram_map.position(self.game)[0],
                "player_column": ram_map.position(self.game)[1],
                "total_party_hit_point": ram_map.total_party_hit_point(self.game),
                "total_party_max_hit_point": ram_map.total_party_max_hit_point(self.game),
                "party_health_ratio": ram_map.party_health_ratio(self.game),
                "total_party_level": sum(ram_map.party(self.game)[2]),
                "each_pokemon_level": np.array(ram_map.party(self.game)[2]),
                "type_of_battle": ram_map.is_in_battle(self.game).value, # 0 means not in battle, 1 means wild battle, 2 means trainer battle
                "player_pokemon_party_id": ram_map.get_party_pokemon_id(self.game),
                "opponent_pokemon_party_id": ram_map.get_opponent_party_pokemon_id(self.game),
                }, {}

    def step(self, action, fast_video=True):
        # Reward the agent for seeing new pokemon that it never had seen 
        current_state_pokemon_seen = ram_map.pokemon_seen(self.game) # this is new pokemon you have seen
        # Reward teh agent for catching new pokemon to copmlete the pokedex
        current_state_completing_the_pokedex = ram_map.pokemon_caught(self.game) # this is new pokemon you have seen
        # Normalize the increase reward base the opportuniyes on having learniIugn rate 1 / ( 100 - current level of pokemon)
        # Check what is the current value of the money
        current_state_money = ram_map.money(self.game)
        # Current x
        r, c, prev_map_n = ram_map.position(self.game)
        # Check if you are in a battle 
        current_state_is_in_battle:ram_map.BattleState = ram_map.is_in_battle(self.game)
        # Reward the increaseing the team pokemon levels
        prev_party, prev_party_size, prev_party_levels = ram_map.party(self.game)
        
        # Previous Health Ratio
        prev_health_ratio = ram_map.party_health_ratio(self.game)
        
        # Previous own a gym badge
        prev_badges_one  = ram_map.check_if_player_has_gym_one_badge(self.game)
        
        # current opponent pokemon health points
        current_state_opponent_pokemon_health_points:np.array = ram_map.get_opponent_party_pokemon_hp(self.game)
        prev_seen_npcs:int  = sum(self.seen_npcs.values())
        
        
        run_action_on_emulator(self.game, self.screen, ACTIONS[action],
            self.headless, fast_video=fast_video)
        self.time += 1
        ### Cut and Talking to NPCS
        
        # Cut check
        # 0xCFC6 - wTileInFrontOfPlayer
        # 0xCFCB - wUpdateSpritesEnabled
        if self.read_m(0xD057) == 0:
            if self.taught_cut:
                player_direction = self.game.get_memory_value(0xC109)
                x, y, map_id = self.get_game_coords()  # x, y, map_id
                if player_direction == 0:  # down
                    coords = (x, y + 1, map_id)
                if player_direction == 4:
                    coords = (x, y - 1, map_id)
                if player_direction == 8:
                    coords = (x - 1, y, map_id)
                if player_direction == 0xC:
                    coords = (x + 1, y, map_id)
                self.cut_state.append(
                    (
                        self.game.get_memory_value(0xCFC6),
                        self.game.get_memory_value(0xCFCB),
                        self.game.get_memory_value(0xCD6A),
                        self.game.get_memory_value(0xD367),
                        self.game.get_memory_value(0xD125),
                        self.game.get_memory_value(0xCD3D),
                    )
                )
                if tuple(list(self.cut_state)[1:]) in CUT_SEQ:
                    self.cut_coords[coords] = 10
                    self.cut_tiles[self.cut_state[-1][0]] = 1
                elif self.cut_state == CUT_GRASS_SEQ:
                    self.cut_coords[coords] = 0.01
                    self.cut_tiles[self.cut_state[-1][0]] = 1
                elif deque([(-1, *elem[1:]) for elem in self.cut_state]) == CUT_FAIL_SEQ:
                    self.cut_coords[coords] = 0.01
                    self.cut_tiles[self.cut_state[-1][0]] = 1

            # check if the font is loaded this occur when you are talking 
            if self.game.get_memory_value(0xCFC4):
                # check if we are talking to a hidden object:
                player_direction = self.game.get_memory_value(0xC109)
                player_y_tiles = self.game.get_memory_value(0xD361)
                player_x_tiles = self.game.get_memory_value(0xD362)
                if (
                    self.game.get_memory_value(0xCD3D) != 0x0
                    and self.game.get_memory_value(0xCD3E) != 0x0
                ):
                    # add hidden object to seen hidden objects
                    self.seen_hidden_objs[
                        (
                            self.game.get_memory_value(0xD35E),
                            self.game.get_memory_value(0xCD3F),
                        )
                    ] = 1
                elif any(
                    self.find_neighboring_sign(
                        sign_id, player_direction, player_x_tiles, player_y_tiles
                    )
                    for sign_id in range(self.game.get_memory_value(0xD4B0))
                ):
                    pass
                else:
                    # get information for player
                    player_y = self.game.get_memory_value(0xC104)
                    player_x = self.game.get_memory_value(0xC106)
                    # get the npc who is closest to the player and facing them
                    # we go through all npcs because there are npcs like
                    # nurse joy who can be across a desk and still talk to you

                    # npc_id 0 is the player
                    npc_distances = (
                        (
                            self.find_neighboring_npc(npc_id, player_direction, player_x, player_y),
                            npc_id,
                        )
                        for npc_id in range(1, self.game.get_memory_value(0xD4E1))
                    )
                    npc_candidates = [x for x in npc_distances if x[0]]
                    if npc_candidates:
                        _, npc_id = min(npc_candidates, key=lambda x: x[0])
                        self.seen_npcs[(self.game.get_memory_value(0xD35E), npc_id)] = 1
                        self.seen_npcs_since_blackout.add(
                            (self.game.get_memory_value(0xD35E), npc_id)
                        )
        # check if the npc is talking to you

        # Exploration reward reward the agent travel a new state in the game 
        row, column, map_n = ram_map.position(self.game)
       
        """Saves the current state if the agent is in a new map location that has not 
        previously given exploration rewards. This allows the agent to continue 
        exploring new states without being rewarded repeatedly for visiting the 
        same states."""
        # State the state when base on many factor on the aspects where you want to be in the increase exploration
        
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
            exploration_reward: float =  1.0 - ( len(self.seen_coords) / ( 436 * 444 ) ) #  it cannot visit all the places, therefore it should low esimate but at the point it should be able many thing at ht epoint
            assert exploration_reward >= 0.0 and exploration_reward <= 1.0, f"normalize_gaine_exploration: {exploration_reward}"
            assert len(self.seen_coords) > prev_size, f"len(self.seen_coords): {len(self.seen_coords)} prev_size: {prev_size}"
            assert len(self.seen_coords) - prev_size == 1, f"len(self.seen_coords): {len(self.seen_coords)} prev_size: {prev_size}"
        
        
        self.update_heat_map(row, column, map_n)


        # Level reward
        #party, party_size, party_levels = ram_map.party(self.game)
        next_state_party, next_state_party_size, next_state_party_levels = ram_map.party(self.game)
        self.max_level_sum = sum(next_state_party_levels)
        reward_the_agent_increase_the_level_of_the_pokemon: float =   sum(next_state_party_levels) - sum(prev_party_levels)
        if np.count_nonzero(next_state_party_levels) != np.count_nonzero(prev_party_levels):  
            reward_the_agent_increase_the_level_of_the_pokemon = 0 # you should get a reward only if you increase the level of the pokemon not by capturing new pokemon 
        reward_the_agent_increase_the_level_of_the_pokemon: float = reward_the_agent_increase_the_level_of_the_pokemon / 600
        reward_the_agent_for_increasing_the_party_size: float = ( next_state_party_size - prev_party_size ) / 6
        #assert reward_the_agent_increase_the_level_of_the_pokemon >= 0 and reward_the_agent_increase_the_level_of_the_pokemon <= 1, f"reward_the_agent_increase_the_level_of_the_pokemon: {reward_the_agent_increase_the_level_of_the_pokemon}"





        # Set rewards
        #healing_reward = self.total_healing
        death_reward = 0
        # gym
        # Badge reward
        badges_reward = 0
        if not prev_badges_one and  ram_map.check_if_player_has_gym_one_badge(self.game):
            badges_reward += 10

        # Event reward
        events = ram_map.events(self.game)
        event_reward  = 0
        if events > self.max_events:
            event_reward += 1
            self.max_events = events
            
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
        
        # Reward the agent increase the health ratio health party by healing only not by adding a new pokemon
        next_health_ratio = ram_map.party_health_ratio(self.game)
        assert next_health_ratio >= 0 and next_health_ratio <= 1, f"next_health_ratio: {next_health_ratio}"
        reward_for_healing = max( next_health_ratio - prev_health_ratio , 0)
        assert reward_for_healing >= 0 and reward_for_healing <= 1.0, f"reward_for_healing: {reward_for_healing}"
        if prev_party_size != next_state_party_size or ram_map.total_party_hit_point(self.game) or prev_health_ratio < next_health_ratio:
            reward_for_healing = 0
        reward_for_battle = 0
        # Reward the Agent for choosing to be in a trainer battle and not losing
        if current_state_is_in_battle == ram_map.BattleState.NOT_IN_BATTLE and next_state_is_in_battle == ram_map.BattleState.TRAINER_BATTLE:
            reward_for_battle += 1
            self.number_of_trainer_battle += 1
            # Reward the Agent for choosing to be in a gym battle
            if ram_map.check_if_gym_leader_music_is_playing(self.game):
                reward_for_battle += 1
                self.number_of_gym_leader_music_is_playing += 1
        if current_state_is_in_battle == ram_map.BattleState.TRAINER_BATTLE and next_state_is_in_battle == ram_map.BattleState.LOST_BATTLE:
            reward_for_battle -= .5 # Punished the agent for losing a trainer battle a bit not to lose but still want to fight
            self.death_count += 1
        if current_state_is_in_battle == ram_map.BattleState.WILD_BATTLE and next_state_is_in_battle == ram_map.BattleState.LOST_BATTLE:
            reward_for_battle -= .5 # Punished the agent for losing a wild battle
            self.death_count += 1
        
        wipe_out = 0
        if ram_map.total_party_hit_point(self.game) == 0:
            self.total_wipe_out += 1 # Wipe out means the agent has lost all of its pokemon in battle or poison
            wipe_out += 1
        opponent_level_reward = 0
        if current_state_is_in_battle == ram_map.BattleState.TRAINER_BATTLE or next_state_is_in_battle == ram_map.BattleState.TRAINER_BATTLE:
            if self.max_opponent_level < max(ram_map.opponent(self.game)):
                self.max_opponent_level = max(ram_map.opponent(self.game))
                opponent_level_reward += 1
        discourage_running_from_battle = 0
        #if current_state_is_in_battle == ram_map.BattleState.WILD_BATTLE and next_state_is_in_battle == ram_map.BattleState.NOT_IN_BATTLE:
        #    self.total_numebr_attempted_to_run += 1
        #    discourage_running_from_battle -= 1
        reward_the_agent_for_fainting_a_opponent_pokemon_during_battle = 0
        if current_state_is_in_battle == ram_map.BattleState.TRAINER_BATTLE and (np.count_nonzero(current_state_opponent_pokemon_health_points) < np.count_nonzero(ram_map.get_opponent_party_pokemon_hp(self.game))):
            reward_the_agent_for_fainting_a_opponent_pokemon_during_battle += 1
            self.total_number_of_opponent_pokemon_fainted += 1
        
        # Reward the agent if taught of one pokemon in the team with the moves cuts 
        reward_for_teaching_a_pokemon_on_the_team_with_move_cuts: int  = self.check_if_party_has_cut() - self.taught_cut
        assert reward_for_teaching_a_pokemon_on_the_team_with_move_cuts >= 0
        self.taught_cut = self.check_if_party_has_cut()
       
        if self.perfect_ivs:
            self.set_perfect_iv_dvs()
        
        next_seen_npcs = sum(self.seen_npcs.values())
        reward_seeen_npcs:int  = next_seen_npcs - prev_seen_npcs
        
        reward_visiting_a_new_pokecenter: Literal[1, 0]  = self.update_visited_pokecenter_list()
         
        

        
        reward: float =  (
                event_reward 
                + reward_the_agent_seing_new_pokemon 
                + opponent_level_reward 
                + death_reward 
                + badges_reward 
                + reward_for_healing 
                +  ( exploration_reward * 2 )
                +  reward_for_completing_the_pokedex if self.reward_the_agent_for_completing_the_pokedex else 0
                + normalize_gain_of_new_money_reward
                + reward_for_battle
                + reward_the_agent_increase_the_level_of_the_pokemon   
                + reward_the_agent_for_increasing_the_party_size
                + discourage_running_from_battle
                + reward_the_agent_for_fainting_a_opponent_pokemon_during_battle
                + wipe_out * -1 if self.punish_wipe_out else 0
                + reward_for_teaching_a_pokemon_on_the_team_with_move_cuts
                + reward_seeen_npcs
                + reward_visiting_a_new_pokecenter
        )

        info = {}
        done = self.time >= self.max_episode_steps
        if self.time % 1024 == 0 or done:
            info = {
                'reward': {
                    'reward': reward,
                    'event': event_reward,
                    'level':reward_the_agent_increase_the_level_of_the_pokemon,
                    'opponent_level': opponent_level_reward,
                    'death': death_reward,
                    'badges': badges_reward,
                    'for_healing': reward_for_healing,
                    'exploration': exploration_reward,
                    "seeing_new_pokemon": reward_the_agent_seing_new_pokemon,
                    "completing_the_pokedex": reward_for_completing_the_pokedex,
                    "normalize_gain_of_new_money": normalize_gain_of_new_money_reward,
                    "winning_battle": reward_for_battle, # Reward the Agent for choosing to be in a trainer battle and not losing
                    "increase_party_size": reward_the_agent_for_increasing_the_party_size,
                    "discourage_running_from_battle": discourage_running_from_battle,
                    "reward_the_agent_for_fainting_a_opponent_pokemon_during_battle": reward_the_agent_for_fainting_a_opponent_pokemon_during_battle,
                    "reaward_for_teaching_a_pokemon_on_the_team_with_move_cuts": reward_for_teaching_a_pokemon_on_the_team_with_move_cuts,
                    "reward_seeen_npcs": reward_seeen_npcs,
                    "reward_visiting_a_new_pokecenter": reward_visiting_a_new_pokecenter,
                },
                'time': self.time,
                "max_episode_steps": self.max_episode_steps,
                'maps_explored': len(self.seen_maps),
                "number_of_uniqiue_coordinate_it_explored": len(self.seen_coords),
                'party_size': next_state_party_size,
                'highest_pokemon_level': max(next_state_party_levels),
                'total_party_level': sum(next_state_party_levels),
                'deaths': self.death_count,
                'badge_1': ram_map.check_if_player_has_gym_one_badge(self.game),
                "badges": self.get_badges(), # Fix it latter
                "npc": sum(self.seen_npcs.values()),
                "hidden_obj": sum(self.seen_hidden_objs.values()),
                'event': events,
                'money': money,
                "met_bill": int(self.read_bit(0xD7F1, 0)),
                "used_cell_separator_on_bill": int(self.read_bit(0xD7F2, 3)),
                "ss_ticket": int(self.read_bit(0xD7F2, 4)),
                "met_bill_2": int(self.read_bit(0xD7F2, 5)),
                "bill_said_use_cell_separator": int(self.read_bit(0xD7F2, 6)),
                "left_bills_house_after_helping": int(self.read_bit(0xD7F2, 7)),
                'pokemon_exploration_map': self.counts_map,
                "pokemon_seen": next_state_pokemon_seen,
                "total_items": item_count,
                "hm_item_counts": hm_count,
                "hm_moves": total_number_hm_moves_that_my_pokemon_party_has,
                "max_opponent_level": self.max_opponent_level,
                "money": next_state_money,
                "pokemon_seen": next_state_pokemon_seen,
                "taught_cut": int(self.check_if_party_has_cut()),
                "cut_coords": sum(self.cut_coords.values()),
                "cut_tiles": len(self.cut_tiles),
                "pokedex": next_state_completing_the_pokedex,
                "number_of_wild_battle": self.number_of_wild_battle,
                "number_of_trainer_battle": self.number_of_trainer_battle,
                "total_party_hit_point" : ram_map.total_party_hit_point(self.game),
                "total_party_max_hit_point" : ram_map.total_party_max_hit_point(self.game),
                "party_health_ratio": ram_map.party_health_ratio(self.game),
                "number_of_time_gym_leader_music_is_playing": self.number_of_gym_leader_music_is_playing,
                "visited_pokemon_center": len(self.visited_pokemon_center),
                "total_wipe_out": self.total_wipe_out,
                "wipe_out:": wipe_out,
                "total_number_of_time_attempted_to_run": self.total_numebr_attempted_to_run,
                "reset_count": self.reset_count,
                "current_state_is_in_battle": current_state_is_in_battle.value , 
                "next_state_is_in_battle": next_state_is_in_battle.value , 
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
            for index , level in enumerate(next_state_party_levels):
                info[f'pokemon_{ index +1 }_level'] = level
            try:
                for index , pokemon_id in enumerate(ram_map.get_party_pokemon_id(self.game)):
                    info[f'pokemon_{ index +1 }_id'] = pokemon_id
            except Exception as e:
                print(f"Error: {e}")
                print(f"ram_map.get_party_pokemon_id(self.game): {ram_map.get_party_pokemon_id(self.game)}")
            for index , pokemon_id in enumerate(ram_map.get_opponent_party_pokemon_id(self.game)):
                info[f'opponent_pokemon_{ index +1 }_id'] = pokemon_id

        if self.verbose:
            print(
                f'time: {self.time}',
                f'exploration reward: {exploration_reward}',
                f'death: {death_reward}',
                f'op_level: {opponent_level_reward}',
                f'badges reward: {badges_reward}',
                f'event reward: {event_reward}',
                f'money: {money}',
                f'ai reward: {reward}',
                f"In a trainer battle: {current_state_is_in_battle}",
                f"Gym Leader Music is playing: {ram_map.check_if_gym_leader_music_is_playing(self.game)}",
                f'Info: {info}',
            )
        # Observation , reward, done, info
        assert isinstance(next_state_party_levels, list), f"next_state_party_levels: {next_state_party_levels}"
        observation = {
            'screen': self.render()[::2, ::2],
            "party_size": next_state_party_size ,
            "player_row": row,
            "player_column": column,
            "total_party_hit_point" : ram_map.total_party_hit_point(self.game),
            "total_party_max_hit_point" : ram_map.total_party_max_hit_point(self.game),
            "party_health_ratio": ram_map.party_health_ratio(self.game),
            "total_party_level": sum(next_state_party_levels),
            "each_pokemon_level": np.array(next_state_party_levels, dtype=np.uint8),
            "type_of_battle": next_state_is_in_battle.value, # 0 means not in battle, 1 means wild battle, 2 means trainer battle
            "player_pokemon_party_id": ram_map.get_party_pokemon_id(self.game),
            "opponent_pokemon_party_id": ram_map.get_opponent_party_pokemon_id(self.game),
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
    # Code snippet https://github.com/thatguy11325/pokemonred_puffer/blob/main/pokemonred_puffer/environment.py
    def find_neighboring_sign(self, sign_id, player_direction, player_x, player_y) -> bool:
        sign_y = self.game.get_memory_value(0xD4B1 + (2 * sign_id))
        sign_x = self.game.get_memory_value(0xD4B1 + (2 * sign_id + 1))

        # Check if player is facing the sign (skip sign direction)
        # 0 - down, 4 - up, 8 - left, 0xC - right
        # We are making the assumption that a player will only ever be 1 space away
        # from a sign
        return (
            (player_direction == 0 and sign_x == player_x and sign_y == player_y + 1)
            or (player_direction == 4 and sign_x == player_x and sign_y == player_y - 1)
            or (player_direction == 8 and sign_y == player_y and sign_x == player_x - 1)
            or (player_direction == 0xC and sign_y == player_y and sign_x == player_x + 1)
        )
    def init_mem(self):
        # Maybe I should preallocate a giant matrix for all map ids
        # All map ids have the same size, right?
        self.seen_coords: set = set()
        self.seen_coords_since_blackout = set([])
        # self.seen_global_coords = np.zeros(GLOBAL_MAP_SHAPE)
        self.seen_map_ids = np.zeros(256)
        self.seen_map_ids_since_blackout = set([])

        self.seen_npcs = {}
        self.seen_npcs_since_blackout = set([])

        self.seen_hidden_objs = {}

        self.cut_coords = {}
        self.cut_tiles = set([])
        self.cut_state = deque(maxlen=3)

        self.seen_start_menu = 0
        self.seen_pokemon_menu = 0
        self.seen_stats_menu = 0
        self.seen_bag_menu = 0
        self.seen_cancel_bag_menu = 0
        
        self.visited_pokecenter_list = []
    def find_neighboring_npc(self, npc_id, player_direction, player_x, player_y) -> int:
        npc_y = self.game.get_memory_value(0xC104 + (npc_id * 0x10))
        npc_x = self.game.get_memory_value(0xC106 + (npc_id * 0x10))

        # Check if player is facing the NPC (skip NPC direction)
        # 0 - down, 4 - up, 8 - left, 0xC - right
        if (
            (player_direction == 0 and npc_x == player_x and npc_y > player_y)
            or (player_direction == 4 and npc_x == player_x and npc_y < player_y)
            or (player_direction == 8 and npc_y == player_y and npc_x < player_x)
            or (player_direction == 0xC and npc_y == player_y and npc_x > player_x)
        ):
            # Manhattan distance
            return abs(npc_y - player_y) + abs(npc_x - player_x)

        return False
    def check_if_party_has_cut(self) -> bool:
        party_size = self.read_m(PARTY_SIZE)
        for i in [0xD16B, 0xD197, 0xD1C3, 0xD1EF, 0xD21B, 0xD247][:party_size]:
            for m in range(4):
                if self.game.get_memory_value(i + 8 + m) == 15:
                    return True
        return False
    def read_m(self, addr):
        return self.game.get_memory_value(addr)

    def read_bit(self, addr, bit: int) -> bool:
        # add padding so zero will read '0b100000000' instead of '0b0'
        return bin(256 + self.read_m(addr))[-bit - 1] == "1"

    def read_event_bits(self) -> list[int]:
        return [
            int(bit)
            for i in range(EVENT_FLAGS_START, EVENT_FLAGS_END)
            for bit in f"{self.read_m(i):08b}"
        ]
    def get_badges(self):
        return self.bit_count(self.read_m(0xD356))
    # built-in since python 3.10
    def bit_count(self, bits):
        return bin(bits).count("1")
    def set_perfect_iv_dvs(self):
        party_size = self.read_m(PARTY_SIZE)
        for i in [0xD16B, 0xD197, 0xD1C3, 0xD1EF, 0xD21B, 0xD247][:party_size]:
            for m in range(12):  # Number of offsets for IV/DV
                self.game.set_memory_value(i + 17 + m, 0xFF)
    
    def get_last_pokecenter_id(self):
        
        last_pokecenter = self.read_m(0xD719)
        # will throw error if last_pokecenter not in pokecenter_ids, intended
        if last_pokecenter == 0:
            # no pokecenter visited yet
            return -1
        if last_pokecenter not in self.pokecenter_ids:
            print(f'\nERROR: last_pokecenter: {last_pokecenter} not in pokecenter_ids')
            return -1
        else:
            return self.pokecenter_ids.index(last_pokecenter)
    def update_visited_pokecenter_list(self) -> Literal[1, 0]:
        last_pokecenter_id = self.get_last_pokecenter_id()
        if last_pokecenter_id != -1 and last_pokecenter_id not in self.visited_pokecenter_list:
            self.visited_pokecenter_list.append(last_pokecenter_id)
            return 1
        return 0
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
    
    return  (x - min_x) / (max_x - min_x)