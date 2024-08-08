from collections import deque
from pdb import set_trace as T
from typing import Literal
from gymnasium import spaces
import numpy as np
import os
import io
from pokegym import game_state, observation
from pokegym.game_state import External_Game_State, Internal_Game_State
from pokegym.reward import Reward
from skimage.transform import resize

from pokegym.pyboy_binding import (ACTIONS, make_env, open_state_file,load_pyboy_state)
from pokegym import ram_map, game_map 
from pokegym.global_map import GLOBAL_MAP_SHAPE , local_to_global
from rich import print as print
PIXEL_VALUES = np.array([0, 85, 153, 255], dtype=np.uint8)

EVENT_FLAGS_START = 0xD747
EVENT_FLAGS_END = (
    0xD7F6  # 0xD761 # 0xD886 temporarily lower event flag range for obs input
)

WCUTTILE = 0xCD4D # 61 if Cut used; 0 default. resets to default on map_n change or battle.
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
        window = "SDL2"
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
        input_events = env.game._handle_events()
        env.render()

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
            state_path=None, headless=True, quiet=False, 
            window = "null" , 
            **kwargs):
        '''Creates a PokemonRed environment'''
        random_starter_pokemon:bool = kwargs.get('random_starter_pokemon', False)
        def determine_pyboy_game_state_file(random_starter_pokemon:bool = False):
            if random_starter_pokemon:
                    # pick a random number between 1 to 3
                import random
                random_number = random.randint(1, 3)
                if random_number == 1:
                    pyboy_game_state_path_file = __file__.rstrip('environment.py') + 'Bulbasaur_fast_text_no_battle_animations_fixed_battle.state'
                    return pyboy_game_state_path_file
                elif random_number == 2:
                    pyboy_game_state_path_file = __file__.rstrip('environment.py') + 'Charmander.state'
                    return pyboy_game_state_path_file
                elif random_number == 3:
                    pyboy_game_state_path_file = __file__.rstrip('environment.py') + 'Squirtle.state'
                    return pyboy_game_state_path_file
                else:
                    ValueError("random_starter_pokemon should be between 1 to 3")
            else:
                pyboy_game_state_path_file = __file__.rstrip('environment.py') + 'Bulbasaur_fast_text_no_battle_animations_fixed_battle.state'
                return pyboy_game_state_path_file
            return "None"
        if state_path is None:
            state_path = determine_pyboy_game_state_file(random_starter_pokemon)
            assert os.path.exists(state_path), f"State file {state_path} does not exist , {T()}"

        # Make the environment
        self.game = make_env(rom_path, headless, quiet , window = window)
        self.initial_states = open_state_file(state_path)
        self.headless = headless
        self.action_space = spaces.Discrete(len(ACTIONS))

    '''
    You can view this where the update of observation is done because in every step 
    the render is called which display the observation 
    '''
    # https://github.com/thatguy11325/pokemonred_puffer/blob/b5b0d0960661ed96e05b08fdbafeb9e8ba803ffa/pokemonred_puffer/environment.py#L582https://github.com/thatguy11325/pokemonred_puffer/blob/b5b0d0960661ed96e05b08fdbafeb9e8ba803ffa/pokemonred_puffer/environment.py#L582
    def render(self):
        # botsupport_manager has been removed. You can now find the API calls directly on the PyBoy object. And in case of tilemap_background, tilemap_window and screen, they are all properties instead of functions:
        return self.game.screen.ndarray

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
            perfect_ivs:bool = True,
            reward_for_increasing_the_highest_pokemon_level_in_the_team_by_battle_coef:float =1.0 ,
            reward_for_entering_a_trainer_battle_coef:float = 1.0,
            negative_reward_for_wiping_out_coef:float = 1.0,
            negative_reward_for_entering_a_trainer_battle_lower_total_pokemon_level_coef:float = 1.0 , 
            reward_for_using_bad_moves_coef:float = 1.0 , 
            disable_wild_encounters:bool = False,
            reward_for_increasing_the_total_party_level:float = 1.0,
            reward_for_knocking_out_wild_pokemon_by_battle_coef:float = 1.0 , 
            reward_for_doing_new_events:float = 1.0,
            level_up_reward_threshold:int = 8 , 
            reward_for_finding_new_maps_coef:float = 1.0,
            reward_for_finding_higher_level_wild_pokemon_coef:float = 1.0,
            multiple_exp_gain_by_n:int = 6,
            reward_for_knocking_out_enemy_pokemon_in_trainer_party_coef:float = 1.0 , 
            set_enemy_pokemon_accuracy_to_zero = True , 
            add_random_moves_to_starter_pokemon = True,
            set_starter_pokemon_speed_values = 0, 
            set_enemy_pokemon_damage_calcuation_to_zero = True,
            **kwargs):
        self.random_starter_pokemon = kwargs.get("random_starter_pokemon", False)
        super().__init__(rom_path, state_path, headless, quiet, **kwargs)
        # https://github.com/xinpw8/pokegym/blob/d44ee5048d597d7eefda06a42326220dd9b6295f/pokegym/environment.py#L233
        self.verbose = verbose
        self.last_map = -1
        self.reset_count = 0
        self.perfect_ivs = perfect_ivs
        self.pokecenter_ids: list[int] = [0x01, 0x02, 0x03, 0x0F, 0x15, 0x05, 0x06, 0x04, 0x07, 0x08, 0x0A, 0x09]
        R, C = self.game.screen.ndarray.shape[0] , self.game.screen.ndarray.shape[1]
        self.two_bit = False
        self.action_freq = 24 #env_config.action_freq which is the frame skip
        self.seed = 1
        
        self.reduce_res = True
        # Obs space-related. TODO: avoid hardcoding?
        if self.reduce_res:
            self.screen_output_shape = (72, 80, 1)
        else:
            self.screen_output_shape = (144, 160, 1)
        self.observation_space = spaces.Dict({
            'screen': spaces.Box(
                low=0, high=255, dtype=np.uint8,
                shape=(R // 2, C // 2, 3),
            ),
            "visited_mask": spaces.Box(
                    low=0, high=255, shape=self.screen_output_shape, dtype=np.uint8
            ),
            "global_map": spaces.Box(
                    low=0, high=255, shape=self.screen_output_shape, dtype=np.uint8
            ),
            # Discrete is more apt, but pufferlib is slower at processing Discrete
            "direction": spaces.Box(low=0, high=4, shape=(1,), dtype=np.uint8),
            "x": spaces.Box(low=0, high=444, shape=(1,), dtype=np.float32),
            "y": spaces.Box(low=0, high=436, shape=(1,), dtype=np.float32),
            "map_id": spaces.Box(low=0, high=0xF7, shape=(1,), dtype=np.float32),
            "map_music_sound_bank": spaces.Box(low=0, high=3, shape=(1,), dtype=np.uint8),
            "map_music_sound_id": spaces.Box(low=0, high=84, shape=(1,), dtype=np.uint8),
            "party_size": spaces.Box(low = 1 , high = 6, shape=(1,), dtype=np.uint8),
            "each_pokemon_level": spaces.Box(low = 1, high = 100, shape=(6,), dtype=np.uint8),
            "total_party_level": spaces.Box(low = 0, high = 100, shape=(1,), dtype=np.uint8),
            "battle_stats": spaces.Box(low = 0, high = 4, shape=(1,), dtype=np.uint8),
            "battle_result": spaces.Box(low = 0, high = 4, shape=(1,), dtype=np.uint8),
            "number_of_turns_in_current_battle": spaces.Box(low = 0, high = 255, shape=(1,), dtype=np.uint8),
            "each_pokemon_health_points": spaces.Box(low = 0, high = 99, shape=(6,), dtype=np.uint8),
            "each_pokemon_max_health_points": spaces.Box(low = 0, high = 99, shape=(6,), dtype=np.uint8),
            "total_party_health_points": spaces.Box(low = 0, high = 99, shape=(1,), dtype=np.uint8),
            "total_party_max_hit_points": spaces.Box(low = 0, high = 1, shape=(1,), dtype=np.float32),
            "low_health_alarm": spaces.Box(low = 0, high = 1, shape=(1,), dtype=np.uint8),
            "opponent_pokemon_levels": spaces.Box(low = 0, high = 100, shape=(6,), dtype=np.uint8),
            "total_number_of_items": spaces.Box(low = 0, high = 64, shape=(1,), dtype=np.uint8),
            "money": spaces.Box(low = 0, high = 999999, shape=(1,), dtype=np.uint32),
            "player_selected_move_id": spaces.Box(low = 0, high = 166, shape=(1,), dtype=np.uint8),
            "enemy_selected_move_id": spaces.Box(low = 0, high = 166, shape=(1,), dtype=np.uint8),
            #"total_number_of_unique_moves_in_the_teams": spaces.Box(low = 0, high = 24, shape=(1,), dtype=np.uint8)
            "player_xp": spaces.Box(low=0, high=1, shape=(6, ), dtype=np.float32),
            "total_player_lineup_xp": spaces.Box(low=0, high=250000, shape=(1,), dtype=np.float32),
            "total_pokemon_seen": spaces.Box(low=0, high=152, shape=(1,), dtype=np.uint8),
            "pokemon_seen_in_the_pokedex": spaces.Box(low=0, high=1, shape=(19,), dtype=np.uint8),
            "byte_representation_of_caught_pokemon_in_the_pokedex": spaces.Box(low=0, high=1, shape=(19,), dtype=np.uint8),
            
            # Player
            
            "pokemon_party_move_id": spaces.Box(low=0, high=255, shape=(24,), dtype=np.uint8),
            
            ## POkemon
            "each_pokemon_pp": spaces.Box(low=0, high=40, shape=(24,), dtype=np.uint8),
            
            
            ### Trainer Opponents
            "enemy_trainer_pokemon_hp": spaces.Box(low=0, high=705, shape=(6,), dtype=np.float32) , 
            
            ### Wild Opponents I think
            "enemy_pokemon_hp": spaces.Box(low=0, high=705, shape=(1,), dtype=np.float32) , 
            
            ## Events
            "total_events_that_occurs_in_game": spaces.Box(low=0, high=2560, shape=(1,), dtype=np.float32),
            "time": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "enemy_monster_actually_catch_rate": spaces.Box(low=0 , high = 1 , shape=(1,), dtype=np.float32) , 
            
            # World
            "last_black_out_map_id": spaces.Box(low=0, high=150, shape=(1,), dtype=np.float32),
            # Battle Stuff
            "player_current_monster_stats_modifier_attack": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "player_current_monster_stats_modifier_defense": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "player_current_monster_stats_modifier_speed": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "player_current_monster_stats_modifier_special": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "player_current_monster_stats_modifier_accuracy": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            
            "enemy_current_pokemon_stats_modifier_attack": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "enemy_current_pokemon_stats_modifier_defense": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "enemy_current_pokemon_stats_modifier_speed": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "enemy_current_pokemon_stats_modifier_special": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "enemy_current_pokemon_stats_modifier_accuracy": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "enemy_current_pokemon_stats_modifier_evasion": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "enemy_current_move_effect": spaces.Box(low=0, high=56, shape=(1,), dtype=np.uint8),
            "enemy_pokemon_move_power" : spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "enemy_pokemon_move_type" : spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "enemy_pokemon_move_accuracy": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32) , 
            "enemy_pokemon_move_max_pp": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
        })
        self.display_info_interval_divisor = kwargs.get("display_info_interval_divisor", 1)
        #print(f"self.display_info_interval_divisor: {self.display_info_interval_divisor}")
        self.max_episode_steps = kwargs.get("max_episode_steps", 2048)
        self.reward_for_increase_pokemon_level_coef = kwargs.get("reward_for_increase_pokemon_level_coef", 1.1)
        self.reward_for_explore_unique_coor_coef = kwargs.get("reward_for_explore_unique_coor_coef", 0)
        self.reward_for_increasing_the_highest_pokemon_level_in_the_team_by_battle_coef:float = reward_for_increasing_the_highest_pokemon_level_in_the_team_by_battle_coef
        self.reward_for_entering_a_trainer_battle_coef:float = reward_for_entering_a_trainer_battle_coef
        self.negative_reward_for_wiping_out_coef:float = negative_reward_for_wiping_out_coef
        self.negative_reward_for_entering_a_trainer_battle_lower_total_pokemon_level_coef:float = negative_reward_for_entering_a_trainer_battle_lower_total_pokemon_level_coef
        self.reward_for_using_bad_moves_coef = reward_for_using_bad_moves_coef
        self. reward_for_increasing_the_total_party_level = reward_for_increasing_the_total_party_level
        self.reward_for_knocking_out_wild_pokemon_by_battle_coef = reward_for_knocking_out_wild_pokemon_by_battle_coef
        self.level_up_reward_threshold = level_up_reward_threshold
        self.reward_for_doing_new_events = reward_for_doing_new_events
        self.reward_for_finding_new_maps_coef = reward_for_finding_new_maps_coef
        self. reward_for_finding_higher_level_wild_pokemon_coef = reward_for_finding_higher_level_wild_pokemon_coef
        self.multiple_exp_gain_by_n = multiple_exp_gain_by_n
        self.reward_for_knocking_out_enemy_pokemon_in_trainer_party_coef = reward_for_knocking_out_enemy_pokemon_in_trainer_party_coef
        self.set_enemy_pokemon_accuracy_to_zero = set_enemy_pokemon_accuracy_to_zero
        self.add_random_moves_to_starter_pokemon = add_random_moves_to_starter_pokemon
        self.set_starter_pokemon_speed_values = set_starter_pokemon_speed_values
        
        self.random_wild_grass_pokemon_encounter_rate_per_env = kwargs.get("random_wild_grass_pokemon_encounter_rate_per_env", False)
        self.go_explored_list_of_episodes:list  = list()
        self.disable_wild_encounters = disable_wild_encounters
        self.set_enemy_pokemon_damage_calcuation_to_zero = set_enemy_pokemon_damage_calcuation_to_zero
        self.register_hooks()
        
        self.probaility_wild_grass_pokemon_encounter_rate_per_env = -1
        if self.random_wild_grass_pokemon_encounter_rate_per_env:
            import random
            self.probaility_wild_grass_pokemon_encounter_rate_per_env = random.randint(0 , 255)
        self.first = True
        self.set_of_map_ids_explored = set()

    def fresh_game_state(self):
        state = io.BytesIO()
        state.seek(0)
        self.game.save_state(state)
        return state
    def reset(self, seed=None,  options = None ):
        '''Resets the game to the previous save steps. Seeding is NOT supported'''
        import random
        if self.first:
            self.external_game_state = External_Game_State()
            self.init_mem()
            self.explore_map = np.zeros(GLOBAL_MAP_SHAPE, dtype=np.float32)
            self.counts_map = np.zeros((444, 436)) # to solve the map
            load_pyboy_state(self.game, self.load_last_state()) # load a saved state
            self.reset_count += 1
            self.time = 0
            self.go_explored_list_of_episodes.append(
                {
                    "external_game_state": self.external_game_state ,
                    "explore_map": self.explore_map,
                    "seen_npcs": self.seen_npcs,
                    "counts_map": self.counts_map,
                    "game_state": self.fresh_game_state(),
                    "reset_count" : self.reset_count ,
                }
                
            ) 
            self.random_number = 0 # random.randint(0 , len(self.go_explored_list_of_episodes) - 1)        # Add Random move id if their is empty
            if self.add_random_moves_to_starter_pokemon:
                import random
                move_ids = ["wPartyMon1Moves"]
                move_pps = ["wPartyMon1PP"]
                print(f"Adding RnadomMOves ")
                for move_id in move_ids:
                    bank , addr = self.game.symbol_lookup(move_id)
                    bank_pp , addr_pp = self.game.symbol_lookup(move_pps[0])
                    for i in range(0, 5):
                        if self.game.memory[addr + i] == 0:
                            self.game.memory[addr + i] = random.randint(0 , 161)
                            self.game.memory[addr_pp + i] = 2
            if self.set_starter_pokemon_speed_values != 0:
                bank , addr = self.game.symbol_lookup("wPartyMon1Speed")
                self.game.memory[addr] = self.set_starter_pokemon_speed_values
        elif not self.first:
            self.go_explored_list_of_episodes.append(
                {
                    "external_game_state": self.external_game_state , 
                    "explore_map": self.explore_map,
                    "seen_npcs": self.seen_npcs,
                    "counts_map": self.counts_map,
                    "game_state": self.fresh_game_state(),
                    "reset_count" : self.reset_count +1 ,
                }
            )
            self.random_number = random.randint(0 , len(self.go_explored_list_of_episodes) - 1)
            self.external_game_state = self.go_explored_list_of_episodes[self.random_number]["external_game_state"]
            self.explore_map = self.go_explored_list_of_episodes[self.random_number]["explore_map"]
            self.seen_npcs = self.go_explored_list_of_episodes[self.random_number]["seen_npcs"]
            self.counts_map = self.go_explored_list_of_episodes[self.random_number]["counts_map"]
            load_pyboy_state(self.game, self.go_explored_list_of_episodes[self.random_number]["game_state"])
            #self.reset_count = self.go_explored_list_of_episodes[random_number]["reset_count"]
            self.reset_count  = self.reset_count + 1
            self.time = 0 
            assert self.time ==0 , T()
            
            
        self.first = False  
        self.time = 0 
        assert self.time == 0 , T() # Please it will fuck you up becaus self.time >= max epsiodes step this will cause a huge problem in this erro o th edetails on the action 
        
        #load_pyboy_state(self.game, self.initial_state)
        """Resets the game. Seeding is NOT supported"""
        # https://github.com/xinpw8/pokegym/blob/baseline_0.6/pokegym/environment.py
        
        internal_game_state: Internal_Game_State = Internal_Game_State(self.game)
        observation_game_state = observation.Observation(internal_game_state , 0, self.max_episode_steps)
        
        

        old_observation = self._get_obs()
        old_observation.update(observation_game_state.to_json())
        return old_observation, {}
    
    def register_hooks(self):
        #if self.setup_make_sure_never_reach_zero()
        #self.setup_multiple_exp_gain_by_n()
        if self.disable_wild_encounters:
            self.setup_disable_wild_encounters()
        if self.set_enemy_pokemon_damage_calcuation_to_zero:
            self.setup_set_enemy_pokemon_damage_calcuation_to_zero()
        #self.Calculate Stat Experience:
        self.calculate_stat_experience()
        if self.set_enemy_pokemon_accuracy_to_zero:
            self.setup_set_enemy_accuracy_to_zero()
    def setup_set_enemy_accuracy_to_zero(self):
        bank, addr = self.game.symbol_lookup("MoveHitTest.calcHitChance")
        self.game.hook_register(
            bank,
            addr,
            self.set_enemy_accuracy_to_zero_hook,
            None,
        )
    def set_enemy_accuracy_to_zero_hook(self):
        self.game.memory[self.game.symbol_lookup("wEnemyMoveAccuracy")[1]] = 0
    def calculate_stat_experience(self):
        bank, addr = self.game.symbol_lookup("GainExperience.partyMonLoop")
        self.game.hook_register(
            bank,
            addr,
            self.calculate_stat_experience_hook,
            None,
        )
    def calculate_stat_experience_hook(self):
        x = 2
        
    def setup_set_enemy_pokemon_damage_calcuation_to_zero(self):
        bank, addr = self.game.symbol_lookup("GetDamageVarsForEnemyAttack")
        self.game.hook_register(
            bank,
            addr,
            self.set_enemy_pokemon_damage_calcuation_to_zero_hook,
            None,
        )
    def set_enemy_pokemon_damage_calcuation_to_zero_hook(self):
        self.game.memory[self.game.symbol_lookup("wEnemyMovePower")[1]] = 0
    def setup_multiple_exp_gain_by_n(self):
        #bank ,  addr = self.game.symbol_lookup("GainExperience.partyMonLoop")
        bank , addr = self.game.symbol_lookup("GainExperience.next")
        self.game.hook_register(
            bank , 
            addr  , 
            self.multiple_exp_gain_by_n_hook,
            None,
        )
    def  multiple_exp_gain_by_n_hook(self):
        value = self.game.memory[self.game.symbol_lookup("wExpAmountGained")]
        assert value >= 0 , T()
        self.game.memory[self.game.symbol_lookup("wExpAmountGained")[1]] = 255
        self.game.memory[self.game.symbol_lookup("wGainBoostedExp")[1]] = 255
        assert self.game.memory[self.game.symbol_lookup("wExpAmountGained")[1]] == 255 , T()
    def setup_make_sure_never_reach_zero(self):
        from pokegym.ram_map import HP_ADDR , MAX_HP_ADDR
        from pokegym.ram_reader.red_memory_battle_stats import PLAYER_CURRENT_BATTLE_POKEMON_CURRENT_HP , PLAYER_CURRENT_BATTLE_POKEMON_MAX_HP
        bank , addr = self.game.symbol_lookup("wBattleMonHP")
        self.game.hook_register(
            bank,
            addr,
            self.make_sure_never_reach_zero,
            None,
        )
        bank , addr = self.game.symbol_lookup("wBattleMonMaxHP")
        self.game.hook_register(
            bank,
            addr,
            self.make_sure_never_reach_zero,
            None,
        )
    def make_sure_never_reach_zero(self):
        self.game.memory[self.game.symbol_lookup("wBattleMonHP")[1]] = 1000
        assert isinstance(self.game.memory[self.game.symbol_lookup("wBattleMonHP")[1]] == 1 , int) , T()
        assert 0xD015 + 1 == 0xD016 , T()
        assert self.game.symbol_lookup("wBattleMonMaxHP")[1] == 0xD016 , T()
        self.game.memory[self.game.symbol_lookup("wBattleMonMaxHP")[1] + 1] = 1
        self.game.memory[self.game.symbol_lookup("wBattleMonMaxHP")[1] + 1] = 1000
        
    def setup_disable_wild_encounters(self):
        bank, addr = self.game.symbol_lookup("TryDoWildEncounter.gotWildEncounterType")
        self.game.hook_register(
            bank,
            addr + 8,
            self.disable_wild_encounter_hook,
            None,
        )
    def disable_wild_encounter_hook(self, *args, **kwargs):
        self.game.memory[self.game.symbol_lookup("wRepelRemainingSteps")[1]] = 0xFF
        # https://bulbapedia.bulbagarden.net/wiki/Repel
        self.game.memory[self.game.symbol_lookup("wCurEnemyLVL")[1]] = 0x01 # In Generation I, and from Generation VI onwards, this applies to wild Pokémon with a lower level than the first member of the party.

    
    def get_game_coords(self):
            return (self.read_m(0xD362), self.read_m(0xD361), self.read_m(0xD35E))
    def update_seen_coords(self):
        x_pos, y_pos, map_n = self.get_game_coords()
        #self.seen_coords.add((x_pos, y_pos, map_n))
        self.explore_map[local_to_global(y_pos, x_pos, map_n)] = 1
        # self.seen_global_coords[local_to_global(y_pos, x_pos, map_n)] = 1
    def render(self):
        # (144, 160, 3)
        try:
            #game_pixels_render = np.expand_dims(self.screen.screen_ndarray()[:, :, 1], axis=-1)
            game_pixels_render = np.expand_dims(
                self.game.screen.ndarray[:, :, 1], axis=-1
            )
        except Exception as e:
            raise  e

        if self.reduce_res:
            game_pixels_render = game_pixels_render[::2, ::2, :] # # x, y, map_id
            # game_pixels_render = skimage.measure.block_reduce(game_pixels_render, (2, 2, 1), np.min)

        # place an overlay on top of the screen greying out places we haven't visited
        # first get our location
        player_x, player_y, map_n = self.get_game_coords()

        # player is centered at 68, 72 in pixel units
        # 68 -> player y, 72 -> player x
        # guess we want to attempt to map the pixels to player units or vice versa
        # Experimentally determined magic numbers below. Beware
        # visited_mask = np.zeros(VISITED_MASK_SHAPE, dtype=np.float32)
        visited_mask = np.zeros_like(game_pixels_render)
        """
        if self.taught_cut:
            cut_mask = np.zeros_like(game_pixels_render)
        else:
            cut_mask = np.random.randint(0, 255, game_pixels_render.shape, dtype=np.uint8)
        """
        # If not in battle, set the visited mask. There's no reason to process it when in battle
        scale = 2 if self.reduce_res else 1
        if self.read_m(0xD057) == 0:
            for y in range(-72 // 16, 72 // 16):
                for x in range(-80 // 16, 80 // 16):
                    # y-y1 = m (x-x1)
                    # map [(0,0),(1,1)] -> [(0,.5),(1,1)] (cause we dont wnat it to be fully black)
                    # y = 1/2 x + .5
                    # current location tiles - player_y*8, player_x*8
                    """
                    visited_mask[y, x, 0] = self.seen_coords.get(
                        (
                            player_x + x + 1,
                            player_y + y + 1,
                            map_n,
                        ),
                        0.15,
                    )
                    """

                    visited_mask[
                        (16 * y + 76) // scale : (16 * y + 16 + 76) // scale,
                        (16 * x + 80) // scale : (16 * x + 16 + 80) // scale,
                        :,
                    ] = int(
                        (
                            (player_x + x + 1, player_y + y + 1, map_n) in self.external_game_state.seen_coords
                        )
                        * 255
                    )

                    """
                    if self.taught_cut:
                        cut_mask[
                            16 * y + 76 : 16 * y + 16 + 76,
                            16 * x + 80 : 16 * x + 16 + 80,
                            :,
                        ] = int(
                            255
                            * (
                                self.cut_coords.get(
                                    (
                                        player_x + x + 1,
                                        player_y + y + 1,
                                        map_n,
                                    ),
                                    0,
                                )
                            )
                        )
                        """
        """
        gr, gc = local_to_global(player_y, player_x, map_n)
        visited_mask = (
            255
            * np.repeat(
                np.repeat(self.seen_global_coords[gr - 4 : gr + 5, gc - 4 : gc + 6], 16, 0), 16, -1
            )
        ).astype(np.uint8)
        visited_mask = np.expand_dims(visited_mask, -1)
        """

        global_map = (255 * resize(self.explore_map, game_pixels_render.shape, anti_aliasing=False)).astype(np.uint8)
        assert game_pixels_render.shape == visited_mask.shape == global_map.shape , T(header=f"game_pixels_render.shape: {game_pixels_render.shape}, visited_mask.shape: {visited_mask.shape}, global_map.shape: {global_map.shape}")

        if self.two_bit:
            game_pixels_render = (
                (
                    np.digitize(
                        game_pixels_render.reshape((-1, 4)), PIXEL_VALUES, right=True
                    ).astype(np.uint8)
                    << np.array([6, 4, 2, 0], dtype=np.uint8)
                )
                .sum(axis=1, dtype=np.uint8)
                .reshape((-1, game_pixels_render.shape[1] // 4, 1))
            )
            visited_mask = (
                (
                    np.digitize(
                        visited_mask.reshape((-1, 4)),
                        np.array([0, 64, 128, 255], dtype=np.uint8),
                        right=True,
                    ).astype(np.uint8)
                    << np.array([6, 4, 2, 0], dtype=np.uint8)
                )
                .sum(axis=1, dtype=np.uint8)
                .reshape(game_pixels_render.shape)
                .astype(np.uint8)
            )
            global_map = (
                (
                    np.digitize(
                        global_map.reshape((-1, 4)),
                        np.array([0, 64, 128, 255], dtype=np.uint8),
                        right=True,
                    ).astype(np.uint8)
                    << np.array([6, 4, 2, 0], dtype=np.uint8)
                )
                .sum(axis=1, dtype=np.uint8)
                .reshape(game_pixels_render.shape)
            )
        return {
            "screen": game_pixels_render,
            "visited_mask": visited_mask,
            "global_map": global_map,
        }
    def run_action_on_emulator(self, action):
        #self.action_hist[action] += 1
        # press button then release after some steps
        # TODO: Add video saving logic
        try:
            self.game.send_input(ACTIONS[action].PRESS)# Press
        except Exception as e:
            print(e)
            print(action)
            print(ACTIONS)
            raise Exception("Action not found")
        
        self.game.send_input(ACTIONS[action].RELEASE, delay=8) # Release
        self.game.tick(self.action_freq, render=True)
    def step(self, action):
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
        
        
        # Previous own a gym badge
        prev_badges_one  = ram_map.check_if_player_has_gym_one_badge(self.game)
        
        # current opponent pokemon health points
        prev_seen_npcs:int  = sum(self.seen_npcs.values())
        
        
        state_internal_game: game_state.Internal_Game_State = Internal_Game_State( game = self.game)
        self.run_action_on_emulator(action)
        self.time += 1
        next_state_internal_game: game_state.Internal_Game_State = Internal_Game_State( game = self.game)
        self.external_game_state.update( game = self.game , current_interngal_game_state = state_internal_game ,  next_next_internal_game_state  = next_state_internal_game)
        reward_for_stateless_class: Reward = Reward( state_internal_game, next_state_internal_game, self.external_game_state , 
                                                    self.reward_for_increase_pokemon_level_coef , 
                                                    reward_for_increasing_the_highest_pokemon_level_in_the_team_by_battle_coef = self.reward_for_increasing_the_highest_pokemon_level_in_the_team_by_battle_coef , 
                                                    reward_for_entering_a_trainer_battle_coef = self.reward_for_entering_a_trainer_battle_coef , 
                                                    negative_reward_for_wiping_out_coef = self.negative_reward_for_wiping_out_coef,
                                                    reward_for_explore_unique_coor_coef = self.reward_for_explore_unique_coor_coef,
                                                    negative_reward_for_entering_a_trainer_battle_lower_total_pokemon_level_coef = self.negative_reward_for_entering_a_trainer_battle_lower_total_pokemon_level_coef,
                                                    reward_for_using_bad_moves_coef = self.reward_for_using_bad_moves_coef , 
                                                    reward_for_knocking_out_wild_pokemon_by_battle_coef = self.reward_for_knocking_out_wild_pokemon_by_battle_coef ,
                                                    reward_for_doing_new_events = self.reward_for_doing_new_events,
                                                    reward_for_increasing_the_total_party_level = self.reward_for_increasing_the_total_party_level,
                                                    reward_for_finding_higher_level_wild_pokemon_coef = self.reward_for_finding_higher_level_wild_pokemon_coef,
                                                    reward_for_knocking_out_enemy_pokemon_in_trainer_party_coef = self.reward_for_knocking_out_enemy_pokemon_in_trainer_party_coef , 
                                                    reward_for_finding_new_maps_coef = self.reward_for_finding_new_maps_coef
                                                    )
        # Seen Coordinate
        self.update_seen_coords()
        ### Cut and Talking to NPCS
        
        # Cut check
        # 0xCFC6 - wTileInFrontOfPlayer
        # 0xCFCB - wUpdateSpritesEnabled
        if self.read_m(0xD057) == 0:
            if False :#self.taught_cut:
                player_direction = self.game.memory[0xC109]
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
                        self.game.memory[0xCFC6],
                        self.game.memory[0xCFCB],
                        self.game.memory[0xCD6A],
                        self.game.memory[0xD367],
                        self.game.memory[0xD125],
                        self.game.memory[0xCD3D],
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
            if self.game.memory[0xCFC4]:
                # check if we are talking to a hidden object:
                player_direction = self.game.memory[0xC109]
                player_y_tiles = self.game.memory[0xD361]
                player_x_tiles = self.game.memory[0xD362]
                if (
                    self.game.memory[0xCD3D] != 0x0
                    and self.game.memory[0xCD3E] != 0x0
                ):
                    # add hidden object to seen hidden objects
                    self.seen_hidden_objs[
                        (
                            self.game.memory[0xD35E],
                            self.game.memory[0xCD3F],
                        )
                    ] = 1
                elif any(
                    self.find_neighboring_sign(
                        sign_id, player_direction, player_x_tiles, player_y_tiles
                    )
                    for sign_id in range(self.game.memory[0xD4B0])
                ):
                    pass
                else:
                    # get information for player
                    player_y = self.game.memory[0xC104]
                    player_x = self.game.memory[0xC106]
                    # get the npc who is closest to the player and facing them
                    # we go through all npcs because there are npcs like
                    # nurse joy who can be across a desk and still talk to you

                    # npc_id 0 is the player
                    npc_distances = (
                        (
                            self.find_neighboring_npc(npc_id, player_direction, player_x, player_y),
                            npc_id,
                        )
                        for npc_id in range(1, self.game.memory[0xD4E1])
                    )
                    npc_candidates = [x for x in npc_distances if x[0]]
                    if npc_candidates:
                        _, npc_id = min(npc_candidates, key=lambda x: x[0])
                        self.seen_npcs[(self.game.memory[0xD35E], npc_id)] = 1
                        self.seen_npcs_since_blackout.add(
                            (self.game.memory[0xD35E], npc_id)
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
        
        
        self.update_heat_map(row, column, map_n)



        # gym
        # Badge reward
        badges_reward = 0
        if not prev_badges_one and  ram_map.check_if_player_has_gym_one_badge(self.game):
            badges_reward += 16

        # Money Reward
        next_state_money = ram_map.money(self.game)
        assert next_state_money >= 0 and next_state_money <= 999999, f"next_state_money: {next_state_money}"
        normalize_gain_of_new_money_reward = normalize_value(next_state_money - current_state_money, -999999, 999999, -1, 1)
        normalize_gain_of_new_money_reward = max(0 , normalize_gain_of_new_money_reward)
        normalize_gain_of_new_money_reward = max(0 , normalize_gain_of_new_money_reward)
        if next_state_money - current_state_money == 0 and normalize_gain_of_new_money_reward == .5:
            assert False, f"next_state_money: {next_state_money} current_state_money: {current_state_money} and the normalize_gain_of_new_money_reward is {normalize_gain_of_new_money_reward}"
        assert normalize_gain_of_new_money_reward >=  ( -1.0 - 1e5) and normalize_gain_of_new_money_reward <= 1.0, f"normalize_gain_of_new_money_reward: {normalize_gain_of_new_money_reward} the current state money is {current_state_money} and the next state money is {next_state_money}"
        
        
        
        # Seen Pokemon
        next_state_pokemon_seen = ram_map.pokemon_seen(self.game)
        reward_the_agent_seing_new_pokemon = next_state_pokemon_seen - current_state_pokemon_seen
        assert reward_the_agent_seing_new_pokemon == 0 or reward_the_agent_seing_new_pokemon == 1 or reward_the_agent_seing_new_pokemon==3, f"reward_the_agent_seing_new_pokemon: {reward_the_agent_seing_new_pokemon}"
        

        
        
        
        # Total item count
        item_count = ram_map.total_items(self.game)
        
        # total hm count
        hm_count = ram_map.total_hm_party_has(self.game)
        
        # number of hm moves my pokemon party has
        total_number_hm_moves_that_my_pokemon_party_has = ram_map.total_hm_party_has(self.game)
        
        # Reward the agent increase the health ratio health party by healing only not by adding a new pokemon
        next_health_ratio = ram_map.party_health_ratio(self.game)
        assert next_health_ratio >= 0 and next_health_ratio <= 1, f"next_health_ratio: {next_health_ratio}"
        
        
       
        if self.perfect_ivs:
            self.set_perfect_iv_dvs()
        
        next_seen_npcs = sum(self.seen_npcs.values())
        reward_seeen_npcs:int  = next_seen_npcs - prev_seen_npcs
        assert next_seen_npcs >= prev_seen_npcs
        assert reward_seeen_npcs == 1 or reward_seeen_npcs == 0, T()
        
         
        

        
        reward: float =  (
                + reward_the_agent_seing_new_pokemon 
                + badges_reward 
                + normalize_gain_of_new_money_reward
                +  reward_seeen_npcs  
        )
        reward += reward_for_stateless_class.total_reward()
    
        self.external_game_state.post_reward_update(next_state_internal_game , current_internal_game_state = state_internal_game , next_internal_game_state = next_state_internal_game)
        # Store the map id 
        if next_state_internal_game.map_id != state_internal_game.map_id and next_state_internal_game.map_id not in self.set_of_map_ids_explored and next_state_internal_game.map_id < 256 and state_internal_game.map_id < 256:
            self.go_explored_list_of_episodes.append(
                {
                    "external_game_state": self.external_game_state , 
                    "explore_map": self.explore_map,
                    "seen_npcs": self.seen_npcs,
                    "counts_map": self.counts_map,
                    "game_state": self.fresh_game_state(),
                    "reset_count" : self.reset_count ,
                }
            )
            self.set_of_map_ids_explored.add(next_state_internal_game.map_id)
            self.set_of_map_ids_explored.add(state_internal_game.map_id)
        '''
        # Store the new evnts 
        if next_state_internal_game.total_events_that_occurs_in_game > state_internal_game.total_events_that_occurs_in_game:
            self.go_explored_list_of_episodes.append(
                {
                    "external_game_state": self.external_game_state ,
                    "explore_map": self.explore_map,
                    "seen_npcs": self.seen_npcs,
                    "counts_map": self.counts_map,
                    "game_state": self.fresh_game_state(),
                    "reset_count" : self.reset_count +1 ,
                }
            )
        '''

        info = {}
        done = self.time >= self.max_episode_steps
        if self.time %  self.display_info_interval_divisor == 0 or done or self.time == 2 or self.time == 4 or self.time == 8:
            info = {
                'reward': {
                    'reward': reward,
                    'badges': badges_reward,
                    "seeing_new_pokemon": reward_the_agent_seing_new_pokemon,
                    "normalize_gain_of_new_money": normalize_gain_of_new_money_reward,
                    "reward_seeen_npcs": reward_seeen_npcs,
                    "reward_visiting_a_new_pokecenter": 0,
                },
                'time': self.time,
                "max_episode_steps": self.max_episode_steps,
                'badge_1': ram_map.check_if_player_has_gym_one_badge(self.game),
                "badges": self.get_badges(), # Fix it latter
                "npc": sum(self.seen_npcs.values()),
                "prev_npc": prev_seen_npcs,
                "next_npc": next_seen_npcs,
                "hidden_obj": sum(self.seen_hidden_objs.values()),
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
                "number_run_attempts": ram_map.get_number_of_run_attempts(self.game),
                "total_party_hit_point" : ram_map.total_party_hit_point(self.game),
                "total_party_max_hit_point" : ram_map.total_party_max_hit_point(self.game),
                "party_health_ratio": ram_map.party_health_ratio(self.game),
                "visited_pokemon_center": len(self.visited_pokecenter_list),
                "total_number_of_time_attempted_to_run": ram_map.get_number_of_run_attempts(self.game),
                "reset_count": self.reset_count,
                "current_state_is_in_battle": current_state_is_in_battle.value , 
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
                "size_of_total_number_of_episodes_in_store": len(self.go_explored_list_of_episodes) , 
                "ranmdom_number":self.random_number
            }
            info.update(next_state_internal_game.to_json())
            info.update(self.external_game_state.to_json())
            info["reward"].update(reward_for_stateless_class.to_json())
            #assert "reward_for_using_bad_moves" in info["reward"].keys(), f"info: {info}"
            ## add next state internal game state into the infos section
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
                f'badges reward: {badges_reward}',
                f'ai reward: {reward}',
                f"In a trainer battle: {current_state_is_in_battle}",
                f"Gym Leader Music is playing: {ram_map.check_if_gym_leader_music_is_playing(self.game)}",
                f'Info: {info}',
            )
        # Observation , reward, done, info
        
        observation_data_class = observation.Observation(next_state_internal_game_state = next_state_internal_game , time = self.time , max_episode_steps = self.max_episode_steps )
        old_observation = self._get_obs()
        old_observation.update(observation_data_class.get_obs())
        
        return old_observation, reward, done, done, info
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
        sign_y = self.game.memory[0xD4B1 + (2 * sign_id)]
        sign_x = self.game.memory[0xD4B1 + (2 * sign_id + 1)]

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
        self.seen_coords_since_blackout = set([])
        # self.seen_global_coords = np.zeros(GLOBAL_MAP_SHAPE)

        self.seen_npcs = {}
        self.seen_npcs_since_blackout = set([])

        self.seen_hidden_objs = {}



        self.seen_start_menu = 0
        self.seen_pokemon_menu = 0
        self.seen_stats_menu = 0
        self.seen_bag_menu = 0
        self.seen_cancel_bag_menu = 0
        
        self.visited_pokecenter_list = []
    def find_neighboring_npc(self, npc_id, player_direction, player_x, player_y) -> int:
        npc_y = self.game.memory[0xC104 + (npc_id * 0x10)]
        npc_x = self.game.memory[0xC106 + (npc_id * 0x10)]

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
    def read_m(self, addr):
        return self.game.memory[addr]

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
        PARTY_SIZE=PARTY_SIZE_ADDR = 0xD163
        party_size = self.read_m(PARTY_SIZE)
        for i in [0xD16B, 0xD197, 0xD1C3, 0xD1EF, 0xD21B, 0xD247][:party_size]:
            for m in range(12):  # Number of offsets for IV/DV
                self.game.memory[i + 17 + m] = 0xFF
    def _get_obs(self):
        player_x, player_y, map_n = ram_map.position(self.game)
        return {
            **self.render(),
            "direction": np.array(ram_map.get_player_direction(self.game) // 4, dtype=np.uint8), 
            "x": np.array(player_x, dtype=np.float32),
            "y": np.array(player_y, dtype=np.float32),
        }
    def get_last_pokecenter_list(self):
        pc_list = [0, ] * len(self.pokecenter_ids)
        last_pokecenter_id = self.get_last_pokecenter_id()
        if last_pokecenter_id != -1:
            pc_list[last_pokecenter_id] = 1
        return pc_list
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
    
    return  (x - min_x) / (max_x - min_x) * (b - a) + a