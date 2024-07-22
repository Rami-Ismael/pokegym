from dataclasses import dataclass , field , asdict
from typing import Any, List
from pokegym import ram_map
from rich import print
import numpy as np
from pdb import set_trace as T
@dataclass
class Observation:
    map_music_sound_bank: int = field(default_factory=int)
    map_music_sound_id: int = field(default_factory=int)
    party_size: int = field(default_factory=int)
    each_pokemon_level: List[float] = field(default_factory=list)
    total_party_level: float = field(default_factory=float)
    
    battle_stats: int = field(default_factory = int)  # Default to NOT_IN_BATTLE or any other default state
    battle_result: int = field(default_factory = int)  # Default to NOT_IN_BATTLE or any other default state
    number_of_turns_in_current_battle: int = field(default_factory=int)
   
    # Health Points 
    each_pokemon_health_points: List[float] = field(default_factory=list)
    each_pokemon_max_health_points: List[float] = field(default_factory=list)
    total_party_health_points: float = field(default_factory=float)
    total_party_max_hit_points: float = field(default_factory=float)
    low_health_alarm: int = field(default_factory=int)
    
    # Items
    #total_number_of_items: int = field(default_factory=int)
    money: int = field(default_factory=int)
    
    # Moves
    player_selected_move_id: int = field(default_factory=int)
    enemy_selected_move_id: int = field(default_factory=int)
    pokemon_party_move_id: list[int] = field(default_factory=list)
    #Player
    total_pokemon_seen: int = field(default_factory=int)
    pokemon_seen_in_the_pokedex: List[int] = field(default_factory=list)
    byte_representation_of_caught_pokemon_in_the_pokedex: List[int] = field(default_factory=list)
    
    ## Pokemon
    
    ### PP
    each_pokemon_pp: List[int] = field(default_factory=list)
    
    # Battle
    
    ## Opponents
    
    opponent_pokemon_levels: List[int] = field(default_factory=list)
    
    ### Trainer
    enemy_trainer_pokemon_hp: List[int] = field(default_factory=list)
    
    ### enermy wild pokemon
    enemy_pokemon_hp:int = field(default_factory=int)
    
    
    last_black_out_map_id: int = field(default_factory=int)
    
    # Events
    total_events_that_occurs_in_game:int = field(default_factory=int)
    enemy_monster_actually_catch_rate: float = field(default_factory=float)
    # Battle Stuff
    player_current_monster_stats_modifier_attack: int = field(default_factory=int)
    player_current_monster_stats_modifier_defense: int = field(default_factory=int)
    player_current_monster_stats_modifier_speed: int = field(default_factory=int)
    player_current_monster_stats_modifier_special: int = field(default_factory=int)
    player_current_monster_stats_modifier_accuracy: int = field(default_factory=int)
    
    enemy_current_pokemon_stats_modifier_attack: int = field(default_factory=int)
    enemy_current_pokemon_stats_modifier_defense: int = field(default_factory=int)
    enemy_current_pokemon_stats_modifier_speed: int = field(default_factory=int)
    enemy_current_pokemon_stats_modifier_special: int = field(default_factory=int)
    enemy_current_pokemon_stats_modifier_accuracy: int = field(default_factory=int)
    enemy_current_move_effect:int = field(default_factory=int)
    enemy_pokemon_move_power:float = field(default_factory=float)
    enemy_pokemon_move_type:int = field(default_factory=int)
    
    # World Map
    map_id: int = field(default_factory=int)
    
   
    def __init__( self , next_state_internal_game_state, time:int , max_episode_steps:int):
       self.map_music_sound_bank = next_state_internal_game_state.map_music_rom_bank
       self.map_music_sound_id = next_state_internal_game_state.map_music_sound_id
       self.party_size = next_state_internal_game_state.party_size
       self.each_pokemon_level = next_state_internal_game_state.each_pokemon_level
       self.total_party_level = next_state_internal_game_state.total_party_level
       self.battle_stats = next_state_internal_game_state.battle_stats.value
       self.battle_result = next_state_internal_game_state.battle_result.value
       self.number_of_turns_in_current_battle = next_state_internal_game_state.number_of_turn_in_pokemon_battle
       self.each_pokemon_health_points = next_state_internal_game_state.each_pokemon_health_points
       self.each_pokemon_max_health_points = next_state_internal_game_state.each_pokemon_max_health_points
       self.total_party_health_points = next_state_internal_game_state.total_party_health_points
       self.total_party_max_hit_points = next_state_internal_game_state.total_party_max_hit_points
       self.low_health_alarm = next_state_internal_game_state.low_health_alaram 
       
       self.total_number_of_items = next_state_internal_game_state.total_number_of_items
       self.money = next_state_internal_game_state.money
       self.player_selected_move_id = next_state_internal_game_state.player_selected_move_id
       self.enemy_selected_move_id = next_state_internal_game_state.enemy_selected_move_id
       self.player_xp = self.obs_player_xp(next_state_internal_game_state.player_lineup_xp)
       self.total_player_lineup_xp = next_state_internal_game_state.total_player_lineup_xp
       self.total_pokemon_seen = next_state_internal_game_state.total_pokemon_seen
       self.pokemon_seen_in_the_pokedex = next_state_internal_game_state.pokemon_seen_in_the_pokedex
       self.byte_representation_of_caught_pokemon_in_the_pokedex = next_state_internal_game_state.byte_representation_of_caught_pokemon_in_the_pokedex
       
       # Player
       
       ## POkemon
       self.pokemon_party_move_id = next_state_internal_game_state.pokemon_party_move_id
       
       ### PP
       self.each_pokemon_pp = next_state_internal_game_state.each_pokemon_pp
       
       # Battle
       
       ## Opponents
       self.opponent_pokemon_levels = next_state_internal_game_state.opponent_pokemon_levels
       
       ### Trainer
       self.enemy_trainer_pokemon_hp = next_state_internal_game_state.enemy_trainer_pokemon_hp
       
       ### enermy wild pokemon
       self.enemy_pokemon_hp = next_state_internal_game_state.enemy_pokemon_hp
       
       # Events
       self.total_events_that_occurs_in_game = next_state_internal_game_state.total_events_that_occurs_in_game
       
       self.time = time /  ( max_episode_steps * 16)
       
       self.enemy_monster_actually_catch_rate = self.obs_enemy_monster_pokemon_actually_catch_rate(next_state_internal_game_state.enemy_monster_actually_catch_rate)
       
       # Battle Stuff
       self.player_current_monster_stats_modifier_attack = next_state_internal_game_state.player_current_monster_stats_modifier_attack
       self.player_current_monster_stats_modifier_defense = next_state_internal_game_state.player_current_monster_stats_modifier_defense
       self.player_current_monster_stats_modifier_speed = next_state_internal_game_state.player_current_monster_stats_modifier_speed
       self.player_current_monster_stats_modifier_special = next_state_internal_game_state.player_current_monster_stats_modifier_special
       self.player_current_monster_stats_modifier_accuracy = next_state_internal_game_state.player_current_monster_stats_modifier_accuracy
       
       self.enemy_current_pokemon_stats_modifier_attack = next_state_internal_game_state.enemy_current_pokemon_stats_modifier_attack
       self.enemy_current_pokemon_stats_modifier_defense = next_state_internal_game_state.enemy_current_pokemon_stats_modifier_defense
       self.enemy_current_pokemon_stats_modifier_speed = next_state_internal_game_state.enemy_current_pokemon_stats_modifier_speed
       self.enemy_current_pokemon_stats_modifier_special = next_state_internal_game_state.enemy_current_pokemon_stats_modifier_special
       self.enemy_current_pokemon_stats_modifier_accuracy = 0 #next_state_internal_game_state.enemy_current_pokemon_stats_modifier_accuracy
       self.enemy_current_pokemon_stats_modifier_evasion = next_state_internal_game_state.enemy_current_pokemon_stats_modifier_evasion
       self.enemy_current_move_effect = next_state_internal_game_state.enemy_current_move_effect
       self.enemy_pokemon_move_power = next_state_internal_game_state.enemy_pokemon_move_power
       self.enemy_pokemon_move_type = next_state_internal_game_state.enemy_pokemon_move_type
       self.enemy_pokemon_move_accuracy = next_state_internal_game_state.enemy_pokemon_move_accuracy
       
       self.last_black_out_map_id = next_state_internal_game_state.last_black_out_map_id
       
       # Map
       self.map_id = next_state_internal_game_state.map_id
       
       
       
       self.validation()
       self.encode()
       self.normalize()
    def validation(self):
        #assert self.low_health_alarm in [0, 145, 152, 136] , T()
        assert self.enemy_pokemon_hp >=0 , T()
        assert self.enemy_pokemon_hp <= 705 , T()
        assert self.time <= 1 , T()
        assert self.last_black_out_map_id <= 150 , T()
        assert isinstance(self.last_black_out_map_id, int) , T()
        assert self.map_id <= 250 , T()
        
    def normalize_np_array(self , np_array, lookup=True, size=256.0):
        if lookup:
            #Anp_array = np.vectorize(lambda x: self.env.memory.byte_to_float_norm[int(x)])(np_array)
            p = 2
        else:
            np_array = np.vectorize(lambda x: int(x) / size)(np_array)

        return np_array
    def obs_player_xp(self, player_lineup_xp):
        xp_array: np.ndarray[Any, np.dtype[np.floating[np._32Bit]]] = np.array(self.normalize_np_array(player_lineup_xp, False, 250000), dtype=np.float32)
        padded_xp = np.pad(xp_array, (0, 6 - len(xp_array)), mode='constant')
        return padded_xp
    def obs_enemy_monster_pokemon_actually_catch_rate(self , enemy_monster_actually_catch_rate):
        return enemy_monster_actually_catch_rate / 255.0
    def encode(self):
        stuff  = { 2: 0 , 8:1 , 31:2}
        self.map_music_sound_bank = stuff[self.map_music_sound_bank]
        # A bad way to encode on the map music sound id so far will later get a better way in these method 
        self.map_music_sound_id = self.map_music_sound_id - 176
        self.low_health_alarm =  1 if self.low_health_alarm != 0 else 0 
        self.encode_move_type()
    def normalize(self):
        for index in range(len(self.each_pokemon_health_points)):
            if self.each_pokemon_max_health_points[index] >0:
                self.each_pokemon_health_points[index] = self.each_pokemon_health_points[index]/self.each_pokemon_max_health_points[index]
        self.obs_player_total_max_health_points()
        self.obs_player_current_monster_stats_modifier_attack()
        self.obs_player_current_monster_stats_modifier_defense()
        self.obs_player_current_monster_stats_modifier_speed()
        self.obs_player_current_monster_stats_modifier_special()
        self.obs_player_current_monster_stats_modifier_accuracy()
        self.obs_enemy_pokemon_move_accuracy()
    def obs_player_total_max_health_points(self):
        self.total_party_max_hit_points = self.total_party_max_hit_points / ( 705.0 * 6.0)
    def obs_player_current_monster_stats_modifier_attack(self):
        return self.player_current_monster_stats_modifier_attack / 255.0
    def obs_player_current_monster_stats_modifier_defense(self):
        return self.player_current_monster_stats_modifier_defense / 255.0
    def obs_player_current_monster_stats_modifier_speed(self):
        return self.player_current_monster_stats_modifier_speed / 255.0
    def obs_player_current_monster_stats_modifier_special(self):
        return self.player_current_monster_stats_modifier_special / 255.0
    def obs_player_current_monster_stats_modifier_accuracy(self):
        return self.player_current_monster_stats_modifier_accuracy / 255.0
    def obs_enemy_current_pokemon_stats_modifier_attack(self):
        return self.enemy_current_pokemon_stats_modifier_attack / 255.0
    def obs_enemy_current_pokemon_stats_modifier_defense(self):
        return self.enemy_current_pokemon_stats_modifier_defense / 255.0
    def obs_enemy_current_pokemon_stats_modifier_speed(self):
        return self.enemy_current_pokemon_stats_modifier_speed / 255.0
    def obs_enemy_current_pokemon_stats_modifier_special(self):
        return self.enemy_current_pokemon_stats_modifier_special / 255.0
    def obs_enemy_current_pokemon_stats_modifier_accuracy(self):
        return self.enemy_current_pokemon_stats_modifier_accuracy / 255.0
    def obs_enemy_current_pokemon_stats_modifier_evasion(self):
        return self.enemy_current_pokemon_stats_modifier_evasion / 255.0
    def obs_enemy_pokemon_move_power(self):
        return self.enemy_pokemon_move_power / 255.0
    def encode_move_type(self):
        new_dict = {
            0: 0,
            1: 1,
            2: 2,
            3: 3,
            4: 4,
            5: 5,
            7: 6,
            8: 7,
            20: 8 , 
            21: 9 , 
            22: 10  ,
            23: 11 ,
            24: 12 ,
            25: 13 ,
            26: 14 ,
        }
        self.enemy_pokemon_move_type = new_dict[self.enemy_pokemon_move_type]
    def obs_enemy_pokemon_move_accuracy(self):
        return self.enemy_pokemon_move_accuracy / 255.0
    
    def get_obs(self) -> dict[str, Any]:
        return asdict(self)
    def to_json(self):
        return self.get_obs()