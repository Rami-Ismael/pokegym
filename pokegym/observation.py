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
    
    each_pokemon_health_points: List[float] = field(default_factory=list)
    each_pokemon_max_health_points: List[float] = field(default_factory=list)
    total_party_health_points: float = field(default_factory=float)
    total_party_max_hit_points: float = field(default_factory=float)
    low_health_alarm: int = field(default_factory=int)
    
    total_number_of_items: int = field(default_factory=int)
    money: int = field(default_factory=int)
    
    # Moves
    player_selected_move_id: int = field(default_factory=int)
    enemy_selected_move_id: int = field(default_factory=int)
    #Player
    total_pokemon_seen: int = field(default_factory=int)
    pokemon_seen_in_the_pokedex: List[int] = field(default_factory=list)
    byte_representation_of_caught_pokemon_in_the_pokedex: List[int] = field(default_factory=list)

   
    def __init__( self , next_state_internal_game_state):
       self.map_music_sound_bank = next_state_internal_game_state.map_music_rom_bank
       self.map_music_sound_id = next_state_internal_game_state.map_music_sound_id
       self.party_size = next_state_internal_game_state.party_size
       self.each_pokemon_level = next_state_internal_game_state.each_pokemon_level
       self.total_party_level = next_state_internal_game_state.total_party_level
       self.battle_stats = next_state_internal_game_state.battle_stats.value
       self.battle_result = next_state_internal_game_state.batle_result.value
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
       self.total_pokemon_seen = next_state_internal_game_state.total_pokemon_seen
       self.pokemon_seen_in_the_pokedex = next_state_internal_game_state.pokemon_seen_in_the_pokedex
       self.byte_representation_of_caught_pokemon_in_the_pokedex = next_state_internal_game_state.byte_representation_of_caught_pokemon_in_the_pokedex
       self.encode()
       self.normalize()
       self.validation()
    def validation(self):
        assert self.low_health_alarm in [0,1] , T()
    def normalize_np_array(self , np_array, lookup=True, size=256.0):
        if lookup:
            #Anp_array = np.vectorize(lambda x: self.env.memory.byte_to_float_norm[int(x)])(np_array)
            p = 2
        else:
            np_array = np.vectorize(lambda x: int(x) / size)(np_array)

        return np_array
    def obs_player_xp(self, player_lineup_xp):
        xp_array = np.array(self.normalize_np_array(player_lineup_xp, False, 250000), dtype=np.float32)
        padded_xp = np.pad(xp_array, (0, 6 - len(xp_array)), mode='constant')
        return padded_xp
    def encode(self):
        stuff  = { 2: 0 , 8:1 , 31:2}
        self.map_music_sound_bank = stuff[self.map_music_sound_bank]
        # A bad way to encode on the map music sound id so far will later get a better way in these method 
        self.map_music_sound_id = self.map_music_sound_id - 176
    def normalize(self):
        for index in range(len(self.each_pokemon_health_points)):
            if self.each_pokemon_max_health_points[index] >0:
                self.each_pokemon_health_points[index] = self.each_pokemon_health_points[index]/self.each_pokemon_max_health_points[index]
        self.obs_player_total_max_health_points()
    def obs_player_total_max_health_points(self):
        self.total_party_max_hit_points = self.total_party_max_hit_points / ( 705.0 * 6.0)



    def get_obs(self) -> dict[str, Any]:
        return asdict(self)
    def to_json(self):
        ## Check if all the value are greater than zero
        for key, value in self.get_obs().items():
            if isinstance(value, List):
                for v in value:
                    assert v >= 0
            else:
                assert value >= 0
        return self.get_obs()
