from dataclasses import dataclass , field , asdict
from typing import Any, List
from pokegym import ram_map
from rich import print
@dataclass
class Observation:
    map_music_sound_bank: int = field(default_factory=int)
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
   
    def __init__( self , next_state_internal_game_state):
       self.map_music_sound_bank = next_state_internal_game_state.map_music_rom_bank
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
       self.encode()
       self.normalize()
    
    def encode(self):
        stuff  = { 2: 0 , 8:1 , 31:2}
        self.map_music_sound_bank = stuff[self.map_music_sound_bank]
    def normalize(self):
        for index in range(len(self.each_pokemon_health_points)):
            if self.each_pokemon_max_health_points[index] >0:
                self.each_pokemon_health_points[index] = self.each_pokemon_health_points[index]/self.each_pokemon_max_health_points[index]


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
