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
    number_of_turns_in_current_battle: int = field(default_factory=int)
   
    def __init__( self , next_state_internal_game_state):
       self.map_music_sound_bank = next_state_internal_game_state.map_music_rom_bank
       self.party_size = next_state_internal_game_state.party_size
       self.each_pokemon_level = next_state_internal_game_state.each_pokemon_level
       self.total_party_level = next_state_internal_game_state.total_party_level
       self.battle_stats = next_state_internal_game_state.battle_stats.value
       self.number_of_turns_in_current_battle = next_state_internal_game_state.number_of_turn_in_pokemon_battle
       self.encode()
       #self.normalize()
    
    def encode(self):
        stuff  = { 2: 0 , 8:1 , 31:2}
        self.map_music_sound_bank = stuff[self.map_music_sound_bank]


    def get_obs(self) -> dict[str, Any]:
        return asdict(self)
    def to_json(self):
        return self.get_obs()
