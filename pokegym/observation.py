from dataclasses import dataclass , field , asdict
from typing import Any, List
from pokegym import ram_map
from rich import print
@dataclass
class Observation:
    map_music_sound_bank: int = field(default_factory=int)
   
    def __init__( self , next_state_internal_game_state):
       self.map_music_sound_bank = next_state_internal_game_state.map_music_rom_bank
       self.encode()
    
    def encode(self):
        stuff  = { 2: 0 , 8:1 , 31:2}
        self.map_music_sound_bank = stuff[self.map_music_sound_bank]


    def get_obs(self) -> dict[str, Any]:
        return asdict(self)
    def to_json(self):
        return self.get_obs()