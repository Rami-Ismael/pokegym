from dataclasses import dataclass , field
from typing import List
from pokegym import ram_map

@dataclass
class Internal_Game_State:
    last_pokecenter_id:int = field(default_factory=int)
    def __init__(self , game = game):
        self.last_pokecenter_id = ram_map.get_last_pokecenter_id(game)

@dataclass
class External_Game_State:
    visited_pokecenter_list: List[int] = field(default_factory=list)
    
    def update(self, game_state):
        self.update_visited_pokecenter_list(game_state)
    
    def update_visited_pokecenter_list(self, game_state) -> None:
        last_pokecenter_id = game_state.get_last_pokecenter_id()
        if last_pokecenter_id != -1 and last_pokecenter_id not in self.visited_pokecenter_list:
            self.visited_pokecenter_list.append(last_pokecenter_id)
        

        