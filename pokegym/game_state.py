from dataclasses import asdict, dataclass , field 
from typing import List
from pokegym import ram_map
@dataclass
class Internal_Game_State:
    #last_pokecenter_id: int = field(default_factory=int)
    battle_stats: ram_map.BattleState = field(default_factory=lambda: ram_map.BattleState.NOT_IN_BATTLE)  # Default to NOT_IN_BATTLE or any other default state
    batle_result: ram_map.BattleResult = field(default_factory=lambda: ram_map.BattleResult.DRAW)  # Default to NOT_IN_BATTLE or any other default state
    map_music_sound_id: int = field(default_factory=int)
    map_music_rom_bank: int = field(default_factory=int)
    
    party_size: int = field(default_factory=int)
    each_pokemon_level: List[int] = field(default_factory=list)
    lowest_pokemon_level: int = field(default_factory=int)
    highest_pokemon_level: int = field(default_factory=int)
    total_party_level: int = field(default_factory=int)
    average_pokemon_level: float = field(default_factory=float)

    def __init__(self, game=None):
        #self.last_pokecenter_id = ram_map.get_last_pokecenter_id(game) if game else 0
        self.battle_stats = ram_map.is_in_battle(game)
        self.batle_result = ram_map.get_battle_result(game)
        self.map_music_sound_id = ram_map.get_map_music_id(game)
        self.map_music_rom_bank = ram_map.get_map_music_rom_bank(game)
        self.each_pokemon_level = ram_map.get_party_pokemon_level(game)
        self.party_size = ram_map.get_party_size(game)
        self.lowest_pokemon_level = min(self.each_pokemon_level)
        self.highest_pokemon_level = max(self.each_pokemon_level)
        self.total_party_level = sum(self.each_pokemon_level)
        self.average_pokemon_level = self.total_party_level/self.party_size
        ## assert all value are not none
    def to_json(self) -> dict:
        for k, v in asdict(self).items():
            if v is None:
                raise ValueError(f"Value of {k} is None")
        return asdict(self )
@dataclass
class External_Game_State:
    visited_pokecenter_list: List[int] = field(default_factory=list)
    number_of_battles_wins: int = field(default_factory=int)
    number_of_battles_loses: int = field(default_factory=int)
    number_of_battles_draw: int = field(default_factory=int)
    
    def update(self, game , game_state):
        #self.update_visited_pokecenter_list(game_state)
        self.update_battle_results(game)
    
    def update_battle_results(self, game) -> None:
        if ram_map.is_in_battle(game):
            battle_result = ram_map.get_battle_result(game)
            if battle_result == ram_map.BattleResult.WIN:
                self.number_of_battles_wins += 1
            elif battle_result == ram_map.BattleResult.LOSE:
                self.number_of_battles_loses += 1
            elif battle_result == ram_map.BattleResult.DRAW:
                self.number_of_battles_draw += 1
    
    def update_visited_pokecenter_list(self, game_state) -> None:
        last_pokecenter_id = ram_map.get_last_pokecenter_id(game_state)
        if last_pokecenter_id != -1 and last_pokecenter_id not in self.visited_pokecenter_list:
            self.visited_pokecenter_list.append(last_pokecenter_id)
        

        