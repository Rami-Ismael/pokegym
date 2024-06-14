from dataclasses import asdict, dataclass , field 
from typing import List
from pokegym import ram_map
import numpy as np
@dataclass
class Internal_Game_State:
    #last_pokecenter_id: int = field(default_factory=int)
    battle_stats: ram_map.BattleState = field(default_factory=lambda: ram_map.BattleState.NOT_IN_BATTLE)  # Default to NOT_IN_BATTLE or any other default state
    batle_result: ram_map.BattleResult = field(default_factory=lambda: ram_map.BattleResult.IDK)  # Default to NOT_IN_BATTLE or any other default state
    map_music_sound_id: int = field(default_factory=int)
    map_music_rom_bank: int = field(default_factory=int)
    
    party_size: int = field(default_factory=int)
    each_pokemon_level: List[int] = field(default_factory=list)
    lowest_pokemon_level: int = field(default_factory=int)
    highest_pokemon_level: int = field(default_factory=int)
    total_party_level: int = field(default_factory=int)
    average_pokemon_level: float = field(default_factory=float)
    
    number_of_turn_in_pokemon_battle: int = field(default_factory=int)
    
    # Health Points
    each_pokemon_health_points: List[int] = field(default_factory=list)
    each_pokemon_max_health_points: List[int] = field(default_factory=list)
    lowest_pokemon_health_points: int = field(default_factory=int)
    highest_pokemon_health_points: int = field(default_factory=int)
    total_party_health_points: int = field(default_factory=int)
    total_party_max_hit_points: int = field(default_factory=int)
    average_pokemon_health_points: float = field(default_factory=float)
    average_pokemon_max_health_points: float = field(default_factory=float)
    
    # Items
    total_number_of_items: int = field(default_factory=int)
    money: int = field(default_factory=int)
    
    # Moves
    player_selected_move_id: int = field(default_factory=int)
    enemy_selected_move_id: int = field(default_factory=int)
    


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
        self.number_of_turn_in_pokemon_battle = ram_map.get_number_of_turns_in_current_battle(game)
        # Health 
        self.each_pokemon_health_points = ram_map.each_pokemon_hit_points(game)
        self.each_pokemon_max_health_points = ram_map.get_each_pokemon_max_hit_points(game)
        self.lowest_pokemon_health_points = min(self.each_pokemon_health_points)
        self.highest_pokemon_health_points = max(self.each_pokemon_health_points)
        self.total_party_health_points = sum(self.each_pokemon_health_points)
        self.total_party_max_hit_points = sum(self.each_pokemon_max_health_points)
        self.average_pokemon_health_points = self.total_party_health_points/self.party_size
        self.average_pokemon_max_health_points = self.total_party_max_hit_points/self.party_size
        
        # Items 
        self.total_number_of_items = ram_map.total_items(game)  # # The count of all the items held in players bag
        self.money = ram_map.money(game)  # # The count of all the items held in players bag
        
        # Moves
        self.player_selected_move_id , self.enemy_selected_move_id = ram_map.get_battle_turn_moves(game)
        # Pokemon
        self.player_lineup_xp = ram_map.get_player_lineup_xp(game)

        
        
        ## assert all value are not none
    def to_json(self) -> dict:
        assert all(v is not None for v in self.each_pokemon_level)
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
        

        