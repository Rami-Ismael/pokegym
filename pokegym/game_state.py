from dataclasses import asdict, dataclass , field 
from typing import List
from pokegym import ram_map
import numpy as np
@dataclass
class Internal_Game_State:
    #last_pokecenter_id: int = field(default_factory=int)
    battle_stats: ram_map.BattleState = field(default_factory=lambda: ram_map.BattleState.NOT_IN_BATTLE)  # Default to NOT_IN_BATTLE or any other default state
    battle_result: ram_map.BattleResult = field(default_factory=lambda: ram_map.BattleResult.IDK)  # Default to NOT_IN_BATTLE or any other default state
    map_music_sound_id: int = field(default_factory=int)
    map_music_rom_bank: int = field(default_factory=int)
    
    party_size: int = field(default_factory=int)
    each_pokemon_level: List[int] = field(default_factory=list)
    lowest_pokemon_level: int = field(default_factory=int)
    highest_pokemon_level: int = field(default_factory=int)
    total_party_level: int = field(default_factory=int)
    average_pokemon_level: float = field(default_factory=float)
    
    number_of_turn_in_pokemon_battle: int = field(default_factory=int)
    number_of_turn_in_pokemon_battle_greater_than_eight : int = field(default_factory=int)
    number_of_turn_in_pokemon_battle_greater_than_sixteen : int = field(default_factory=int)
    number_of_turn_in_pokemon_battle_greater_than_thirty_two : int = field(default_factory=int)
    
    
    # Health Points
    each_pokemon_health_points: List[int] = field(default_factory=list)
    each_pokemon_max_health_points: List[int] = field(default_factory=list)
    lowest_pokemon_health_points: int = field(default_factory=int)
    highest_pokemon_health_points: int = field(default_factory=int)
    total_party_health_points: int = field(default_factory=int)
    total_party_max_hit_points: int = field(default_factory=int)
    average_pokemon_health_points: float = field(default_factory=float)
    average_pokemon_max_health_points: float = field(default_factory=float)
    low_health_alaram: int = field(default_factory=int)
    #total_party_health_ratio: float = field(default_factory=float)
    
    # Items
    total_number_of_items: int = field(default_factory=int)
    money: int = field(default_factory=int)
    #items_ids: List[int] = field(default_factory=list)
    #items_ids_quantities: List[int] = field(default_factory=list)
    
    # Moves
    player_selected_move_id: int = field(default_factory=int)
    enemy_selected_move_id: int = field(default_factory=int)
    pokemon_party_move_id: list[int] = field(default_factory=list)
    number_of_uniques_moves_in_party:int = field(default_factory=int)
    opponent_party_move_ids: list[int] = field(default_factory=list)
    number_of_uniques_moves_in_opponent_party: int = field(default_factory=int)
    #total_number_pokemon_moves_in_the_teams : int = field(default_factory=int)
    #number_of_unique_moves_in_the_teams: int = field(default_factory=int)
    
    # Player
    total_pokemon_seen:int = field(default_factory=int)
    pokemon_seen_in_the_pokedex: List[int] = field(default_factory=list)
    byte_representation_of_caught_pokemon_in_the_pokedex: List[int] = field(default_factory=list)
    
    ## Pokemon
    
    ### Pokedex
    total_pokemon_seen_in_pokedex:int = field(default_factory=int)
    
    ### PP
    each_pokemon_pp: List[int] = field(default_factory=list)
    
    # Battle
    
    ## opponetns
    opponent_pokemon_levels: List[int] = field(default_factory=list)
    
    ### Trainer
    # Only valid for trainers/gyms not wild mons. HP doesn't dec until mon is dead, then it's 0
    enemy_trainer_pokemon_hp: List[int] = field(default_factory=list)
   
    ### Enermy Pokemon Hp
    enemy_pokemon_hp:int = field(default_factory=int)
    
    # Events
    total_events_that_occurs_in_game:int = field(default_factory=int)
    
    
    # Music
    gym_leader_music_is_playing: bool = field(default_factory=bool)
    
    # Missacellnous
    wild_pokemon_encounter_rate_on_grass:int = field(default_factory = int) #ram_map.wild_pokemon_encounter_rate_on_grass(game)
    enemy_pokemon_base_exp_yeild:int = field(default_factory=int)
    enemy_monster_actually_catch_rate:int = field(default_factory=int)
    
    
    


    def __init__(self, game=None):
        #self.last_pokecenter_id = ram_map.get_last_pokecenter_id(game) if game else 0
        self.battle_stats = ram_map.is_in_battle(game)
        self.battle_result = ram_map.get_battle_result(game)
        self.map_music_sound_id = ram_map.get_map_music_id(game)
        self.map_music_rom_bank = ram_map.get_map_music_rom_bank(game)
        self.each_pokemon_level = ram_map.get_party_pokemon_level(game)
        self.party_size = ram_map.get_party_size(game)
        self.lowest_pokemon_level = min(self.each_pokemon_level)
        self.highest_pokemon_level = max(self.each_pokemon_level)
        self.total_party_level = sum(self.each_pokemon_level)
        self.average_pokemon_level = self.total_party_level/self.party_size
        self.number_of_turn_in_pokemon_battle = ram_map.get_number_of_turns_in_current_battle(game)
        self.number_of_turn_in_pokemon_battle_greater_than_eight = 1 if self.number_of_turn_in_pokemon_battle > 8 else 0
        self.number_of_turn_in_pokemon_battle_greater_than_sixteen = 1 if self.number_of_turn_in_pokemon_battle > 16 else 0
        self.number_of_turn_in_pokemon_battle_greater_than_thirty_two = 1 if self.number_of_turn_in_pokemon_battle > 32 else 0
        # Health 
        self.each_pokemon_health_points = ram_map.each_pokemon_hit_points(game)
        self.each_pokemon_max_health_points = ram_map.get_each_pokemon_max_hit_points(game)
        self.lowest_pokemon_health_points = min(self.each_pokemon_health_points)
        self.highest_pokemon_health_points = max(self.each_pokemon_health_points)
        self.total_party_health_points = sum(self.each_pokemon_health_points)
        self.wipe_out = 1 if self.total_party_health_points == 0 else 0
        self.total_party_max_hit_points = sum(self.each_pokemon_max_health_points)
        self.average_pokemon_health_points = self.total_party_health_points/self.party_size
        self.average_pokemon_max_health_points = self.total_party_max_hit_points/self.party_size
        self.low_health_alaram = ram_map.get_low_health_alarm(game)
        
        # Items 
        self.total_number_of_items = ram_map.total_items(game)  # # The count of all the items held in players bag
        self.money = ram_map.money(game)  # # The count of all the items held in players bag
        
        # Moves
        self.player_selected_move_id , self.enemy_selected_move_id = ram_map.get_battle_turn_moves(game)
        self.pokemon_party_move_id  = ram_map.get_pokemon_party_move_ids(game)
        self.opponent_party_move_ids  = ram_map.get_opponent_party_move_id(game)
        #self.total_number_pokemon_moves_in_the_teams = sum( self.pokemon_party_move_id >0 )
        # Player
        
        ### Pokedex
        
        self.total_pokemon_seen_in_pokedex = ram_map.total_pokemon_seen(game)
        
        ### PP
        self.each_pokemon_pp = ram_map.get_pokemon_pp_avail(game)
        
        ### XP 
        self.player_lineup_xp = ram_map.get_player_lineup_xp(game)
        self.total_player_lineup_xp = sum(self.player_lineup_xp)
        
        ## Battles
        
        ### Opponents
        self.opponent_pokemon_levels = ram_map.get_opponent_pokemon_levels(game)
        
        #### Trainer
        self.enemy_trainer_pokemon_hp = ram_map.get_enemy_trainer_pokemon_hp(game)
        
        #### Enemy 
        self.enemy_pokemon_hp = ram_map.get_enemys_pokemon_hp(game)
        
        # Seen Pokemon
        self.total_pokemon_seen = ram_map.total_pokemon_seen(game)
        self.pokemon_seen_in_the_pokedex = ram_map.pokemon_see_in_the_pokedex(game)
        self.byte_representation_of_caught_pokemon_in_the_pokedex  = ram_map.get_pokedex_entries_of_caught_pokemon(game)
        ## assert all value are not none
        
       # Events
        self.total_events_that_occurs_in_game = ram_map.total_events_that_occurs_in_game(game) 
        
        # Music
        self.gym_leader_music_is_playing = ram_map.check_if_gym_leader_music_is_playing(game)
        
        self.wild_pokemon_encounter_rate_on_grass = ram_map.wild_pokemon_encounter_rate_on_grass(game)
        self.enemy_pokemon_base_exp_yeild = ram_map.get_enemy_pokemon_base_exp_yield(game)
        self.enemy_monster_actually_catch_rate = ram_map.get_enemy_monster_actually_catch_rate(game)
        self.validation()
    def to_json(self) -> dict:
        assert all(v is not None for v in self.each_pokemon_level)
        for k, v in asdict(self).items():
            if v is None:
                raise ValueError(f"Value of {k} is None")
            if isinstance(v, list):
                ValueError(f"Value of {k} is a list")
        return asdict(self )
    def validation(self):
        for index in range(len(self.pokemon_party_move_id)):
            assert self.pokemon_party_move_id[index] <= 255 , "Pokemon Party Move ID is not valid" # https://gamefaqs.gamespot.com/gameboy/367023-pokemon-red-version/faqs/74734?page=4#section30
        assert len(self.pokemon_seen_in_the_pokedex) <=152
@dataclass
class External_Game_State:
    #visited_pokecenter_list: List[int] = field(default_factory=list)
    number_of_battles_wins: int = field(default_factory=int)
    number_of_battles_loses: int = field(default_factory=int)
    number_of_battles_draw: int = field(default_factory=int)
    
    max_party_size: int = field(default_factory=int)
    
    total_events_that_occurs_in_game:int = field(default_factory=int)
    total_number_of_wipe_out_in_episode: int = field(default_factory=int)
    
    # Levels
    max_total_party_level: int = field(default_factory=int)
    max_highest_level_in_the_party_teams:int = field(default_factory=int)
    max_enemy_pokemon_base_exp_yeild:int = field(default_factory = int)
    
    def update(self, game , next_next_internal_game_state ):
        #self.update_visited_pokecenter_list(game_state)
        self.update_battle_results(game)
    def post_reward_update(self, game):
        self.max_party_size = max(self.max_party_size, game.party_size)
        self.total_events_that_occurs_in_game = game.total_events_that_occurs_in_game
        self.max_total_party_level = max(self.max_total_party_level, game.total_party_level)
        self.max_highest_level_in_the_party_teams = max( self.max_highest_level_in_the_party_teams , game.highest_pokemon_level)
        self.total_number_of_wipe_out_in_episode+=game.wipe_out
        self.max_enemy_pokemon_base_exp_yeild:int = max(self.max_enemy_pokemon_base_exp_yeild , game.enemy_pokemon_base_exp_yeild)
    
    def update_battle_results(self, game) -> None:
        if ram_map.is_in_battle(game):
            battle_result = ram_map.get_battle_result(game)
            if battle_result == ram_map.BattleResult.WIN:
                self.number_of_battles_wins += 1
            elif battle_result == ram_map.BattleResult.LOSE:
                self.number_of_battles_loses += 1
            elif battle_result == ram_map.BattleResult.DRAW:
                self.number_of_battles_draw += 1
    
    #def update_visited_pokecenter_list(self, game_state) -> None:
    #    last_pokecenter_id = ram_map.get_last_pokecenter_id(game_state)
    #    if last_pokecenter_id != -1 and last_pokecenter_id not in self.visited_pokecenter_list:
    #        self.visited_pokecenter_list.append(last_pokecenter_id)
    def to_json(self) -> dict:
        return asdict(self)
        

        