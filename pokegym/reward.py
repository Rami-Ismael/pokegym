from typing import Dict
from dataclasses import asdict, dataclass
from pokegym.ram_map import BattleState , BattleResult
from pokegym.ram_reader.red_memoy_moves import GROWL_DECIMAL_VALUE_OF_MOVE_ID , TAIL_DECIAML_VALUE_OF_MOVE_ID , LEER_DECIMAL_VALUE_OF_MOVE_ID

@dataclass
class Reward:
    #use max to make sure the agent cannot gain the system with the PC
    reward_for_increasing_the_max_size_of_the_trainer_team:float = 0
    
    # Reward
    reward_for_doing_new_events_that_occurs_in_game_calculating_by_game_state:int = 0
    reward_for_doing_new_events_that_occurs_in_game_calculating_by_external_game_state:int = 0 
    
    # Level which can be effected by the PC or daycare we want the max value
    reward_for_increasing_the_total_party_level:float = 0
    reward_for_increasing_the_highest_pokemon_level_in_the_team_by_battle:int = 0
    
    # Reward for seeing new pokemon 
    #reward_for_seeing_new_pokemon_in_the_pokedex:int = 0
    
    reward_for_taking_action_that_start_playing_the_gym_player_music:int = 0
    
    reward_for_using_bad_moves:int = 0
    
    knocking_out_enemy_pokemon:int = 0
    
    knocking_out_wild_pokemon:int = 0
    
    negative_reward_for_wiping_out:float = 0

    reward_for_entering_a_trainer_battle:float = 0
    
    # Extra Exploration Bonus
    reward_for_finding_higher_enemy_pokemon_base_exp_yeild:int = 0
    reward_for_having_last_black_out_id_proximaly_an_pokecenter:int = 0
    
    negative_reward_for_player_monster_stats_modifier_accuracy_drop:float = 0
    negative_reward_for_using_lower_level_pokemon_against_higher_level_pokemon:float = 0
    
    def __init__(self, current_state_internal_game_state , next_state_internal_game_state , external_game_state ,  
                 reward_for_increase_pokemon_level_coef:float = 2 , 
                 reward_for_increasing_the_highest_pokemon_level_in_the_team_by_battle_coef:float = 1 , 
                 reward_for_entering_a_trainer_battle_coef:float = 1.0 , 
                 negative_reward_for_wiping_out_coef:float = 1.0,
                 ):
        
        if current_state_internal_game_state.party_size < next_state_internal_game_state.party_size and next_state_internal_game_state.party_size > external_game_state.max_party_size and external_game_state.max_party_size:
            self.reward_for_increasing_the_max_size_of_the_trainer_team = 0
        # Events
        if current_state_internal_game_state.total_events_that_occurs_in_game < next_state_internal_game_state.total_events_that_occurs_in_game:
            self.reward_for_doing_new_events_that_occurs_in_game_calculating_by_game_state +=  ( ( next_state_internal_game_state.total_events_that_occurs_in_game - current_state_internal_game_state.total_events_that_occurs_in_game)  * 2 ) 
            assert self.reward_for_doing_new_events_that_occurs_in_game_calculating_by_game_state >= 0
        if external_game_state.total_events_that_occurs_in_game < next_state_internal_game_state.total_events_that_occurs_in_game:
            self.reward_for_doing_new_events_that_occurs_in_game_calculating_by_external_game_state +=  ( next_state_internal_game_state.total_events_that_occurs_in_game - external_game_state.total_events_that_occurs_in_game )
            assert self.reward_for_doing_new_events_that_occurs_in_game_calculating_by_external_game_state >= 0
        
        if current_state_internal_game_state.total_party_level < next_state_internal_game_state.total_party_level and next_state_internal_game_state.total_party_level > external_game_state.max_total_party_level:
            self.reward_for_increasing_the_total_party_level =  0
        
        if not current_state_internal_game_state.gym_leader_music_is_playing and next_state_internal_game_state.gym_leader_music_is_playing:
            self.reward_for_taking_action_that_start_playing_the_gym_player_music = 4
        
        if next_state_internal_game_state.player_selected_move_id in [GROWL_DECIMAL_VALUE_OF_MOVE_ID , TAIL_DECIAML_VALUE_OF_MOVE_ID , LEER_DECIMAL_VALUE_OF_MOVE_ID]:
            self.reward_for_using_bad_moves -= 1
            assert self.reward_for_using_bad_moves <= 0
        
        if current_state_internal_game_state.battle_stats == BattleState.WILD_BATTLE and next_state_internal_game_state.battle_result == BattleResult.WIN and next_state_internal_game_state.battle_stats == BattleState.NOT_IN_BATTLE and current_state_internal_game_state.party_size == next_state_internal_game_state.party_size:
            self.knocking_out_wild_pokemon = 1
        
        if current_state_internal_game_state.enemy_pokemon_hp  > 0 and next_state_internal_game_state.enemy_pokemon_hp == 0 and current_state_internal_game_state.battle_stats.value != BattleState.NOT_IN_BATTLE and current_state_internal_game_state.party_size == next_state_internal_game_state.party_size:
            self.knocking_out_enemy_pokemon = 1
        
        
        if current_state_internal_game_state.highest_pokemon_level < next_state_internal_game_state.highest_pokemon_level and next_state_internal_game_state.highest_pokemon_level > external_game_state.max_highest_level_in_the_party_teams:
            self.reward_for_increasing_the_highest_pokemon_level_in_the_team_by_battle = ( next_state_internal_game_state.highest_pokemon_level - external_game_state.max_highest_level_in_the_party_teams )
        
        if not current_state_internal_game_state.wipe_out and next_state_internal_game_state.wipe_out:
            self.negative_reward_for_wiping_out = -1.0 * negative_reward_for_wiping_out_coef
        
        # Exploration Benefit
        if current_state_internal_game_state.enemy_pokemon_base_exp_yeild < next_state_internal_game_state.enemy_pokemon_base_exp_yeild and external_game_state.max_enemy_pokemon_base_exp_yeild < next_state_internal_game_state.enemy_pokemon_base_exp_yeild:
            self.reward_for_finding_higher_enemy_pokemon_base_exp_yeild+=1
        self.update_reward_for_entering_a_trainer_battle(current_state_internal_game_state , next_state_internal_game_state , reward_for_entering_a_trainer_battle_coef)
        
        self.update_negative_reward_for_player_monster_stats_modifier_accuracy_drop(current_state_internal_game_state , next_state_internal_game_state)
        
        self.update_negative_reward_for_using_lower_level_pokemon_against_higher_level_pokemon(current_state_internal_game_state , next_state_internal_game_state)
    
    def update_reward_for_entering_a_trainer_battle(self , current_state_internal_game_state , next_state_internal_game_state , reward_for_entering_a_trainer_battle_coef:float = 1.0):
        if current_state_internal_game_state.battle_stats == BattleState.NOT_IN_BATTLE and next_state_internal_game_state.battle_stats == BattleState.TRAINER_BATTLE:
            self.reward_for_entering_a_trainer_battle = 1 * reward_for_entering_a_trainer_battle_coef
    def update_reward_for_having_last_black_out_id_proximaly_an_pokecenter(self , current_state_internal_game_state , next_state_internal_game_state , reward_for_having_last_black_out_id_proximaly_an_pokecenter_coef:float = 1.0):
        if current_state_internal_game_state.last_black_out_map_id < next_state_internal_game_state.last_black_out_map_id and next_state_internal_game_state.last_black_out_map_id in [1 , 2]:
            self.reward_for_having_last_black_out_id_proximaly_an_pokecenter = 1
    def update_negative_reward_for_player_monster_stats_modifier_accuracy_drop(self , current_state_internal_game_state , next_state_internal_game_state , reward_for_player_moving_to_a_pokecenter_coef:float = 1.0):
        if current_state_internal_game_state.player_current_monster_Stats_modifier_accuracy > next_state_internal_game_state.player_current_monster_Stats_modifier_accuracy and next_state_internal_game_state.player_current_monster_Stats_modifier_accuracy > 0:
            self.negative_reward_for_player_monster_stats_modifier_accuracy_drop = -1 # to make sand attack stop considering this action
    def update_negative_reward_for_using_lower_level_pokemon_against_higher_level_pokemon(self , current_state_internal_game_state , next_state_internal_game_state , reward_for_player_moving_to_a_pokecenter_coef:float = 1.0):
        if next_state_internal_game_state.player_current_pokemon_level < next_state_internal_game_state.enemy_current_pokemon_levelel:
            self.negative_reward_for_using_lower_level_pokemon_against_higher_level_pokemon = -1 # stop fighting with weak pokemon 
        
    def total_reward(self) -> int:
        return sum(asdict(self).values())
    def to_json(self) -> Dict[str , Dict[str , int]]:
        #return asdict(self)
        # make the dictoinary a subkey of reward
        return asdict(self)
    
        
    