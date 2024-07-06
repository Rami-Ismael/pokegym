from typing import Dict
from dataclasses import asdict, dataclass
from pokegym.ram_map import BattleState , BattleResult

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
    
    # List of negative Reward for having a long battle I want them short 8 -1 16 -2 32 -4
    negative_reward_for_battle_longer_than_eight_turn:int = 0
    negative_reward_for_battle_longer_than_sixteen_turn:int = 0
    negative_reward_for_battle_longer_than_thirty_two_turn:int = 0
    negative_reward_for_wiping_out:int = 0
    
    def __init__(self, current_state_internal_game_state , next_state_internal_game_state , external_game_state ,  
                 reward_for_increase_pokemon_level_coef:float = 2 , 
                 reward_for_increasing_the_highest_pokemon_level_in_the_team_by_battle_coef:float = 1
                 ):
        
        if current_state_internal_game_state.party_size < next_state_internal_game_state.party_size and next_state_internal_game_state.party_size > external_game_state.max_party_size and external_game_state.max_party_size:
            self.reward_for_increasing_the_max_size_of_the_trainer_team = .25
        # Events
        if current_state_internal_game_state.total_events_that_occurs_in_game < next_state_internal_game_state.total_events_that_occurs_in_game:
            self.reward_for_doing_new_events_that_occurs_in_game_calculating_by_game_state +=  ( ( next_state_internal_game_state.total_events_that_occurs_in_game - current_state_internal_game_state.total_events_that_occurs_in_game)  * 2 ) 
            assert self.reward_for_doing_new_events_that_occurs_in_game_calculating_by_game_state >= 0
        if external_game_state.total_events_that_occurs_in_game < next_state_internal_game_state.total_events_that_occurs_in_game:
            self.reward_for_doing_new_events_that_occurs_in_game_calculating_by_external_game_state +=  ( next_state_internal_game_state.total_events_that_occurs_in_game - external_game_state.total_events_that_occurs_in_game )
            assert self.reward_for_doing_new_events_that_occurs_in_game_calculating_by_external_game_state >= 0
        
        if current_state_internal_game_state.total_party_level < next_state_internal_game_state.total_party_level and next_state_internal_game_state.total_party_level > external_game_state.max_total_party_level:
            self.reward_for_increasing_the_total_party_level =  .25
        
        if not current_state_internal_game_state.gym_leader_music_is_playing and next_state_internal_game_state.gym_leader_music_is_playing:
            self.reward_for_taking_action_that_start_playing_the_gym_player_music = 4
        
        if next_state_internal_game_state.player_selected_move_id in [45 , 49 , 27]:
            self.reward_for_using_bad_moves -= 4
            assert self.reward_for_using_bad_moves <= 0
        
        if current_state_internal_game_state.battle_stats == BattleState.WILD_BATTLE and next_state_internal_game_state.battle_result == BattleResult.WIN and next_state_internal_game_state.battle_stats == BattleState.NOT_IN_BATTLE:
            self.knocking_out_wild_pokemon = 4
        
        if current_state_internal_game_state.enemy_pokemon_hp  > 0 and next_state_internal_game_state.enemy_pokemon_hp == 0 and current_state_internal_game_state.battle_stats.value != BattleState.NOT_IN_BATTLE:
            self.knocking_out_enemy_pokemon = 1
        
        if current_state_internal_game_state.number_of_turn_in_pokemon_battle == 7 and next_state_internal_game_state.number_of_turn_in_pokemon_battle == 8:
            self.negative_reward_for_battle_longer_than_eight_turn = -1
        if current_state_internal_game_state.number_of_turn_in_pokemon_battle == 15 and next_state_internal_game_state.number_of_turn_in_pokemon_battle == 16:
            self.negative_reward_for_battle_longer_than_sixteen_turn = -2
        if current_state_internal_game_state.number_of_turn_in_pokemon_battle == 31 and next_state_internal_game_state.number_of_turn_in_pokemon_battle == 32:
            self.negative_reward_for_battle_longer_than_thirty_two_turn = -4
        
        if current_state_internal_game_state.highest_pokemon_level < next_state_internal_game_state.highest_pokemon_level and next_state_internal_game_state.highest_pokemon_level > external_game_state.max_highest_level_in_the_party_teams:
            self.reward_for_increasing_the_highest_pokemon_level_in_the_team_by_battle = ( next_state_internal_game_state.highest_pokemon_level >- external_game_state.max_highest_level_in_the_party_teams ) * reward_for_increasing_the_highest_pokemon_level_in_the_team_by_battle_coef
        
        if not current_state_internal_game_state.wipe_out and next_state_internal_game_state.wipe_out:
            self.negative_reward_for_wiping_out = -1
        
    def total_reward(self) -> int:
        return sum(asdict(self).values())
    def to_json(self) -> Dict[str , Dict[str , int]]:
        #return asdict(self)
        # make the dictoinary a subkey of reward
        return asdict(self)
    
        
    