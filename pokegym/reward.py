from typing import Dict
from dataclasses import asdict, dataclass

@dataclass
class Reward:
    #use max to make sure the agent cannot gain the system with the PC
    reward_for_increasing_the_max_size_of_the_trainer_team:int = 0
    
    # Reward
    reward_for_doing_new_events_that_occurs_in_game_calculating_by_game_state:int = 0
    reward_for_doing_new_events_that_occurs_in_game_calculating_by_external_game_state:int = 0 
    
    # Level which can be effected by the PC or daycare we want the max value
    reward_for_increasing_the_total_party_level:int = 0
    
    # Reward for seeing new pokemon 
    #reward_for_seeing_new_pokemon_in_the_pokedex:int = 0
    
    reward_for_taking_action_that_start_playing_the_gym_player_music:int = 0
    
    def __init__(self, current_state_internal_game_state , next_state_internal_game_state , external_game_state ,  
                 reward_for_increase_pokemon_level_coef:float = 1.1
                 ):
        
        if current_state_internal_game_state.party_size < next_state_internal_game_state.party_size and next_state_internal_game_state.party_size > external_game_state.max_party_size:
            self.reward_for_increasing_the_max_size_of_the_trainer_team = 1
        # Events
        if current_state_internal_game_state.total_events_that_occurs_in_game < next_state_internal_game_state.total_events_that_occurs_in_game:
            self.reward_for_doing_new_events_that_occurs_in_game_calculating_by_game_state += next_state_internal_game_state.total_events_that_occurs_in_game - current_state_internal_game_state.total_events_that_occurs_in_game
        if external_game_state.total_events_that_occurs_in_game < next_state_internal_game_state.total_events_that_occurs_in_game:
            self.reward_for_doing_new_events_that_occurs_in_game_calculating_by_external_game_state += next_state_internal_game_state.total_events_that_occurs_in_game - external_game_state.total_events_that_occurs_in_game
        
        if current_state_internal_game_state.total_party_level < next_state_internal_game_state.total_party_level and next_state_internal_game_state.total_party_level > external_game_state.max_total_party_level:
            self.reward_for_increasing_the_total_party_level =  ( next_state_internal_game_state.total_party_level - current_state_internal_game_state.total_party_level  ) * reward_for_increase_pokemon_level_coef
        
        
        if not current_state_internal_game_state.gym_leader_music_is_playing and next_state_internal_game_state.gym_leader_music_is_playing:
            self.reward_for_taking_action_that_start_playing_the_gym_player_music = 2
        
        
    def total_reward(self) -> int:
        return sum(asdict(self).values())
    def to_json(self) -> Dict[str , Dict[str , int]]:
        #return asdict(self)
        # make the dictoinary a subkey of reward
        return {"reward":asdict(self)}
    
        
    