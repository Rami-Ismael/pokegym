from pdb import set_trace as T
from dataclasses import dataclass

@dataclass
class RedGymPlayer:
    
    def __init__(self , env):
        self.env = env
    
    def obs_bad_quanties(self):
        try:
            return self.env.items.get_bag_item_count() / 20 
        except Exception as e:
            print(e)
            T()
            

        