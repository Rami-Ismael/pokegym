from pdb import set_trace as T
from dataclasses import dataclass
from pokegym.ram_reader.red_memory_items import BAG_TOTAL_ITEMS


class PyBoyRAMInterface:
    def __init__(self, pyboy):
        self.pyboy = pyboy

    def read_memory(self, address):
        return self.pyboy.get_memory_value(address)

    def write_memory(self, address, value):
        return self.pyboy.set_memory_value(address, value)

@dataclass
class Game:
    
    def __init__(self, pyboy):
        self.ram_interface = PyBoyRAMInterface(pyboy)
        try:
            self.items = Items()
        except Exception as e:
            print(e) 
            T()
        self.player = Player(self)


@dataclass
class Items:
    def __int__(self , env):
        self.env = env
    def get_bag_item_count(self):
        return self.env.ram_interface.read_memory(BAG_TOTAL_ITEMS)

@dataclass
class Player:
    def __init__(self, env):
        self.env = env