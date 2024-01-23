from pdb import set_trace as T
from gymnasium import Env, spaces
import numpy as np
import os
from pokegym import pyboy_binding
from rich import print

from pokegym.pyboy_binding import (ACTIONS, make_env, open_state_file,
    load_pyboy_state, run_action_on_emulator)
from pokegym import ram_map, game_map
from pokegym.environment import Environment

from pyboy import PyBoy
from pyboy import logger
from pyboy.utils import WindowEvent
from rich import print
from pokegym.environment import play



env = Environment(rom_path='pokemon_red.gb', state_path=None, headless=False,
    disable_input=False, sound=False, sound_emulated=False, verbose=True
)
env.reset()
# while True:
        # # Get input from pyboy's get_input method
        # env.step(int(user_input))
        # env.render()
play()