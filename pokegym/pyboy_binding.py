from pdb import set_trace as T
from io import BytesIO
import os

from pyboy import PyBoy
from pyboy.utils import WindowEvent

try:
    from pyboy import logger
    logger.logger.setLevel('ERROR')
except Exception as e:
    print(e)
    pass


class Down:
    PRESS = WindowEvent.PRESS_ARROW_DOWN
    RELEASE = WindowEvent.RELEASE_ARROW_DOWN

class Left:
    PRESS = WindowEvent.PRESS_ARROW_LEFT
    RELEASE = WindowEvent.RELEASE_ARROW_LEFT

class Right:
    PRESS = WindowEvent.PRESS_ARROW_RIGHT
    RELEASE = WindowEvent.RELEASE_ARROW_RIGHT

class Up:
    PRESS = WindowEvent.PRESS_ARROW_UP
    RELEASE = WindowEvent.RELEASE_ARROW_UP

class A:
    PRESS = WindowEvent.PRESS_BUTTON_A
    RELEASE = WindowEvent.RELEASE_BUTTON_A

class B:
    PRESS = WindowEvent.PRESS_BUTTON_B
    RELEASE = WindowEvent.RELEASE_BUTTON_B

class Start:
    PRESS = WindowEvent.PRESS_BUTTON_START
    RELEASE = WindowEvent.RELEASE_BUTTON_START

class Select:
    PRESS = WindowEvent.PRESS_BUTTON_SELECT
    RELEASE = WindowEvent.RELEASE_BUTTON_SELECT

# TODO: Add start button to actions when we need it
VALID_ACTION = ACTIONS = (Down, Left, Right, Up, A, B)

def make_env(gb_path, headless=True, quiet=False):
    assert os.path.exists(gb_path), f"Could not find {gb_path}"
    assert os.path.exists(os.path.join(os.path.dirname(__file__), "pokered.sym")), "Could not find pokered.sym"  
    gb_path='pokemon_red.gb'
    game = PyBoy(
        gb_path,
        sound = False , 
        window_type='null' , 
        log_level = "CRITICAL" , 
        symbols=os.path.join(os.path.dirname(__file__), "pokered.sym"),
    )

    #screen = game.botsupport_manager().screen()

    if not headless:
        game.set_emulation_speed(6)

    return game

def open_state_file(path):
    '''Load state file with BytesIO so we can cache it'''
    with open(path, 'rb') as f:
        initial_state = BytesIO(f.read())

    return initial_state

def load_pyboy_state(pyboy, state):
    '''Reset state stream and load it into PyBoy'''
    state.seek(0)
    pyboy.load_state(state)

def run_action_on_emulator_old(pyboy, screen, action,
        headless=True, fast_video=True, frame_skip=24):
    '''Sends actions to PyBoy'''
    press, release = action.PRESS, action.RELEASE
    pyboy.send_input(press)

    if headless or fast_video:
        pyboy._rendering(False)

    frames = []
    for i in range(frame_skip):
        if i == 8: # Release button after 8 frames
            pyboy.send_input(release)
        if not fast_video: # Save every frame
            frames.append(screen.screen_ndarray())
        if i == frame_skip - 1:
            pyboy._rendering(True)
        pyboy.tick()

    if fast_video: # Save only the last frame
        frames.append(screen.screen_ndarray())
