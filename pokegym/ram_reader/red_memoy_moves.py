import numpy as np
# LIST of BAD Moves
GROWL_HEX_MOVE_ID = 0x2D
TAIL_WIPE_HEX_MOVE_ID = 0x27
LEER_HEX_MOVE_ID = 0x2B

GROWL_DECIMAL_VALUE_OF_MOVE_ID = 45
TAIL_DECIAML_VALUE_OF_MOVE_ID = 39
LEER_DECIMAL_VALUE_OF_MOVE_ID = 43
EXPLOSION_DECIMAL_VALUE_OF_MOVE_ID = 153

LIST_OF_BAD_MOVES_ID = [
    GROWL_DECIMAL_VALUE_OF_MOVE_ID,
    TAIL_DECIAML_VALUE_OF_MOVE_ID,
    LEER_DECIMAL_VALUE_OF_MOVE_ID,
]
def create_a_list_of_good_moves():
    # Create a numpy array between the values of 1 to 161
    numbers = np.arange(1, 161)
    # I want to remove the values of 45 , 39 , 43 from numbers
    numbers = np.delete(numbers, LIST_OF_BAD_MOVES_ID)
    
    return numbers