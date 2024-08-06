import numpy as np
from enum import Enum
from pokegym.red_memory_player import POKEMON_1_CURRENT_HP, POKEMON_1_MAX_HP
from pokegym.ram_reader.red_memory_opponents import OPPONENT_TRAINER_PARTY_MONSTER_1_STATS_DEFENSE , OPPONENT_TRAINER_PARTY_COUNT
from pokegym.ram_reader.red_memory_world import LAST_BLACK_OUT_MAP_ID
from pyboy.utils import WindowEvent
from typing import List
from pdb import set_trace as T
# addresses from https://datacrystal.romhacking.net/wiki/Pok%C3%A9mon_Red/Blue:RAM_map
# https://github.com/pret/pokered/blob/91dc3c9f9c8fd529bb6e8307b58b96efa0bec67e/constants/event_constants.asm
HP_ADDR =  [0xD16C, 0xD198, 0xD1C4, 0xD1F0, 0xD21C, 0xD248] # This work fine
MAX_HP_ADDR = [0xD18D, 0xD1B9, 0xD1E5, 0xD211, 0xD23D, 0xD269]
PARTY_SIZE_ADDR = 0xD163
PARTY_ADDR = [0xD164, 0xD165, 0xD166, 0xD167, 0xD168, 0xD169]
PARTY_LEVEL_ADDR = [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268] # https://github.com/JumpyWizard-projects/Pokemon-Team-Randomizer-Red-Blue/blob/master/Pokemon%20Team%20Randomizer%20Red%20and%20Blue.lua#L1265
POKE_XP_ADDR = [0xD179, 0xD1A5, 0xD1D1, 0xD1FD, 0xD229, 0xD255]
CAUGHT_POKE_ADDR = range(0xD2F7, 0xD309) # base on the pokemon did you caught the pokemon
SEEN_POKE_ADDR = range(0xD30A, 0xD31D) # base on the pokemon did you seen the pokemon
OPPONENT_LEVEL_ADDR = [0xD8C5, 0xD8F1, 0xD91D, 0xD949, 0xD975, 0xD9A1]
OPPONENT_HP_ADDR = [0xD8C6, 0xD8F2, 0xD91E, 0xD94A, 0xD976, 0xD9A2]
X_POS_ADDR = 0xD362
Y_POS_ADDR = 0xD361
MAP_N_ADDR = 0xD35E
BADGE_1_ADDR = 0xD356
OAK_PARCEL_ADDR = 0xD74E
OAK_POKEDEX_ADDR = 0xD74B
OPPONENT_LEVEL = 0xCFF3
EVENT_FLAGS_START_ADDR = 0xD747
EVENT_FLAGS_END_ADDR = 0xD761
MUSEUM_TICKET_ADDR = 0xD754
MONEY_ADDR_1 = 0xD347
MONEY_ADDR_100 = 0xD348
MONEY_ADDR_10000 = 0xD349
TOTAL_ITEMS_ADDR = 0xD31D
PLAYER_POKEMON_TEAM_ADDR = [0xD16B, 0xD197, 0xD1C3, 0xD1EF, 0xD21B, 0xD247]
HM_ITEMS_ADDR = [0xC4, 0xC5, 0xC6, 0xC7, 0xC8]
FIRST_ITEM_ADDR = 0xD31E
POKEMON_PARTY_MOVES_ADDR = [0xD173,0xD174, 0xD175, 0xD176 , 0xD19F, 0xD1A0, 0xD1A1, 0xD1A2, 0xD1CB, 0xD1CC, 0xD1CD, 0xD1CE, 0xD1F7, 0xD1F8, 0xD1F9, 0xD1FA, 0xD223, 0xD224, 0xD225, 0xD226, 0xD24F, 0xD250, 0xD251, 0xD252]
BATTLE_FLAG = 0xD057
BOOLEAN_FLAG_THAT_INDICATES_THE_GAME_GYM_LEADER_MUSIC_IS_PLAYING = 0xD05C # https://datacrystal.romhacking.net/wiki/Pok%C3%A9mon_Red/Blue:RAM_map#Menu_Data
NUMBER_RUN_ATTEMPTS_ADDR = 0xD120
POKEMONI_PARTY_IDS_ADDR: list[int] = [0xD164, 0xD165, 0xD166, 0xD167, 0xD168, 0xD169]
OPPONENT_PARRTY_IDS_ADDR: list[int] = [0xD89D, 0xD89E, 0xD89F, 0xD8A0, 0xD8A1, 0xD8A2]
NUMBER_RUN_ATTEMPTS_ADDR = 0xD120
LAST_BLACKOUT_MAP = wLastBlackoutMap = 0xD719
BATTLE_RESULT_FLAG = 0XCF0B
MAP_MUSIC_ID = 0xD35B
MAP_MUSIC_ROM_BANK = 0xD35C
NUMBER_OF_TURNS_IN_CURRENT_BATTLE = 0xCCD5
# Battle Turn Info
TURNS_IN_CURRENT_BATTLE = 0xCCD5 # Player + Enemy Move = 1 Turn (Resets only on next battle)
PLAYER_SELECTED_MOVE = 0xCCDC # Stale out of battle
ENEMY_SELECTED_MOVE = 0xCCDD # Stale out of battle
BATTLE_TEXT_PAUSE_FLAG = 0xCC52
# Player Party Overview
PARTY_OFFSET = 0x2C
POKEMON_PARTY_COUNT = 0xD163
POKEMON_1_ID = 0xD164 # ID of mon or 0x00 when none
POKEMON_2_ID = 0xD165 # 0xFF marks end of list, but prev EoL isn't cleared when party size shrinks, must
POKEMON_3_ID = 0xD166 # use LSB as 0xFF marker
POKEMON_4_ID = 0xD167
POKEMON_5_ID = 0xD168
POKEMON_6_ID = 0xD169

# Player Constants
POKEMON_TOTAL_ATTRIBUTES = 20
MAX_MONEY = 999999.0

# Pokemon 1 Details
POKEMON_1 = 0xD16B
POKEMON_1_STATUS = 0xD16F
POKEMON_1_TYPES = (0xD170,0xD171)
POKEMON_1_MOVES_ID = (0xD173, 0xD174, 0xD175, 0xD176)
POKEMON_1_EXPERIENCE = (0xD179, 0xD17A, 0xD17B) # Current XP @ l as 3 hex concat numbers ie. 0x00 0x01 0x080 == 348
POKEMON_1_PP_MOVES = (0xD188, 0xD189, 0xD18A, 0xD18B)
POKEMON_1_LEVEL_ACTUAL = 0xD18C
POKEMON_1_ATTACK = (0xD18F, 0xD190)
POKEMON_1_DEFENSE = (0xD191, 0xD192)
POKEMON_1_SPEED = (0xD193, 0xD194)
POKEMON_1_SPECIAL = (0xD195, 0xD196)

# Player Party Overview
PARTY_OFFSET = 0x2C
POKEMON_PARTY_COUNT = 0xD163
POKEMON_1_ID = 0xD164 # ID of mon or 0x00 when none
POKEMON_2_ID = 0xD165 # 0xFF marks end of list, but prev EoL isn't cleared when party size shrinks, must
POKEMON_3_ID = 0xD166 # use LSB as 0xFF marker
POKEMON_4_ID = 0xD167

LOW_HELATH_ALARM = wLowHealthAlarm = 0xD083

#https://github.com/xinpw8/pokegym/blob/a8b75e4ad2694461f661acf5894d498b69d1a3fa/pokegym/bin/ram_reader/red_memory_battle.py#L50C1-L66C39

# Enemy's Pokémon Stats (In-Battle)
ENEMY_PARTY_SPECIES = (0xD89D, 0xD89E, 0xD89F, 0xD8A0, 0xD8A1, 0xD8A2)  # N/A wild mon, Stale out of battle, 0xFF term
ENEMYS_POKEMON = 0xCFE5 # Enemy/wild current Pokemon, Stale out of battle
ENEMYS_POKEMON_LEVEL = 0xCFF3  # Enemy's level, Stale out of battle
ENEMYS_POKEMON_HP = (0xCFE6, 0xCFE7)  # Enemy's current HP

ENEMYS_POKEMON_STATUS = 0xCFE9  # Enemy's status effects, Stale out of battle
ENEMYS_POKEMON_TYPES = (0xCFEA, 0xCFEB)  # Enemy's type, Stale out of battle
ENEMYS_POKEMON_MOVES = (0xCFED, 0xCFEE, 0xCFEF, 0xCFF0)  # Enemy's moves, Stale out of battle
ENEMYS_POKEMON_INDEX_LEVEL = 0xD8C5  # Enemy's level, use offset to get all
ENEMYS_POKEMON_OFFSET = 0x2C

ENEMY_TRAINER_POKEMON_HP = (0xD8A5, 0xD8A6)  # Only valid for trainers/gyms not wild mons. HP doesn't dec until mon is dead, then it's 0
ENEMY_TRAINER_POKEMON_HP_OFFSET = 0x2C

EVENT_FLAGS_START = 0xD747
EVENTS_FLAGS_LENGTH = 320
MUSEUM_TICKET = (0xD754, 0)
WILD_POKEMON_ENCONTER_RATE_ON_GRASS:int =  0xD887


ENEMY_POKEMON_BASE_EXP_YIELD = 0xD008
ENEMY_MONSTER_ACTUALLY_CATCH_RATE = 0xD007

# Opponent Party Data
ENEMY_PARTY_COUNT = 0xD89C # N/A for wild mon, Stale out of battle
OPPONENT_POKEMON_PARTY_MOVE_ID_ADDRESS = ( 0xD8AC , 0xD8AD, 0xD8AE, 0xD8AF)

# Battles Stuff


class BattleState(Enum):
    NOT_IN_BATTLE = 0
    WILD_BATTLE = 1
    TRAINER_BATTLE = 2
    LOST_BATTLE = 3
class BattleResult(Enum):
    WIN = 0
    LOSE = 1
    DRAW = 2
    IDK = 3


def bcd(num):
    return 10 * ((num >> 4) & 0x0f) + (num & 0x0f)

def bit_count(bits):
    return bin(bits).count('1')

def read_bit(game, addr, bit) -> bool:
    # add padding so zero will read '0b100000000' instead of '0b0'
    return bin(256 + game.memory[addr])[-bit-1] == '1'

def read_uint16(game, start_addr)-> int:
    '''Read 2 bytes'''
    ## Binary you read right to left
    val_256 = game.memory[start_addr]
    val_1 = game.memory[start_addr + 1]
    return 256*val_256 + val_1

def position(game):
    r_pos = game.memory[Y_POS_ADDR]
    c_pos = game.memory[X_POS_ADDR]
    map_n = game.memory[MAP_N_ADDR]
    return r_pos, c_pos, map_n

def party(game):
    party = [game.memory[addr] for addr in PARTY_ADDR]
    party_size = game.memory[PARTY_SIZE_ADDR]
    party_levels = [game.memory[addr] for addr in PARTY_LEVEL_ADDR]
    return party, party_size, party_levels
def get_party_pokemon_level(game)-> list[int]:
    return [game.memory[addr] for addr in PARTY_LEVEL_ADDR]
def get_party_size(game)-> int:
    return game.memory[PARTY_SIZE_ADDR]

def opponent(game):
    return [game.memory[addr] for addr in OPPONENT_LEVEL_ADDR]

def oak_parcel(game):
    return read_bit(game, OAK_PARCEL_ADDR, 1) 

def pokedex_obtained(game):
    return read_bit(game, OAK_POKEDEX_ADDR, 5)
 
def pokemon_seen(game):
    seen_bytes = [game.memory[addr] for addr in SEEN_POKE_ADDR]
    return sum([bit_count(b) for b in seen_bytes])
def total_pokemon_seen(game):
    seen_bytes = [game.memory[addr] for addr in SEEN_POKE_ADDR]
    return sum([bit_count(b) for b in seen_bytes])
def pokemon_see_in_the_pokedex(game):
    '''
    wPokedexSeen:: flag_array NUM_POKEMON
    wPokedexSeenEnd::
    https://github.com/pret/pokered/blob/fc23e72a39eb9cb9ca0651ea805abb6f47ee458c/ram/wram.asm#L1735
    '''
    seen_bytes = [game.memory[addr] for addr in SEEN_POKE_ADDR]
    return  seen_bytes
def get_pokedex_entries_of_caught_pokemon(game):
    # https://datacrystal.romhacking.net/wiki/Pok%C3%A9mon_Red_and_Blue/RAM_map#Menu_Data
    bytes_address_represeting_eight_pokemon_pokedex_entries = [game.memory[addr] for addr in SEEN_POKE_ADDR]
    return bytes_address_represeting_eight_pokemon_pokedex_entries
def pokemon_caught(game):
    '''
    This will calculate how much pokemon you have that complete the pokedex
    '''
    caught_bytes = [game.memory[addr] for addr in CAUGHT_POKE_ADDR]
    return sum([bit_count(b) for b in caught_bytes])

def party_health_ratio(game) -> float:
    '''Percentage of total party HP'''
    party_hp:int = total_party_hit_point(game)
    party_max_hp:int  = total_party_max_hit_point(game)

    # Avoid division by zero if no pokemon
    #sum_max_hp = sum(party_max_hp)
    if party_hp == 0 or party_max_hp == 0:
        return 0
    return party_hp / party_max_hp



def money(game):
    return (100 * 100 * bcd(game.memory[MONEY_ADDR_1])
        + 100 * bcd(game.memory[MONEY_ADDR_100])
        + bcd(game.memory[MONEY_ADDR_10000]))

def check_if_player_has_gym_one_badge(game):
    badges = game.memory[BADGE_1_ADDR]
    return bit_count(badges)

def events(game):
    '''Adds up all event flags, exclude museum ticket'''
    num_events = sum(bit_count(game.memory[i])
        for i in range(EVENT_FLAGS_START_ADDR, EVENT_FLAGS_END_ADDR))
    museum_ticket = int(read_bit(game, MUSEUM_TICKET_ADDR, 0))

    # Omit 13 events by default
    return max(num_events - 13 - museum_ticket, 0)


def total_items(game):
    # https://github.com/pret/pokered/blob/0b20304e6d22baaf7c61439e5e087f2d93f98e39/ram/wram.asm#L1741
    # https://datacrystal.romhacking.net/wiki/Pok%C3%A9mon_Red/Blue:RAM_map#Items
    return game.memory[TOTAL_ITEMS_ADDR]


def total_unique_moves(game):
    # https://datacrystal.romhacking.net/wiki/Pok%C3%A9mon_Red/Blue:RAM_map#Wild_Pok%C3%A9mon
    hash_set = set()
    for pokemon_addr in PLAYER_POKEMON_TEAM_ADDR:
        if game.memory[pokemon_addr] != 0:
            for increment in range(8, 12):
                move_id = game.memory[pokemon_addr + increment]
                if move_id != 0:
                    hash_set.add(move_id)
    return len(hash_set)


def get_items_in_bag(game):
        # total 20 items
        # item1, quantity1, item2, quantity2, ...
        item_ids = []
        for i in range(0, 20, 2):
            item_id = game.memory[FIRST_ITEM_ADDR + i]
            if item_id != 0:
                item_ids.append(item_id)
        return item_ids
def total_hm_party_has(game) -> int:
    # https://github.com/luckytyphlosion/pokered/blob/master/data/moves.asm#L13
    # https://github.com/luckytyphlosion/pokered/blob/c43bd68f01b794f61025ac2e63c9e043634ffdc8/constants/move_constants.asm#L17
    # https://github.com/luckytyphlosion/pokered/blob/c43bd68f01b794f61025ac2e63c9e043634ffdc8/constants/item_constants.asm#L103C1-L109C27
    # https://github.com/xinpw8/pokegym/blob/alpha_pokegym_bill/pokegym/environment.py#L609
    
    total_hm_count = 0
    for hm_iitem_addr in HM_ITEMS_ADDR:
        hm_item_id = game.memory[hm_iitem_addr]
        if hm_item_id != 0:
            total_hm_count += 1
    return total_hm_count
def number_of_pokemon_that_hm_in_move_pool_in_your_part_your_party(game) -> int:
    # https://datacrystal.romhacking.net/wiki/Pok%C3%A9mon_Red/Blue:RAM_map#Player
    
    count = 0
    for pokemon_party_move_addr in POKEMON_PARTY_MOVES_ADDR:
        pokemon_party_move_id = game.memory[pokemon_party_move_addr]
        if pokemon_party_move_id in HM_ITEMS_ADDR:
            count += 1
    return count
def is_in_battle(game):
    # D057
    # 0 not in battle
    # 1 wild battle
    # 2 trainer battle
    # -1 lost battle
    #https://github.com/luckytyphlosion/pokered/blob/c43bd68f01b794f61025ac2e63c9e043634ffdc8/wram.asm#L1629C1-L1634C6
    bflag = game.memory[BATTLE_FLAG]
    try:
        return BattleState(bflag)
    except ValueError:
        # We will solve this error later
        return BattleState.NOT_IN_BATTLE
def pokecenter(game):
    #https://github.com/CJBoey/PokemonRedExperiments1/blob/4024b8793e25a895a07efb07529c5728f076412d/baselines/boey_baselines/red_gym_env.py#L629C2-L635C53
    return 5
def check_if_gym_leader_music_is_playing(game):
    # https://datacrystal.romhacking.net/wiki/Pok%C3%A9mon_Red/Blue:RAM_map#Menu_Data
    return game.memory[BOOLEAN_FLAG_THAT_INDICATES_THE_GAME_GYM_LEADER_MUSIC_IS_PLAYING]
def number_of_attempt_running(game)-> int:
    return game.memory[NUMBER_RUN_ATTEMPTS_ADDR]

def get_party_pokemon_id(self) -> np.array:
    return np.array( [self.memory[single_pokemon_pokemon_id_addr] for single_pokemon_pokemon_id_addr in POKEMONI_PARTY_IDS_ADDR]) 
def get_opponent_party_pokemon_id(self) -> np.array:
    return np.array( [self.memory[single_pokemon_pokemon_id_addr] for single_pokemon_pokemon_id_addr in OPPONENT_PARRTY_IDS_ADDR])

def get_opponent_party_pokemon_hp(self) -> np.array:
    return np.array( [self.memory[single_pokemon_pokemon_hp_addr] for single_pokemon_pokemon_hp_addr in OPPONENT_HP_ADDR])

def set_perfect_iv_dvs(self):
    party_size:int = self.memory[PARTY_SIZE_ADDR]
    for i in [0xD16B, 0xD197, 0xD1C3, 0xD1EF, 0xD21B, 0xD247][:party_size]:
        for m in range(12):  # Number of offsets for IV/DV
            self.pyboy.set_memory_value(i + 17 + m, 0xFF)

def check_if_party_has_cut(self) -> bool:
    party_size:int = self.memory[PARTY_SIZE_ADDR]
    for i in [0xD16B, 0xD197, 0xD1C3, 0xD1EF, 0xD21B, 0xD247][:party_size]:
        for m in range(4):
            if self.pyboy.memory[i + 8 + m] == 15:
                return True
    return False

def check_if_in_start_menu(self) -> bool:
    return (
        self.memory[0xD057] == 0
        and self.memory[0xCF13] == 0
        and self.memory[0xFF8C] == 6
        and self.memory[0xCF94] == 0
    )

def check_if_in_pokemon_menu(self) -> bool:
    return (
        self.memory[0xD057] == 0
        and self.memory[0xCF13] == 0
        and self.memory[0xFF8C] == 6
        and self.memory[0xCF94] == 2
    )

def check_if_in_stats_menu(self) -> bool:
    return (
        self.memory[0xD057] == 0
        and self.memory[0xCF13] == 0
        and self.memory[0xFF8C] == 6
        and self.memory[0xCF94] == 1
    )

def check_if_in_bag_menu(self) -> bool:
    return (
        self.memory[0xD057] == 0
        and self.memory[0xCF13] == 0
        # and self.memory[0xFF8C] == 6 # only sometimes
        and self.memory[0xCF94] == 3
    )

def check_if_cancel_bag_menu(game,  action) -> bool:
    return (
        action == WindowEvent.PRESS_BUTTON_A
        and game.memory[0xD057] == 0
        and game.memory[0xCF13] == 0
        # and self.memory[0xFF8C] == 6
        and game.memory[0xCF94] == 3
        and game.memory[0xD31D] == game.memory[0xCC36] + game.memory[0xCC26]
    )

def check_if_in_overworld(game) -> bool:
    return game.memory[0xD057] == 0 and game.memory[0xCF13] == 0 and game.memory[0xFF8C] == 0

def get_number_of_run_attempts(game) -> int:
    return game.memory[NUMBER_RUN_ATTEMPTS_ADDR]
def get_player_direction(game) -> int:
    # C1x9: facing direction (0: down, 4: up, 8: left, $c: right)
    return game.memory[0xC109] 
def get_battle_result(game)-> BattleResult:
    battle_result_flag = game.memory[BATTLE_RESULT_FLAG]
    try:
        return BattleResult(battle_result_flag)
    except ValueError:
        return BattleResult.IDK
## https://github.com/luckytyphlosion/pokered/blob/c43bd68f01b794f61025ac2e63c9e043634ffdc8/wram.asm#L2361C1-L2368C1
def get_map_music_id(game):
    return game.memory[MAP_MUSIC_ID]
def get_map_music_rom_bank(game):
    return game.memory[MAP_MUSIC_ROM_BANK]
def get_last_pokecenter_id(game , pokecenter_ids) -> int:
    last_pokecenter = game.memory[LAST_BLACKOUT_MAP]
    # will throw error if last_pokecenter not in pokecenter_ids, intended
    if last_pokecenter == 0:
        # no pokecenter visited yet
        return -1
    if last_pokecenter not in pokecenter_ids:
        print(f'\nERROR: last_pokecenter: {last_pokecenter} not in pokecenter_ids')
        return -1
    else:
        return pokecenter_ids.index(last_pokecenter)
def get_number_of_turns_in_current_battle(game):
    return game.memory[NUMBER_OF_TURNS_IN_CURRENT_BATTLE]
'''
That guy  code ideas
def get_pokemon_health(self , offset):
        hp_total = (self.env.ram_interface.read_memory(POKEMON_1_MAX_HP[0] + offset) << 8) + self.env.ram_interface.read_memory(POKEMON_1_MAX_HP[1] + offset)
        hp_avail = (self.env.ram_interface.read_memory(POKEMON_1_CURRENT_HP[0] + offset) << 8) + self.env.ram_interface.read_memory(POKEMON_1_CURRENT_HP[1] + offset)
        return hp_total, hp_avail
'''
def get_each_pokemon_max_hit_points(game):
    return [read_uint16(game, addr) for addr in MAX_HP_ADDR]
def each_pokemon_hit_points(game):
    '''Percentage of total party HP'''
    return [read_uint16(game, addr) for addr in HP_ADDR]
def total_party_hit_point(game) -> int:
    '''Percentage of total party HP'''
    party_hp = [read_uint16(game, addr) for addr in HP_ADDR]
    return sum(party_hp)
def total_party_max_hit_point(game) -> int:
    '''Percentage of total party HP'''
    party_max_hp = [read_uint16(game, addr) for addr in MAX_HP_ADDR]
    return sum(party_max_hp)
def get_battle_turn_moves(game):
        player_selected_move = game.memory[PLAYER_SELECTED_MOVE]
        enemy_selected_move = game.memory[ENEMY_SELECTED_MOVE]

        return player_selected_move, enemy_selected_move
def read_memory(game , address):
    return game.memory[address]
def get_pokemon_xp(game, offset):
    xp = ((read_memory( game , POKEMON_1_EXPERIENCE[0] + offset) << 16) +
            (read_memory( game , POKEMON_1_EXPERIENCE[1] + offset) << 8) +
            read_memory( game , POKEMON_1_EXPERIENCE[2] + offset))

    return xp
def _get_lineup_size(game):
    return read_memory( game , POKEMON_PARTY_COUNT)
def get_player_lineup_xp(game ):
    return [get_pokemon_xp( game , i * PARTY_OFFSET) for i in range(_get_lineup_size(game))]
def get_low_health_alarm(game):
    return read_memory( game , LOW_HELATH_ALARM)
def get_opponent_pokemon_levels(game) -> List[int]:
    opponent_level_addr = [0] * 6
    for index , addr in enumerate(OPPONENT_LEVEL_ADDR):
        opponent_level_addr[index] = game.memory[addr]
    return opponent_level_addr

def get_enemys_pokemon_hp(game)-> int:
    # return read_uint16(game, addr) for addr in ENEMYS_POKEMON_HP
    return read_uint16(game, ENEMYS_POKEMON_HP[0])
    

def get_enemy_trainer_pokemon_hp(game)-> List[int]:
    enemy_trainer_pokemon_hp = [0] * 6
    for index in range(0 , get_party_size(game)):
        enemy_trainer_pokemon_hp[index] = 256*game.memory[ENEMY_TRAINER_POKEMON_HP[0] + ( index * ENEMY_TRAINER_POKEMON_HP_OFFSET )] + game.memory[ENEMY_TRAINER_POKEMON_HP[1] + ( index * ENEMY_TRAINER_POKEMON_HP_OFFSET )]
    return enemy_trainer_pokemon_hp
def get_enemy_trainer_current_pokemon_hp(game)-> List[int]:
    enemy_trainer_pokemon_hp = [0] * 6
    for index in range(0 , get_party_size(game)):
        enemy_trainer_pokemon_hp[index] = 256*game.memory[ENEMY_TRAINER_POKEMON_HP[0] + ( index * ENEMY_TRAINER_POKEMON_HP_OFFSET )] + game.memory[ENEMY_TRAINER_POKEMON_HP[1] + ( index * ENEMY_TRAINER_POKEMON_HP_OFFSET )]
    return enemy_trainer_pokemon_hp
def total_events_that_occurs_in_game(game):
    # adds up all event flags, exclude museum ticket
    return max(
        sum(
            [
                game.memory[i].bit_count()
                for i in range(EVENT_FLAGS_START, EVENT_FLAGS_START + EVENTS_FLAGS_LENGTH)
            ]
        )
        - int(read_bit(game, MUSEUM_TICKET_ADDR, 0)),
        0,
    )
#def total_envets_representing_one_hot_encoding(game):
    
    
def get_pokemon_pp_avail(game) -> List[int]:
    pp_teams:list[int] = [0] * 24
    for index in range(0 , get_party_size(game)):
        pp_teams[index * 4 + 0] = game.memory[POKEMON_1_PP_MOVES[0] + (index * PARTY_OFFSET)]
        pp_teams[index * 4 + 1] = game.memory[POKEMON_1_PP_MOVES[1] + (index * PARTY_OFFSET)]
        pp_teams[index * 4 + 2] = game.memory[POKEMON_1_PP_MOVES[2] + (index * PARTY_OFFSET)]
        pp_teams[index * 4 + 3] = game.memory[POKEMON_1_PP_MOVES[3] + (index * PARTY_OFFSET)]
    return pp_teams

def wild_pokemon_encounter_rate_on_grass(game):
    return game.memory[WILD_POKEMON_ENCONTER_RATE_ON_GRASS]

def get_pokemon_party_move_ids(game , party_size):
    pokemon_party_move_ids = [0] *24
    # https://gamefaqs.gamespot.com/gameboy/367023-pokemon-red-version/faqs/74734#section8
    assert PARTY_OFFSET == 44
    try: 
        for index in range( 0 , party_size):
            
            pokemon_party_move_ids[ index * 4 + 0 ] = game.memory[ POKEMON_1_MOVES_ID[0] + ( index * PARTY_OFFSET ) ]
            pokemon_party_move_ids[ index * 4 + 1 ] = game.memory[ POKEMON_1_MOVES_ID[1] + ( index * PARTY_OFFSET ) ]
            pokemon_party_move_ids[ index * 4 + 2 ] = game.memory[ POKEMON_1_MOVES_ID[2] + ( index * PARTY_OFFSET ) ]
            pokemon_party_move_ids[ index * 4 + 3 ] = game.memory[ POKEMON_1_MOVES_ID[3] + ( index * PARTY_OFFSET ) ]
    except Exception as e:
        print(e)
        T()
    return pokemon_party_move_ids 
#def get_Opponent Party Data
def total_number_of_enemy_pokemon_in_opponent_party(game):
    return game.memory[ENEMY_PARTY_COUNT]
def get_opponent_party_move_id(game , party_size):
    # https://gamefaqs.gamespot.com/gameboy/367023-pokemon-red-version/faqs/74734#section5
    opponent_party_move_ids: List[int] = [0] * 24
    opponent_party_uniques_moves_id = set()
    for index in range( 0 , party_size ):
        opponent_party_move_ids[ index * 4 + 0 ] = game.memory[OPPONENT_POKEMON_PARTY_MOVE_ID_ADDRESS[0] + ( index * PARTY_OFFSET ) ]
        opponent_party_move_ids[ index * 4 + 1 ] = game.memory[ OPPONENT_POKEMON_PARTY_MOVE_ID_ADDRESS[1] + ( index * PARTY_OFFSET ) ]
        opponent_party_move_ids[ index * 4 + 2 ] = game.memory[ OPPONENT_POKEMON_PARTY_MOVE_ID_ADDRESS[2] + ( index * PARTY_OFFSET ) ]
        opponent_party_move_ids[ index * 4 + 3 ] = game.memory[ OPPONENT_POKEMON_PARTY_MOVE_ID_ADDRESS[3] + ( index * PARTY_OFFSET ) ]
    return opponent_party_move_ids 

#def get_opponent_party_defense_stats(game):
#    opponent_party_defense_stats: List[int] = [0] * 24
#    for index in range( 0 , total_number_of_enemy_pokemon_in_opponent_party(game)):
    
        
    

def get_enemy_pokemon_base_exp_yield(game):
    # https://github.com/pret/pokered/blob/095c7d7227ea958c1afa76765c044793b9e8dc5a/pokered.sym#L18619C1-L18620C25
    return game.memory[ ENEMY_POKEMON_BASE_EXP_YIELD]

def get_enemy_monster_actually_catch_rate(game):
    # https://github.com/pret/pokered/blob/095c7d7227ea958c1afa76765c044793b9e8dc5a/pokered.sym#L18618
    return game.memory[ ENEMY_MONSTER_ACTUALLY_CATCH_RATE]

def get_opponent_trainer_party_count(game):
    return game.memory[OPPONENT_TRAINER_PARTY_COUNT]

def get_opponent_trainer_party_monster_stats_defense(game):
    defense_stats = [0] * 6
    for index in range(0 , get_opponent_trainer_party_count(game)):
        defense_stats[index] = 256*game.memory[OPPONENT_TRAINER_PARTY_MONSTER_1_STATS_DEFENSE[0] + ( index * PARTY_OFFSET )] + game.memory[OPPONENT_TRAINER_PARTY_MONSTER_1_STATS_DEFENSE[1] + ( index * PARTY_OFFSET )]
    return defense_stats

def get_last_black_out_map_id(game):
    return game.memory[LAST_BLACK_OUT_MAP_ID]
# Battle Stuff
from pokegym.ram_reader.red_memory_battle_stats import PLAYER_MONSTER_STATS_MODIFIER_ATTACK , PLAYER_MONSTER_STATS_MODIFIER_DEFENSE , PLAYER_MONSTER_STATS_MODIFIER_SPEED , PLAYER_MONSTER_STATS_MODIFIER_SPECIAL , PLAYER_MONSTER_STATS_MODIFIER_ACCURACY
def get_player_current_monster_modifier_attack(game):
    return game.memory[PLAYER_MONSTER_STATS_MODIFIER_ATTACK]
def get_player_current_monster_modifier_defense(game):
    return game.memory[PLAYER_MONSTER_STATS_MODIFIER_DEFENSE]
def get_player_current_monster_modifier_speed(game):
    return game.memory[PLAYER_MONSTER_STATS_MODIFIER_SPEED]
def get_player_current_monster_modifier_special(game):
    return game.memory[PLAYER_MONSTER_STATS_MODIFIER_SPECIAL]
def get_player_current_monster_modifier_accuracy(game):
    return game.memory[PLAYER_MONSTER_STATS_MODIFIER_ACCURACY]
from pokegym.ram_reader.red_memory_battle_stats import ENEMY_CURRENT_POKEMON_STATS_MODIFIER_ATTACK , ENEMY_CURRENT_POKEMON_STATS_MODIFIER_DEFENSE , ENEMY_CURRENT_POKEMON_STATS_MODIFIER_SPEED , ENEMY_CURRENT_POKEMON_STATS_MODIFIER_SPECIAL , ENEMY_CURRENT_POKEMON_STATS_MODIFIER_ACCURACY , ENEMY_CURRENT_POKEMON_STATS_MODIFIER_EVASTION
def get_enemy_current_monster_modifier_attack(game):
    return game.memory[ENEMY_CURRENT_POKEMON_STATS_MODIFIER_ATTACK]
def get_enemy_current_monster_modifier_defense(game):
    return game.memory[ENEMY_CURRENT_POKEMON_STATS_MODIFIER_DEFENSE]
def get_enemy_current_monster_modifier_speed(game):# -> Any:
    return game.memory[ENEMY_CURRENT_POKEMON_STATS_MODIFIER_SPEED]
def get_enemy_current_monster_modifier_special(game):
    return game.memory[ENEMY_CURRENT_POKEMON_STATS_MODIFIER_SPECIAL]
def get_enemy_current_monster_modifier_accuracy(game):
    return game.memory[ENEMY_CURRENT_POKEMON_STATS_MODIFIER_ACCURACY]
def get_enemy_current_monster_modifier_evastion(game):
    return game.memory[ENEMY_CURRENT_POKEMON_STATS_MODIFIER_EVASTION]
from pokegym.ram_reader.red_memory_battle_stats import ENEMY_CURRENT_POKEMON_LEVEL , PLAYER_CURRENT_POKEMON_LEVEL
def get_player_current_pokemon_level(game):
    return game.memory[PLAYER_CURRENT_POKEMON_LEVEL]
def get_enemy_current_pokemon_level(game):
    return game.memory[ENEMY_CURRENT_POKEMON_LEVEL]
from pokegym.ram_reader.red_memory_battle_stats import ENEMY_MOVE_EFFECT
def get_enemy_move_effect(game):
    return game.memory[ENEMY_MOVE_EFFECT]
from pokegym.ram_reader.red_memory_battle_stats import ENEMY_POKEMON_MOVE_POWER
def get_enemy_move_effect_target_address(game):
    return game.memory[ENEMY_POKEMON_MOVE_POWER]
from pokegym.ram_reader.red_memory_battle_stats import ENEMY_POKEMON_MOVE_TYPE , ENEMY_POKEMON_MOVE_ACCURACY
def get_enemy_pokemon_move_type(game):
    return game.memory[ENEMY_POKEMON_MOVE_TYPE]
def get_enemy_pokemon_move_accuracy(game):
    return game.memory[ENEMY_POKEMON_MOVE_ACCURACY]
from pokegym.ram_reader.red_memory_world import CURRENT_MAP_ID
def get_current_map_id(game):
    return game.memory[CURRENT_MAP_ID]
from pokegym.ram_reader.red_memory_battle_stats import ENEMY_POKEMON_MOVE_MAX_PP
def get_enemy_pokemon_move_max_pp(game):
    return game.memory[ENEMY_POKEMON_MOVE_MAX_PP]
def get_enemy_pokemon_level(game):
    return game.memory[ENEMYS_POKEMON_LEVEL]
def get_wGainBoostedExp(game):
    return game.memory[
        game.symbol_lookup(
            "wGainBoostedExp"
        )
    ]
def get_w_exp_amount_gained(game):
    return game.memory[
        game.symbol_lookup(
            "wExpAmountGained"
        )
    ]
from pokegym.ram_reader.red_memory_opponents import OPPOENT_TRAINER_PARTY_MONSTER_1_MAX_HP

def get_enemy_trainer_max_hp(game):
    enemy_trainer_max_hp = [0]* 6
    for index in range(0 , get_opponent_trainer_party_count(game)):
        enemy_trainer_max_hp[index] = 256*game.memory[OPPOENT_TRAINER_PARTY_MONSTER_1_MAX_HP[0] + ( index * PARTY_OFFSET )] + game.memory[OPPOENT_TRAINER_PARTY_MONSTER_1_MAX_HP[1] + ( index * PARTY_OFFSET )]
    return enemy_trainer_max_hp
        
def number_of_dead_pokemon_in_opponent_trainer_party(game):
    # Becuase we know the max hpa nd currrent hp
    enemy_trainer_max_hp = get_enemy_trainer_max_hp(game)
    enemy_trainer_current_hp = get_enemy_trainer_current_pokemon_hp(game)
    number_of_dead_pokemon_in_enemy_trainer_party = 0
    for index in range(0 , get_opponent_trainer_party_count(game)):
        if enemy_trainer_current_hp[index] == 0 and enemy_trainer_max_hp[index] > 0:
            number_of_dead_pokemon_in_enemy_trainer_party += 1
    return number_of_dead_pokemon_in_enemy_trainer_party