from enum import Enum
# addresses from https://datacrystal.romhacking.net/wiki/Pok%C3%A9mon_Red/Blue:RAM_map
# https://github.com/pret/pokered/blob/91dc3c9f9c8fd529bb6e8307b58b96efa0bec67e/constants/event_constants.asm
HP_ADDR =  [0xD16C, 0xD198, 0xD1C4, 0xD1F0, 0xD21C, 0xD248]
MAX_HP_ADDR = [0xD18D, 0xD1B9, 0xD1E5, 0xD211, 0xD23D, 0xD269]
PARTY_SIZE_ADDR = 0xD163
PARTY_ADDR = [0xD164, 0xD165, 0xD166, 0xD167, 0xD168, 0xD169]
PARTY_LEVEL_ADDR = [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]
POKE_XP_ADDR = [0xD179, 0xD1A5, 0xD1D1, 0xD1FD, 0xD229, 0xD255]
CAUGHT_POKE_ADDR = range(0xD2F7, 0xD309) # base on the pokemon did you caught the pokemon
SEEN_POKE_ADDR = range(0xD30A, 0xD31D) # base on the pokemon did you seen the pokemon
OPPONENT_LEVEL_ADDR = [0xD8C5, 0xD8F1, 0xD91D, 0xD949, 0xD975, 0xD9A1]
X_POS_ADDR = 0xD362
Y_POS_ADDR = 0xD361
MAP_N_ADDR = 0xD35E
BADGE_1_ADDR = 0xD356
OAK_PARCEL_ADDR = 0xD74E
OAK_POKEDEX_ADDR = 0xD74B
OPPONENT_LEVEL = 0xCFF3
ENEMY_POKE_COUNT = 0xD89C
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

class BattleState(Enum):
    NOT_IN_BATTLE = 0
    WILD_BATTLE = 1
    TRAINER_BATTLE = 2
    LOST_BATTLE = -1


def bcd(num):
    return 10 * ((num >> 4) & 0x0f) + (num & 0x0f)

def bit_count(bits):
    return bin(bits).count('1')

def read_bit(game, addr, bit) -> bool:
    # add padding so zero will read '0b100000000' instead of '0b0'
    return bin(256 + game.get_memory_value(addr))[-bit-1] == '1'

def read_uint16(game, start_addr):
    '''Read 2 bytes'''
    val_256 = game.get_memory_value(start_addr)
    val_1 = game.get_memory_value(start_addr + 1)
    return 256*val_256 + val_1

def position(game):
    r_pos = game.get_memory_value(Y_POS_ADDR)
    c_pos = game.get_memory_value(X_POS_ADDR)
    map_n = game.get_memory_value(MAP_N_ADDR)
    return r_pos, c_pos, map_n

def party(game):
    party = [game.get_memory_value(addr) for addr in PARTY_ADDR]
    party_size = game.get_memory_value(PARTY_SIZE_ADDR)
    party_levels = [game.get_memory_value(addr) for addr in PARTY_LEVEL_ADDR]
    return party, party_size, party_levels

def opponent(game):
    return [game.get_memory_value(addr) for addr in OPPONENT_LEVEL_ADDR]

def oak_parcel(game):
    return read_bit(game, OAK_PARCEL_ADDR, 1) 

def pokedex_obtained(game):
    return read_bit(game, OAK_POKEDEX_ADDR, 5)
 
def pokemon_seen(game):
    seen_bytes = [game.get_memory_value(addr) for addr in SEEN_POKE_ADDR]
    return sum([bit_count(b) for b in seen_bytes])

def pokemon_caught(game):
    '''
    This will calculate how much pokemon you have that complete the pokedex
    '''
    caught_bytes = [game.get_memory_value(addr) for addr in CAUGHT_POKE_ADDR]
    return sum([bit_count(b) for b in caught_bytes])

def hp(game):
    '''Percentage of total party HP'''
    party_hp = [read_uint16(game, addr) for addr in HP_ADDR]
    party_max_hp = [read_uint16(game, addr) for addr in MAX_HP_ADDR]

    # Avoid division by zero if no pokemon
    sum_max_hp = sum(party_max_hp)
    if sum_max_hp == 0:
        return 1

    return sum(party_hp) / sum_max_hp

def money(game):
    return (100 * 100 * bcd(game.get_memory_value(MONEY_ADDR_1))
        + 100 * bcd(game.get_memory_value(MONEY_ADDR_100))
        + bcd(game.get_memory_value(MONEY_ADDR_10000)))

def badges(game):
    badges = game.get_memory_value(BADGE_1_ADDR)
    return bit_count(badges)

def events(game):
    '''Adds up all event flags, exclude museum ticket'''
    num_events = sum(bit_count(game.get_memory_value(i))
        for i in range(EVENT_FLAGS_START_ADDR, EVENT_FLAGS_END_ADDR))
    museum_ticket = int(read_bit(game, MUSEUM_TICKET_ADDR, 0))

    # Omit 13 events by default
    return max(num_events - 13 - museum_ticket, 0)


def total_items(game) -> int:
    # https://github.com/pret/pokered/blob/0b20304e6d22baaf7c61439e5e087f2d93f98e39/ram/wram.asm#L1741
    # https://datacrystal.romhacking.net/wiki/Pok%C3%A9mon_Red/Blue:RAM_map#Items
    return game.get_memory_value(TOTAL_ITEMS_ADDR)


def total_unique_moves(game) -> int:
    # https://datacrystal.romhacking.net/wiki/Pok%C3%A9mon_Red/Blue:RAM_map#Wild_Pok%C3%A9mon
    hash_set = set()
    for pokemon_addr in PLAYER_POKEMON_TEAM_ADDR:
        if game.get_memory_value(pokemon_addr) != 0:
            for increment in range(8, 12):
                move_id = game.get_memory_value(pokemon_addr + increment)
                if move_id != 0:
                    hash_set.add(move_id)
    return len(hash_set)


def get_items_in_bag(game):
        # total 20 items
        # item1, quantity1, item2, quantity2, ...
        item_ids = []
        for i in range(0, 20, 2):
            item_id = game.get_memory_value(FIRST_ITEM_ADDR + i)
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
        hm_item_id = game.get_memory_value(hm_iitem_addr)
        if hm_item_id != 0:
            total_hm_count += 1
    return total_hm_count
def number_of_pokemon_that_hm_in_move_pool_in_your_part_your_party(game) -> int:
    # https://datacrystal.romhacking.net/wiki/Pok%C3%A9mon_Red/Blue:RAM_map#Player
    
    count = 0
    for pokemon_party_move_addr in POKEMON_PARTY_MOVES_ADDR:
        pokemon_party_move_id = game.get_memory_value(pokemon_party_move_addr)
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
    bflag = game.get_memory_value(BATTLE_FLAG)
    try:
        return BattleState(bflag)
    except ValueError as e:
        # We will solve this error later
        return BattleState.NOT_IN_BATTLE
def pokecenter(game):
    #https://github.com/CJBoey/PokemonRedExperiments1/blob/4024b8793e25a895a07efb07529c5728f076412d/baselines/boey_baselines/red_gym_env.py#L629C2-L635C53
    return 5