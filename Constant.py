BLACK = (0, 0, 0)
WHITE = (200, 200, 200)

MAP_H = 10
MAP_W = 10
BLOCKSIZE = 100 # for gridsize 
WINDOW_H = MAP_H * BLOCKSIZE
WINDOW_W = MAP_W * BLOCKSIZE


# NOTE: Tile info (player number, actor type, health_points, is_carrying_gold)
    # Player 2 empty, 0 and 1 are players 3 bits
    
NO_PLAYER = 2    

# Initial hp 4 bits
TC_HP = 4
VILLAGER_HP = 2
TROOP_HP = 3
GOLD_HP = -1
BARRACK_HP = 3

VILLAGER_COST = 1
TROOP_COST = 1
TC_COST = 4
BARRACK_COST = 3
    
MAX_PLAYERS = 4
MAX_ACTORS = 8
MAX_HP = 15
CARRY_CAPACITY = 10

# Actor type 4 bits
EMPTY_TYPE = 5
TC_TYPE = 0
VILLAGER_TYPE = 1
BARRACK_TYPE = 2
TROOP_TYPE = 3
GOLD_TYPE = 4

# character representation for each unit type
ASCII_CHARS = {
    TC_TYPE: 'T',       
    BARRACK_TYPE: 'B',  
    VILLAGER_TYPE: 'V',
    TROOP_TYPE: 'S',    
    GOLD_TYPE: 'G',     
    EMPTY_TYPE: '.',    
}

# colors for specific players
PLAYER_COLORS = {
    0: (255, 100, 100), # p0: red
    1: (100, 100, 255), # p1: blue
    NO_PLAYER: (200, 200, 200) # grey
}



# NOTE: The model will return an action map of action to take for each actor on every tile.
ACT_STAY  = 0  
ACT_UP    = 1
ACT_DOWN  = 2
ACT_LEFT  = 3
ACT_RIGHT = 4
TURN_TC = 5 # Turn villager to TC
TURN_BARRACK = 6

NUM_ACTIONS = 7

# +1 are hp and carry gold
CHANNEL_NUM = MAX_ACTORS + MAX_PLAYERS + 1 + 1

LEFT_MAIN_TC_POS = (3, 4)
RIGHT_MAIN_TC_POS = (6, 4)
