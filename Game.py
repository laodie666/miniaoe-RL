import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pygame
import random
import numpy as np

BLACK = (0, 0, 0)
WHITE = (200, 200, 200)

MAP_H = 10
MAP_W = 10
BLOCKSIZE = 20 # for gridsize 
WINDOW_H = MAP_H * BLOCKSIZE
WINDOW_W = MAP_W * BLOCKSIZE


# NOTE: Tile info (player number, actor type, health_points, is_carrying_gold)
    # Player 2 empty, 0 and 1 are players 3 bits
    
NO_PLAYER = 2    

# Initial hp 4 bits
TC_HP = 10
VILLAGER_HP = 2
TROOP_HP = 3
GOLD_HP = -1
BARRACK_HP = 5

VILLAGER_COST = 1
TROOP_COST = 2
TC_COST = 5
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

class Player():
    
    def __init__(self, side):
        self.side = side
    
    def getAction(self):
        pass

# TODO actor critic system

# Need some adjustment to the game features.
class PolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(CHANNEL_NUM, 8, 3)
        self.conv2 = nn.Conv2d(8, 16, 3)
        
        self.fc1 = nn.Linear((MAP_W - 4) * (MAP_H - 4) * 16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, MAP_W * MAP_H * NUM_ACTIONS)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, 1) 
        
        x = F.relu(self.fc1(x))
        x = F.dropout(x, 0.2)
        
        x = F.relu(self.fc2(x))
        x = F.dropout(x, 0.2)
        
        x = self.fc3(x)
        
        x = x.view(MAP_W, MAP_H, NUM_ACTIONS)
        
        return x
class CriticNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(CHANNEL_NUM, 8, 3)
        self.conv2 = nn.Conv2d(8, 16, 3)
        
        self.fc1 = nn.Linear((MAP_W - 4) * (MAP_H - 4) * 16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, 1) 
        
        x = F.relu(self.fc1(x))
        x = F.dropout(x, 0.2)
        
        x = F.relu(self.fc2(x))
        x = F.dropout(x, 0.2)
        
        x = self.fc3(x)
        
        return x

class NNPlayer(Player):
        
    def __init__(self, side, policy:PolicyNetwork, critic:PolicyNetwork):
        self.side = side
        self.policy = policy
        self.critic = critic
        
    def getAction(self, game: "RTSGame"):
        state_tensor = torch.from_numpy(game.get_state()).float().unsqueeze(0)

        logits = self.policy(state_tensor)
        self.m = torch.distributions.Categorical(logits=logits)

        action = self.m.sample()
        self.last_action = action

        return action

    
    def getProbabilities(self, action):
        return self.m.log_prob(action) 
    
class RandomPlayer(Player):

    def getAction(self, game: "RTSGame"):
        return np.random.random_integers(0, NUM_ACTIONS, (MAP_W, MAP_H))


class tile:

    def __init__(self, player_n, actor_type, hp, carry_gold):
        self.player_n = player_n      
        self.actor_type = actor_type  
        self.hp = hp                 
        self.carry_gold = carry_gold  


    def onehotEncode(self):

        features = []

        player_vec = [0] * MAX_PLAYERS
        if 0 <= self.player_n < MAX_PLAYERS:
            player_vec[self.player_n] = 1
        features.extend(player_vec)

        actor_vec = [0] * MAX_ACTORS
        if 0 <= self.actor_type < MAX_ACTORS:
            actor_vec[self.actor_type] = 1
        features.extend(actor_vec)

        features.append(self.hp / MAX_HP)
        features.append(self.carry_gold / CARRY_CAPACITY)

        return np.array(features, dtype=np.float32)

def bitpackTile(tile: tile):
    value = 0
    index = 0

    value |= (tile.player_n) << index
    index += MAX_PLAYERS.bit_length()

    value |= (tile.actor_type) << index
    index += MAX_ACTORS.bit_length()

    value |= (tile.hp) << index
    index += MAX_HP.bit_length()

    value |= (tile.carry_gold) << index

    return value

def bitunpackTile(value: int):
    player_n = value & (1 << MAX_PLAYERS.bit_length()) - 1
    value >>= MAX_PLAYERS.bit_length()

    actor_type = value & (1<<MAX_ACTORS.bit_length()) - 1
    value >>= MAX_ACTORS.bit_length()

    hp = value & (1<<MAX_HP.bit_length()) - 1
    value >>= MAX_HP.bit_length()

    carry_gold = value & (1<<CARRY_CAPACITY.bit_length())-1

    return tile(player_n, actor_type, hp, carry_gold)

def newBarrackTile(side):
    return tile(side, BARRACK_TYPE, BARRACK_HP, 2)

def newTCTile(side):
    return tile(side, TC_TYPE, TC_COST, 1)

class RTSGame():

    # Referencing this https://github.com/suragnair/alpha-zero-general/tree/master/rts
    # NOTES: one hot encode every tile
    # Archer, spear, horseman rock paper scissor style.
    # Resource gathering.
    
    # TODO
    # Set up game env
    #   Unit interaction
    #   Set up available moves (Will need optimization)
    #   Playable
    
    # TODO
    # Training
    #   Learn and set up basic policy gradient
    #   Set up PPO

    def setScreen(self, screen):
        self.screen = screen

    def __init__(self):
        empty_val = bitpackTile(tile(NO_PLAYER, EMPTY_TYPE, 0, 0))
        
        self.map = np.full((MAP_W, MAP_H), empty_val)

        self.map[2, 4] = bitpackTile(tile(0, TC_TYPE, TC_HP, 3))
        self.map[7, 4] = bitpackTile(tile(1, TC_TYPE, TC_HP, 3))

        self.map[0, 4] = bitpackTile(tile(NO_PLAYER, GOLD_TYPE, GOLD_HP, 0))
        self.map[9, 4] = bitpackTile(tile(NO_PLAYER, GOLD_TYPE, GOLD_HP, 0))


        if not pygame.font.get_init():
            pygame.font.init()
        self.unit_font = pygame.font.SysFont("monospace", 15, bold=True)
        self.stats_font = pygame.font.SysFont("Arial", 8, bold=False)

    def drawGrid(self):
        for x in range(MAP_W):
            for y in range(MAP_H):
                
                rect_x = x * BLOCKSIZE
                rect_y = y * BLOCKSIZE
                center_x = rect_x + (BLOCKSIZE // 2)
                center_y = rect_y + (BLOCKSIZE // 2)

                # Draw border
                rect = pygame.Rect(rect_x, rect_y, BLOCKSIZE, BLOCKSIZE)
                pygame.draw.rect(self.screen, (50, 50, 50), rect, 1)

                packed_val = self.map[x, y]
                tile_info = bitunpackTile(int(packed_val))

                # default to '?' if type not found in dictionary
                char_to_draw = ASCII_CHARS.get(tile_info.actor_type, '?') 

                text_color = PLAYER_COLORS.get(tile_info.player_n, WHITE)

                # render text
                unit_text_surf = self.unit_font.render(char_to_draw, True, text_color)
                unit_text_rect = unit_text_surf.get_rect(center=(center_x, center_y))
            
                self.screen.blit(unit_text_surf, unit_text_rect)

    def display (self):
        self.screen.fill(BLACK)
        while True:
            self.drawGrid()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            
            if pygame.mouse.get_pressed()[0]:

                if 0 <= pygame.mouse.get_pos()[0] <= WINDOW_W and 0 <= pygame.mouse.get_pos()[1] <= WINDOW_H:
                    grid_x = pygame.mouse.get_pos()[0] // BLOCKSIZE
                    grid_y = pygame.mouse.get_pos()[1] // BLOCKSIZE
                    tile_info = bitunpackTile(self.map[grid_x, grid_y])
                    
                    print(f"mouse click at grid {(grid_x, grid_y)}")
                    print(f"tile info: hp:{tile_info.hp}, gold:{tile_info.carry_gold}")


            pygame.display.update()
        
    def move_unit(map, cur_pos, target_pos):
        map[target_pos] = map[cur_pos]
        map[cur_pos] = bitpackTile(tile(NO_PLAYER, EMPTY_TYPE, 0, 0))
        return map

    # Return the features of every tile
    def get_state(self):
        return np.array([bitunpackTile(self.map[x, y]).onehotEncode() for x in range(MAP_W) for y in range(MAP_H)]).reshape(MAP_W, MAP_H, -1)

    def step(self, action, side):
        empty_val = bitpackTile(tile(NO_PLAYER, EMPTY_TYPE, 0, 0))

        for x in range(MAP_W):
            for y in range(MAP_H):
                tile_info = bitunpackTile(self.map[x][y])
                if tile_info.player_n == side:
                    tx, ty = x, y
                    
                    if action[x][y] == ACT_DOWN:
                        ty += 1
                    elif action[x][y] == ACT_UP:
                        ty -= 1
                    elif action[x][y] == ACT_LEFT:
                        tx -= 1
                    elif action[x][y] == ACT_RIGHT:
                        tx += 1
                    else:
                        continue
                    
                    if not (0 <= tx < MAP_W and 0 <= ty < MAP_H):
                        continue

                    target_tile_info = bitunpackTile(self.map[tx][ty])
                    # Now actually make a move.
                    # TC makes villager
                    if tile_info.actor_type == TC_TYPE:
                        if target_tile_info.player_n == NO_PLAYER and tile_info.carry_gold >= VILLAGER_COST:
                            self.map[tx, ty] = bitpackTile(tile(side, VILLAGER_TYPE, VILLAGER_HP, 0))
                            tile_info.carry_gold -= VILLAGER_COST
                            self.map[x, y] = bitpackTile(tile_info)

                    if tile_info.actor_type == BARRACK_TYPE:
                        if target_tile_info.player_n == NO_PLAYER and tile_info.carry_gold >= TROOP_COST:
                            self.map[tx, ty] = bitpackTile(tile(side, TROOP_TYPE, TROOP_HP, 0))
                            tile_info.carry_gold -= TROOP_COST
                            self.map[x, y] = bitpackTile(tile_info)

                    # Villager collect, return gold, make tc and barrack
                    elif tile_info.actor_type == VILLAGER_TYPE:
                        if action[x][y] == TURN_BARRACK and tile_info.carry_gold >= BARRACK_COST:
                            map[x][y] = bitpackTile(newBarrackTile())
                        elif action[x][y] == TURN_TC and tile_info.carry_gold >= TC_COST:
                            map[x][y] = bitpackTile(newTCTile())
                        elif target_tile_info.actor_type == GOLD_TYPE:
                            if tile_info.carry_gold <= 10:
                                tile_info.carry_gold += 1
                            self.map[x, y] = bitpackTile(tile_info)
                        elif target_tile_info.player_n == NO_PLAYER:
                            self.map = self.move_unit(self.map, (x, y), (tx, ty))
                        elif target_tile_info.player_n == side and target_tile_info.actor_type == TC_TYPE or target_tile_info.actor_type == BARRACK_TYPE:
                            target_tile_info.carry_gold += tile_info.carry_gold
                            tile_info.carry_gold = 0
                            self.map[x, y] = bitpackTile(tile_info)
                            self.map[tx, ty] = bitpackTile(target_tile_info)
                        
                    elif tile_info.actor_type == TROOP_TYPE:
                        if target_tile_info.player_n == NO_PLAYER:
                            self.map = self.move_unit(self.map, (x, y), (tx, ty))
                        elif target_tile_info.player_n != side:
                            # Opponent unit do dmg
                            target_tile_info.hp -= 1
                            self.map[tx,ty] = bitpackTile(target_tile_info)

                    
        reward = self.get_score()
        win = -1
        
        return action, self.get_state(), win, reward
        # Game end check, return action, state, win, reward

    def get_score(self, side):
        score = -12
        for x in range(MAP_W):
            for y in range(MAP_H):
                tile_info = bitunpackTile(self.map[x][y])
                if tile_info.player_n == side:
                    if tile_info.actor_type == TC_TYPE:
                        score += tile_info.hp
                        score += tile_info.carry_gold
                    elif tile_info.actor_type == VILLAGER_TYPE:
                        score += 1
                        
def train(trainee: NNPlayer, opponent: Player, episodes):
    # ACTOR CRITIC?
    # Initialize optimizer
    
    policy_optimizer = optim.Adam(trainee.policy.parameters())
    critic_optimizer = optim.Adam(trainee.critic.parameters())
    
    pygame.init()
    
    side = 0
    # Episodes
    for episode in range(episodes):
        print(episode)
        step = 0
        done = False
        game = RTSGame()
        
        rewards = []
        log_probs = []
        state_values = []
        
        policy_optimizer.zero_grad()
        critic_optimizer.zero_grad()
        
        while not done:
            if side == 0:
                state = game.get_state()
                state_values.append(trainee.critic(state))
                action = trainee.getAction(game)
                
                log_prob = trainee.getProbabilities(action)
                log_probs.append(log_prob)
                
                action, state, win, reward = game.step(action, side)
                
                rewards.append(reward)
                
            else:
                action, state, win, reward = game.step(opponent.getAction(game), side)
            
            side = (side + 1) % 2
        
        # Reward
        rewards = torch.tensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std())
        # Lossfunctions
        state_values = torch.tensor(state_values)
        advantage = torch.sub(rewards, state_values)
        log_probs = torch.tensor(log_probs)
        policy_loss = -log_probs * advantage
            
        critic_loss = F.huber_loss(state_values, rewards)
        policy_loss.backward()
        critic_loss.backward()
        # Optimze
        policy_optimizer.step()
        critic_optimizer.step()
    
    return None


new = RTSGame()
pygame.init()
SCREEN = pygame.display.set_mode((WINDOW_W, WINDOW_H))
new.setScreen(SCREEN)

new.display()
