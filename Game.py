import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pygame
import random
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
TC_HP = 4
VILLAGER_HP = 1
TROOP_HP = 2
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



class Player():
    
    def __init__(self, side):
        self.side = side
    
    def getAction(self):
        pass


# Need some adjustment to the game features.
class PolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(CHANNEL_NUM, 8, 3)
        self.conv2 = nn.Conv2d(8, 16, 3)
        
        # Note this conversion from CNN to Linear is so hand wavy, there is probably some way to make this easier that I should look into.
        # there is a + 1 in the end to make the agent aware of what side it is on.
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
    
# Since I'm only training the network as side = 0, when its playing as side 1 the unit side should be swapped. 
def get_cannonical_state(state_tensor, side):
    if side == 0:
        return state_tensor
    canonical_state = state_tensor.clone()
    # (Batch, Channels, W, H), swap 0 and 1 player number of one hot encoding.
    canonical_state[:,0,:,:] = state_tensor[:,1,:,:]
    canonical_state[:,1,:,:] = state_tensor[:,0,:,:]
    return canonical_state


class NNPlayer(Player):
        
    def __init__(self, side, policy:PolicyNetwork, critic:PolicyNetwork):
        self.side = side
        self.policy = policy
        self.critic = critic
        
    def getAction(self, game: "RTSGame"):
        cannonical_state = get_cannonical_state(game.get_state_tensor(), self.side)

        logits = self.policy(cannonical_state)
        self.m = torch.distributions.Categorical(logits=logits)

        action = self.m.sample()
        self.last_action = action

        return action

    
    def getProbabilities(self, action):
        return self.m.log_prob(action) 
    
class RandomPlayer(Player):

    def getAction(self, game: "RTSGame"):
        return np.random.randint(0, NUM_ACTIONS, size = (MAP_W, MAP_H))


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
    return tile(side, BARRACK_TYPE, BARRACK_HP, 0)

def newTCTile(side):
    return tile(side, TC_TYPE, TC_HP, 0)

class RTSGame():

    # Referencing this https://github.com/suragnair/alpha-zero-general/tree/master/rts
    # NOTES: one hot encode every tile
    # Archer, spear, horseman rock paper scissor style.
    # Resource gathering.


    def setScreen(self, screen):
        self.screen = screen

    # TODO: RANDOMIZE RESOURCE AND START POSITION

    def __init__(self):
        empty_val = bitpackTile(tile(NO_PLAYER, EMPTY_TYPE, 0, 0))
        
        self.map = np.full((MAP_W, MAP_H), empty_val)
        self.left_side = random.randint(0,1)
        self.right_side = (1+self.left_side)%2
        self.map[LEFT_MAIN_TC_POS] = bitpackTile(tile(self.left_side, TC_TYPE, TC_HP, 3))
        self.map[RIGHT_MAIN_TC_POS] = bitpackTile(tile(self.right_side, TC_TYPE, TC_HP, 3))

        self.map[0, 4] = bitpackTile(tile(NO_PLAYER, GOLD_TYPE, GOLD_HP, 0))
        self.map[9, 4] = bitpackTile(tile(NO_PLAYER, GOLD_TYPE, GOLD_HP, 0))
        
        self.map[0, 1] = bitpackTile(tile(NO_PLAYER, GOLD_TYPE, GOLD_HP, 0))
        self.map[9, 1] = bitpackTile(tile(NO_PLAYER, GOLD_TYPE, GOLD_HP, 0))
        
        self.map[0, 7] = bitpackTile(tile(NO_PLAYER, GOLD_TYPE, GOLD_HP, 0))
        self.map[9, 7] = bitpackTile(tile(NO_PLAYER, GOLD_TYPE, GOLD_HP, 0))

        self.update_onehot_encoding()

        if not pygame.font.get_init():
            pygame.font.init()
        self.unit_font = pygame.font.SysFont("monospace", 15, bold=True)
        self.stats_font = pygame.font.SysFont("Arial", 8, bold=False)
                
    def update_onehot_encoding(self):
        self.onehot_encoded_tiles = np.array([(bitunpackTile(self.map[x][y])).onehotEncode() for x in range(MAP_W) for y in range(MAP_H)]).reshape(MAP_W, MAP_H, -1)
    
    
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

                tile_info = bitunpackTile(self.map[x][y])

                # default to '?' if type not found in dictionary
                char_to_draw = ASCII_CHARS.get(tile_info.actor_type, '?') 

                text_color = PLAYER_COLORS.get(tile_info.player_n, WHITE)

                # render text
                unit_text_surf = self.unit_font.render(char_to_draw, True, text_color)
                unit_text_rect = unit_text_surf.get_rect(center=(center_x, center_y))
            
                self.screen.blit(unit_text_surf, unit_text_rect)

    def display (self):
        self.screen.fill(BLACK)
        self.drawGrid()
        pygame.display.update()
        
    def move_unit(self, cur_pos, target_pos):
        self.map[target_pos] = self.map[cur_pos]
        self.map[cur_pos] = bitpackTile(tile(NO_PLAYER, EMPTY_TYPE, 0, 0))


    # Return the features of every tile
    def get_state(self):
        return self.onehot_encoded_tiles

    def get_state_tensor(self):
        state_tensor = torch.from_numpy(self.get_state()).float().unsqueeze(0)

        # Move number channels to the front.
        state_tensor = state_tensor.permute(0, 3, 1, 2) 
        return state_tensor.to(device)

    # TODO: Vectorize game loop to make it more efficient.

    def step(self, action, side):
        processed = np.zeros((MAP_W, MAP_H), dtype=bool) 

        for x in range(MAP_W):
            for y in range(MAP_H):
                # Say a unit moves right, this is to cover the case for the unit to move again.
                if processed[x][y]: continue 

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
                    
                    
                    if not (0 <= tx < MAP_W and 0 <= ty < MAP_H):
                        continue

                    target_tile_info = bitunpackTile(self.map[tx][ty])
                    # Now actually make a move.
                    # TC makes villager
                    if tile_info.actor_type == TC_TYPE:
                        if target_tile_info.actor_type == EMPTY_TYPE and tile_info.carry_gold >= VILLAGER_COST:
                            self.map[tx, ty] = bitpackTile(tile(side, VILLAGER_TYPE, VILLAGER_HP, 0))
                            tile_info.carry_gold -= VILLAGER_COST
                            self.map[x, y] = bitpackTile(tile_info)
                            processed[x][y] = 1
                            processed[tx][ty] = 1

                    # Barrack makes troop
                    elif tile_info.actor_type == BARRACK_TYPE:
                        if target_tile_info.actor_type == EMPTY_TYPE and tile_info.carry_gold >= TROOP_COST:
                            self.map[tx, ty] = bitpackTile(tile(side, TROOP_TYPE, TROOP_HP, 0))
                            tile_info.carry_gold -= TROOP_COST
                            self.map[x, y] = bitpackTile(tile_info)
                            processed[x][y] = 1
                            processed[tx][ty] = 1


                    # Villager collect, return gold, make tc and barrack
                    elif tile_info.actor_type == VILLAGER_TYPE:
                        if action[x][y] == TURN_BARRACK and tile_info.carry_gold >= BARRACK_COST:
                            self.map[x][y] = bitpackTile(newBarrackTile(side))
                        elif action[x][y] == TURN_TC and tile_info.carry_gold >= TC_COST:
                            self.map[x][y] = bitpackTile(newTCTile(side))
                        elif target_tile_info.actor_type == GOLD_TYPE:
                            if tile_info.carry_gold <= 5:
                                tile_info.carry_gold += 1
                            self.map[x, y] = bitpackTile(tile_info)
                        elif target_tile_info.actor_type == EMPTY_TYPE:
                            self.move_unit((x, y), (tx, ty))
                            processed[x][y] = 1
                            processed[tx][ty] = 1
                        elif target_tile_info.player_n == side and (target_tile_info.actor_type == TC_TYPE or target_tile_info.actor_type == BARRACK_TYPE):
                            # Let TC and Barrack only carry 6 gold
                            transfered_gold = min(6 - target_tile_info.carry_gold, tile_info.carry_gold)
                            target_tile_info.carry_gold += transfered_gold
                            tile_info.carry_gold -= transfered_gold
                            self.map[x, y] = bitpackTile(tile_info)
                            self.map[tx, ty] = bitpackTile(target_tile_info)
                        
                    elif tile_info.actor_type == TROOP_TYPE:
                        if target_tile_info.actor_type == EMPTY_TYPE:
                            self.move_unit((x, y), (tx, ty))
                            processed[x][y] = 1
                            processed[tx][ty] = 1
                        elif target_tile_info.player_n != side and target_tile_info.player_n != NO_PLAYER:
                            # Opponent unit do dmg
                            target_tile_info.hp -= 1
                            if target_tile_info.hp <= 0:
                                target_tile_info = tile(NO_PLAYER, EMPTY_TYPE, 0, 0)
                            self.map[tx,ty] = bitpackTile(target_tile_info)

        self.update_onehot_encoding()
        win = -1
        if self.get_score((side + 1)%2) is None:
            print(f"GG player {(side + 1)%2} DIED")
            win = side
            reward = self.get_score(side) + 100
        elif self.get_score(side) is None:
            print(f"GG player {side} DIED")
            win = (side + 1)%2
            reward = -100
        else:
            reward = self.get_score(side) - self.get_score((side + 1)%2)
        
        
        return action, self.get_state(), win, reward

    def get_score(self, side):
        score = -12
        villager_count = 0
        tc_count = 0

        left_main_tc_tile = bitunpackTile(self.map[LEFT_MAIN_TC_POS])
        right_main_tc_tile = bitunpackTile(self.map[RIGHT_MAIN_TC_POS])

        if left_main_tc_tile.player_n != side and right_main_tc_tile.player_n != side:
            # TODO: This is SUPER SCUFFED. Since any number can be reached with the current score, to distinguish game ending return None
            return None

        for x in range(MAP_W):
            for y in range(MAP_H):
                tile_info = bitunpackTile(self.map[x][y])
                if tile_info.player_n == side:
                    if tile_info.actor_type == TC_TYPE:
                        score += tile_info.hp
                        score += tile_info.carry_gold * 1.5
                        tc_count += 1
                    elif tile_info.actor_type == VILLAGER_TYPE:
                        score += tile_info.hp
                        score += tile_info.carry_gold
                    else:
                        score += tile_info.hp
                        score += tile_info.carry_gold
        return score
                        
def train(trainee: NNPlayer, opponent: Player, episodes, gamma, entropy_coef):
    # ACTOR CRITIC?
    # Initialize optimizer
    
    policy_optimizer = optim.Adam(trainee.policy.parameters())
    critic_optimizer = optim.Adam(trainee.critic.parameters())
    
    side = 0
    # Episodes
    for episode in range(episodes):
        step = 0
        done = False
        game = RTSGame()
        
        rewards = []
        log_probs = []
        state_values = []
        # Mask out empty spaces
        masks = []
        
        policy_optimizer.zero_grad()
        critic_optimizer.zero_grad()
        
        last_reward = 0


        while not done and step <= 150:

            if side == 0: 
                state_tensor = game.get_state_tensor()
                
                player_map = state_tensor[0, side, :, :]

                # Create a mask for where the player has units, then only consider those tiles when calculating loss to avoid being distracted by empty tiles.
                mask = player_map > 0
                masks.append(mask)

                state_values.append(trainee.critic(state_tensor))
                action = trainee.getAction(game)
                
                log_prob = trainee.getProbabilities(action)
                log_probs.append(log_prob)
                
                action, state_tensor, win, reward = game.step(action, side)
                
                rewards.append(reward - last_reward)
                last_reward = reward
                
            else:
                action, state_tensor, win, reward = game.step(opponent.getAction(game), side)

            if win != -1:
                done = True

            side = (side + 1) % 2
            step += 1 
        
        # Reward, change of game score
        rewards = np.array(rewards)
        returns = []
        R = 0
        for r in rewards[::-1]:
            R = r + gamma * R
            returns.insert(0,R)
        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        # Make sure dont divide by 0
        returns = (returns - returns.mean()) / (returns.std() + 1e-7)

        # Loss functions
        log_probs = torch.stack(log_probs) 
        state_values = torch.cat(state_values).squeeze()
        masks = torch.stack(masks)

        advantage = returns - state_values.detach()
        
        entropy = trainee.m.entropy() * masks.float()  
        entropy = entropy.sum() / (masks.sum() + 1e-7)
        
        # Advantage above is [step] shaped while log_prob is probility for [step, MAP_W, MAP_H], need to expand out 
        advantage = advantage.view(-1, 1, 1).expand_as(log_probs)
        policy_loss = -(log_probs * advantage) * masks.float()
        policy_loss = policy_loss.sum() / (masks.sum() + 1e-7)

        policy_loss = policy_loss - entropy * entropy_coef

        critic_loss = F.huber_loss(state_values, returns)
        policy_loss.backward()
        critic_loss.backward()
        # Optimze
        policy_optimizer.step()
        critic_optimizer.step()

        print(f"Ep {episode}: Policy_loss {policy_loss}: Critic_loss {critic_loss}")
        
    
    return None

def pit(p1: Player, p2: Player, num_games):
    pygame.init()
    clock = pygame.time.Clock()
    
    side = 0
    win_rate = [0,0]
    # Episodes
    for game_num in range(num_games):
        step = 0
        done = False
        game = RTSGame()
        printed_side = 0
        slow = False
        skip = False
        while not done and step <= 150:
            
            if  game_num == 0:
                if not printed_side:
                    print("sides ", game.left_side, game.right_side)
                    printed_side = True
                screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
                game.setScreen(screen)
                game.display()
                if skip:
                    clock.tick(0)
                elif not slow:
                    clock.tick(15)
                else:
                    clock.tick(2)
                
                if pygame.mouse.get_pressed()[0]:
                    if 0 <= pygame.mouse.get_pos()[0] <= WINDOW_W and 0 <= pygame.mouse.get_pos()[1] <= WINDOW_H:
                        grid_x = pygame.mouse.get_pos()[0] // BLOCKSIZE
                        grid_y = pygame.mouse.get_pos()[1] // BLOCKSIZE
                        tile_info = bitunpackTile(game.map[grid_x][grid_y])
                        
                        print(f"mouse click at grid {(grid_x, grid_y)}")
                        print(f"tile info: hp:{tile_info.hp}, gold:{tile_info.carry_gold}")
                events = pygame.event.get()
                for event in events:
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_SPACE and not slow:
                            slow = True
                        elif event.key == pygame.K_SPACE and slow:
                            slow = False
                        elif event.key == pygame.K_s:
                            skip = True

            if side == 0: 
                action, state_tensor, win, reward = game.step(p1.getAction(game), side)
            else:
                action, state_tensor, win, reward = game.step(p2.getAction(game), side)

            
            side = (side + 1) % 2
            step += 1 

            if win != -1:
                done = True
        if done:
            win_rate[win] += 1
        else:
            if game.get_score(0) > game.get_score(1):
                win_rate[0] += 1
            elif game.get_score(0) < game.get_score(1):
                win_rate[1] += 1
    return win_rate

def copy_player(policy_nn, critic_nn, policy_player, side):
    policy_nn_copy = PolicyNetwork().to(device)
    policy_nn_copy.load_state_dict(policy_nn.state_dict())

    critic_nn_copy = CriticNetwork().to(device)
    critic_nn_copy.load_state_dict(critic_nn.state_dict())

    policy_player_copy = NNPlayer(side, policy_nn_copy, critic_nn_copy)

    return policy_nn_copy, critic_nn_copy, policy_player_copy

print(torch.cuda.is_available())

policy_nn = PolicyNetwork().to(device)

# print("loaded policy checkpoint")
# policy_state_dict = torch.load("policy_checkpoint.pt")
# policy_nn.load_state_dict(policy_state_dict)

critic_nn = CriticNetwork().to(device)

# print("loaded critic checkpoint")
# critic_state_dict = torch.load("critic_checkpoint.pt")
# critic_nn.load_state_dict(critic_state_dict)

policy_player = NNPlayer(0, policy_nn, critic_nn)

policy_nn_copy, critic_nn_copy, policy_player_copy = copy_player(policy_nn, critic_nn, policy_player, 1)

win_rate = pit(policy_player, policy_player_copy, 50)
print(win_rate)

epochs = 20
for epoch in range(epochs):
    torch.save(policy_nn.state_dict(), f"policy_checkpoint.pt")
    torch.save(critic_nn.state_dict(), f"critic_checkpoint.pt")
    print(f"epoch {epoch}")
    train(policy_player, policy_player_copy, 500, 0.95, 0.1)
    win_rate = pit(policy_player, policy_player_copy, 50)
    print(win_rate)
    if win_rate[0] - 5 >= win_rate[1]:
        print("Performed better than before, updating agent.")
        policy_nn_copy, critic_nn_copy, policy_player_copy = copy_player(policy_nn, critic_nn, policy_player, 1)
    else:
        print("Performed worse, Keep training.")
        # policy_nn, critic_nn, policy_player = copy_player(policy_nn_copy, critic_nn_copy, policy_player_copy)