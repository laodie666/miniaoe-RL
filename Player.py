import Game
import torch
import NN
import numpy as np

from Constant import *

class Player():
    
    def __init__(self, side):
        self.side = side
    
    def getAction(self):
        pass

    
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
        
    def __init__(self, side, policy:NN.PolicyNetwork, critic:NN.CriticNetwork):
        self.side = side
        self.policy = policy
        self.critic = critic
        
    def getAction(self, game: Game.RTSGame):
        cannonical_state = get_cannonical_state(game.get_state_tensor(), self.side)

        logits = self.policy(cannonical_state)
        self.m = torch.distributions.Categorical(logits=logits)

        action = self.m.sample()

        return action

    
    def getProbabilities(self, action):
        return self.m.log_prob(action) 
    
class RandomPlayer(Player):

    def getAction(self, game: Game.RTSGame):
        return np.random.randint(0, NUM_ACTIONS, size = (MAP_W, MAP_H))

