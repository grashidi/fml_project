import os
import pickle
import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import sys
sys.path.append('agent_code/our_agent')
sys.path.append('bomberman_rl/agent_code/our_agent')
from modified_rule_based_agent import Modified_Rule_Based_Agent


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
#MODEL_FILE_NAME = "our-saved-model.pt"
MODEL_FILE_NAME = "layer3_batch4_lr001_wd0005_sgd.pt"
SIZE_OF_INPUT = 257
RANDOM_PROB = 0.0


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # check if cuda is available and set device accordingly
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # check if sved version is available, otherwise initialize new model
    if False:#os.path. isfile(MODEL_FILE_NAME):
        self.model = torch.load(MODEL_FILE_NAME, map_location=self.device)
        self.logger.info("Loaded saved model.")
        print("Loaded saved Model.")
    else:
        self.model = OurNeuralNetwork(SIZE_OF_INPUT)
        self.logger.info("Setting up model from Scratch.")
        print("Setting up model from Scratch.")
    
    # set model to device
    self.model = self.model.to(self.device)
    print("Model runs on " + str(self.device))
    print("Number of parameters: ", sum(p.numel() for p in self.model.parameters() if p.requires_grad))
    self.logger.info("Model runs on " + str(self.device))
    self.logger.info("Model has " +str(sum(p.numel() for p in self.model.parameters() if p.requires_grad)) + " parameters")

    # make sure model is in eval mode
    self.model.eval()
    self.logger.info("Model runs in eval mode.")


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # exploration only when training and with prob = RANDOM_PROB
    if self.train and random.random() < RANDOM_PROB:
        self.logger.debug("Choosing action as random for exploration.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[0.2, 0.2, 0.2, 0.2, 0.1, 0.1])

    state_vector = torch.tensor(state_to_features(game_state), dtype=torch.float).to(self.device)
    out = self.model(state_vector)
   
    self.logger.debug("Querring model for best action.")
    return ACTIONS[torch.argmax(out)]


def state_to_features(game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None
    
    
    # note: in the game we have a field of 17x17, but the borders are always
    # stone so we reduce the dimension to 15x15
    hybrid_vectors = np.zeros((7, 7, 5), dtype=int)
    
    # check where there are stones on the field
    # just use the field without the borders (1:-1)
    # set the first entry in the vector to 1
    hybrid_vectors[ np.where(game_state['field'][1:-1, 1:-1] == -1), 0 ] = 1

    # check where there are crates
    # set the second entry in the vector to 1
    hybrid_vectors[ np.where(game_state['field'][1:-1, 1:-1] == 1), 1 ] = 1

    
    if len(game_state['coins']) > 0:
        coin_coords = np.moveaxis(np.array(game_state['coins']), -1, 0)
        hybrid_vectors[ coin_coords[0]-1, coin_coords[1]-1, 2 ] = 1

    # check where bombs are
    # set the fourth entry in the vector to 1
    # discard the time since this can be learned by the model because we
    # use a LSTM network
    if len(game_state['bombs']) > 0:
        bomb_coords = np.array([[bomb[0][0], bomb[0][1], bomb[1]] for bomb in game_state['bombs']]).T
        hybrid_vectors[ bomb_coords[0]-1, bomb_coords[1]-1, 3 ] = bomb_coords[2]

    # vectorized version of above implementation
    '''
    bombs = game_state['bombs']
    n_bombs = len(bombs)
    bombs = np.asarray(bombs).T
    bombs_xy = np.concatenate(bombs[0])
    bombs_xy = bombs_xy.reshape(n_bombs, 2).T
    bombs_t = np.concatenate(bombs[1])
    hybrid_vectors[bombs_xy[0]-1, bombs_xy[1]-1, 3 ] = bombs_t
    '''

    # check where fire is
    # set the fifth entry in the vector to 1
    hybrid_vectors[ :, :, 4 ] = game_state['explosion_map'][1:-1, 1:-1]

    # flatten 3D array to 1D vector
    hyb_vec = hybrid_vectors.flatten()

    # add enemy coords and their bomb boolean as additional entries at the end
    # non-existing enemies have -1 at each position as default
    for i in range(3):
        if len(game_state['others']) > i:
            enemy = game_state['others'][i]
            hyb_vec = np.append(hyb_vec, [ enemy[3][0], enemy[3][1], int(enemy[2]) ])
        else:
            hyb_vec = np.append(hyb_vec, [ -1 , -1 , -1 ])

    # add own position and availability of bomb as 3 additional entries at the end
    hyb_vec = np.append(hyb_vec, [ game_state['self'][3][0], game_state['self'][3][1], int(game_state['self'][2]) ])

    return hyb_vec # len(hyb_vec) = (15 x 15 x 5) + (4 x 3) = 1137


# wieviele layer? wie groß? sprünge in layer größe okay oder sogar gut?
# wie baut man lstm layer ein? reicht eins?
# tensorboard einbauen
class OurNeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(OurNeuralNetwork, self).__init__()
        self.linear1 = nn.Linear(input_size, 64) # input_size 257
        self.linear2 = nn.Linear(64, 16)
        self.linear3 = nn.Linear(16, 6)

    def forward(self, x):
        out = self.linear1(x)
        out = F.selu(out)
        out = self.linear2(out)
        out = F.selu(out)
        out = self.linear3(out)
        return out


    '''
    def __init__(self, input_size):
        super(OurNeuralNetwork, self).__init__()
        self.linear1 = nn.Linear(input_size, 256) # input_size 1137
        self.linear2 = nn.Linear(256, 64)
        self.linear3 = nn.Linear(64, 6)
        #self.linear2 = nn.Linear(512, 128)
        #self.linear3 = nn.Linear(128, 32)
        #self.linear4 = nn.Linear(32, 6)

    def forward(self, x):
        out = self.linear1(x)
        out = F.selu(out)
        out = self.linear2(out)
        out = F.selu(out)
        out = self.linear3(out)
        #out = F.selu(out)
        #out = self.linear4(out)
        return out

    
    def init_old(self, input_size):
        super(OurNeuralNetwork, self).__init__()
        self.linear1 = nn.Linear(input_size, 2048)
        self.linear2 = nn.Linear(2048, 512)
        self.linear3 = nn.Linear(512, 128)
        self.linear4 = nn.Linear(128, 32)
        self.linear5 = nn.Linear(32, 6)

    def forward_old(self, x):
        # könnte auch andere activation function nehmen
        out = self.linear1(x)
        out = F.selu(out)
        out = self.linear2(out)
        out = F.selu(out)
        out = self.linear3(out)
        out = F.selu(out)
        out = self.linear4(out)
        out = F.selu(out)
        out = self.linear5(out)
        # out = F.softmax nicht nötig, weil CrossEntropy das auch anwendet
        return out
    '''
