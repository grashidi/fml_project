import os
import pickle
import random

import numpy as np
import torch

#Import learning algorithm including hyperparameters
from .train import QLearner

MODEL_FILE_NAME = "DQNN.pt"

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

    #check if cuda is available and set device accordingly
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if self.train or not os.path.isfile(MODEL_FILE_NAME):
        self.logger.info("Setting up model from scratch.")
        self.qlearner   = QLearner(self.logger)
    else:
        self.logger.info("Loading model from saved state.")
        self.qlearner   = QLearner(self.logger)

        #Load neural network
        NN = torch.load(MODEL_FILE_NAME, map_location=self.device)
        self.qlearner.TNN.load_state_dict(NN.state_dict())
        self.qlearner.TNN.eval()
        self.qlearner.PNN.load_state_dict(NN.state_dict())
        self.qlearner.PNN.eval()
        self.logger.info("Loaded parameters of NN.")

        #Load transitions from memory
        #with open("transitions.pt", "rb") as file:
        #    self.qlearner.transitions = pickle.load(file)
        
        self.qlearner.is_training = False
        self.qlearner.is_fit = True
        

def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    
    return self.qlearner.propose_action(game_state)