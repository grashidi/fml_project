import pickle
import random
import numpy as np
from collections import namedtuple, deque
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

from torch.utils.tensorboard import SummaryWriter

from callbacks import state_to_features

from modified_rule_based_agent import Modified_Rule_Based_Agent


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
MODEL_FILE_NAME = "7x7_layer3_batch1_lr05_sgd_10000games"


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # set learning parameters
    self.criterion = nn.CrossEntropyLoss()
    #self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=0.0005)
    self.optimizer = optim.SGD(self.model.parameters(), lr=0.05)
    #self.optimizer = optim.Adam(self.model.parameters(), lr=0.1)
    self.batch_size = 1
    
    # init counter
    self.global_step = 0
    self.correct_counter = 0

    # writer for tensorboard
    self.writer = SummaryWriter("../../runs/"+MODEL_FILE_NAME)

    self.states = [] # array to save the game states that occured
    self.targets = [] # array to save what the rule based agent would do
    self.expert = Modified_Rule_Based_Agent()
    self.logger.debug("Everything is set up for this training game.")


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    # get opinion of expert
    target = self.expert.act(new_game_state)
    if target is not None:
        # append to states and target
        self.targets.append(ACTIONS.index(target)) # CrossEntropyLoss just needs the index of target class
        self.states.append(state_to_features(new_game_state))
        self.global_step += 1

        if self.global_step % self.batch_size == 0:
            # set model to trianing mode
            self.model.train()
            self.logger.info("Model set to training mode.")

            # translate states and targets to tensor and send to device, calculate output of network
            states = torch.tensor(self.states, dtype=torch.float).to(self.device)
            targets = torch.tensor(self.targets).type(torch.LongTensor).to(self.device)

            self.logger.debug("States and targets translated to tensors.")

            out = self.model(states).to(self.device)

            self.logger.debug("Output calculated.")

            # actual training with loss calculation, back propagation and optimization step
            loss = self.criterion(out, targets)

            self.model.zero_grad()
            loss.backward()
            self.optimizer.step()

            # calculate some other metrics for printing and tensorboard
            our_pred = np.array(ACTIONS)[torch.argmax(out.cpu(), dim=1)]
            target_pred = np.array(ACTIONS)[self.targets]

            correct = np.sum((our_pred == target_pred)*1)
            
            self.correct_counter += correct
            
            # print and write to tensoboard
            print(f"[{new_game_state['round']:5}] [{new_game_state['step']:3}] [{self.global_step:7}]: loss={loss/self.batch_size:.4f}, acc_batch={correct*100/self.batch_size:5}%, acc_glo={round(self.correct_counter*100./self.global_step, 2):6}%, our={our_pred}")
            #print(f"{'':15}our={our_pred}")
            print(f"{'':73}exp={target_pred}")
            self.writer.add_scalar("training loss per step", loss/self.batch_size, self.global_step)
            self.writer.add_scalar("training global accruracy", self.correct_counter/self.global_step, self.global_step)
            self.writer.add_scalar("training batch accruracy", correct/self.batch_size, self.global_step)

            #for param in self.model.parameters():
                #print(torch.max(param.data))

            # set everything back for next batch
            # not sure if necessary, becuase I'm not sure when the setup method is called
            # once at the beginning or at the beginning of every game
            self.states = []
            self.targets = []
            self.model.eval()
            self.logger.info("Everything set back for new game.")

    else:
        print("Target is None!!!", self.global_step)
        print(old_game_state['round'], ", ", old_game_state['step'])
        print(new_game_state['round'], ", ", new_game_state['step'])


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.

    This is similar to reward_update. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    for param in self.model.parameters():
      print(torch.max(param.data))
    print(f"end of round, {last_game_state['round']}, {last_game_state['step']}")
    print("************************************************************************")
    print()
    
    self.states = []
    self.targets = []
    self.model.eval()
    self.global_step += 1
    self.logger.info("Everything set back for new game.")

    # flush summary writer
    self.writer.flush()
    if last_game_state['round'] == 10000:
      self.writer.close()

    # save the model
    torch.save(self.model, MODEL_FILE_NAME+".pt")
    self.logger.info("Model saved to " + MODEL_FILE_NAME+".pt")