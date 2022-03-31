from collections import namedtuple, deque
import pickle
from typing import List
import numpy as np
import events as e
import settings as s
from .callbacks import state_to_features, bomb_is_within_reach 
from .callbacks import get_crate_directions, get_explosions 
from .callbacks import get_free_directions 
from .callbacks import get_bombs_about_to_explode_directions
import matplotlib.pyplot as plt


# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 4  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

LEARNING_RATE = 0.1
DISCOUNT = 0.9
EVERY = 100

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.steps_done = 0
    self.rounds_done = 0
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)

    self.rewards_per_game = []
    self.rewards_per_step = []
    self.delta = []
    self.stats = {"avg": [], "max": [], "min": [], "q_table_len": [], "eps": [], "delta": []}

def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    old_state_features = state_to_features(self, old_game_state)
    new_state_features = state_to_features(self, new_game_state)

    if new_game_state != None and old_game_state != None:	
        # check if agent is closer to enemy
        old_pos, new_pos = old_game_state["self"][-1], new_game_state["self"][-1]
        within_reach_old = bomb_is_within_reach(old_game_state)
        within_reach_new = bomb_is_within_reach(new_game_state)
        old_dis = []
        new_dis = []

        for others in old_game_state["others"]:
            enemy_pos = others[-1]
            old_dis += [get_distance(old_pos, enemy_pos)]
        
        for others in new_game_state["others"]:
            enemy_pos = others[-1]
            new_dis += [get_distance(new_pos, enemy_pos)]

        old_dis, new_dis = np.array(old_dis), np.array(new_dis)
        closer_to_enemy = old_dis[:new_dis.shape[0]] > new_dis
        further_from_enemy = old_dis[:new_dis.shape[0]] < new_dis
        
        if np.any(closer_to_enemy):
            events += ["CLOSER_TO_ENEMY"]
        
        if np.any(further_from_enemy):
            events += ["FURTHER_FROM_ENEMY"]

        # check if bomb has been used when not available
        if not old_game_state["self"][2] and self_action == "BOMB":
            events += ["INVALID_BOMB"]

        # check if bomb is within each
        if within_reach_old:
            events += ["WITHIN_REACH_OF_BOMB"]
        
            # check if agent moves to safe area
            if old_state_features[1] != -1:
                all_moves = np.array(["DOWN", "UP", "RIGHT", "LEFT"])
                safe = np.array([int(i) for i in format(int(old_state_features[1]), "04b")])
                safe_moves = all_moves[safe == 1]
                
                if self_action in safe_moves:
                    events += ["MOVED_TO_SAFE_AREA"]        
                else:
                    events += ["MOVED_AWAY_FROM_SAVE_AREA"]
            
        # check if agent's move is blocked
        up = self_action == "UP"
        down = self_action == "DOWN"
        right = self_action == "RIGHT"
        left = self_action == "LEFT"

        if up or down or right or left:
            x_is_same = old_pos[0] == new_pos[0]
            y_is_same = old_pos[1] == new_pos[1]
            
            if x_is_same and y_is_same:
                events += ["BLOCKED_MOVE"]

        # check if agent moves closer to crate
        if not within_reach_old:
            if old_state_features[3] == 8 and self_action == "DOWN":
                events += ["CLOSER_TO_OBJECT_OF_INTEREST"]
            
            if old_state_features[3] == 4 and self_action == "UP":
                events += ["CLOSER_TO_OBJECT_OF_INTEREST"]
            
            if old_state_features[3] == 2 and self_action == "RIGHT":
                events += ["CLOSER_TO_OBJECT_OF_INTEREST"]
            
            if old_state_features[3] == 1 and self_action == "LEFT":
                events += ["CLOSER_TO_OBJECT_OF_INTEREST"]
      
        # check if agent does not move closer to crate 
        if not self_action in ["WAIT", "BOMB"] and not within_reach_old: 
            if old_state_features[3] == 8 and self_action != "DOWN":
                events += ["NOT_CLOSER_TO_OBJECT_OF_INTEREST"]
            
            if old_state_features[3] == 4 and self_action != "UP":
                events += ["NOT_CLOSER_TO_OBJECT_OF_INTEREST"]
            
            if old_state_features[3] == 2 and self_action != "RIGHT":
                events += ["NOT_CLOSER_TO_OBJECT_OF_INTEREST"]
            
            if old_state_features[3] == 1 and self_action != "LEFT":
                events += ["NOT_CLOSER_TO_OBJECT_OF_INTEREST"]

        # check for save bomb drop
        if (old_state_features[2] > 0 and old_state_features[3] == 0
            and self_action == "BOMB"):
            events += ["SAVE_CRATE_BOMB"]
        
        # missed dropping a bomb
        if (old_state_features[2] > 0 and old_state_features[3] == 0
            and self_action != "BOMB"):
            events += ["MISSED_BOMB"]

        # check if agent drops ineffective bomb
        if old_state_features[3] > 0 and old_state_features[5] == 0 and self_action == "BOMB":
            events += ["INEFFECTIVE_BOMB"]

        # check for move to into save direction after bomb drop
        if self.transitions[-1].action == "BOMB" and (old_state_features[2] == 8 and self_action == "DOWN"):
            events += ["SAFE_DIRECTION_MOVE_AFTER_BOMB_DROP"]
        
        if self.transitions[-1].action == "BOMB" and (old_state_features[2] == 4 and self_action == "UP"):
            events += ["SAFE_DIRECTION_MOVE_AFTER_BOMB_DROP"]
        
        if self.transitions[-1].action == "BOMB" and (old_state_features[2] == 2 and self_action == "RIGHT"):
            events += ["SAFE_DIRECTION_MOVE_AFTER_BOMB_DROP"]
        
        if self.transitions[-1].action == "BOMB" and (old_state_features[2] == 1 and self_action == "LEFT"):
            events += ["SAFE_DIRECTION_MOVE_AFTER_BOMB_DROP"]
           
        if self.transitions[-1].action == "BOMB" and (old_state_features[2] == 8 and self_action != "DOWN"):
            events += ["NOT_SAFE_DIRECTION_MOVE_AFTER_BOMB_DROP"]

        if self.transitions[-1].action == "BOMB" and (old_state_features[2] == 4 and self_action != "UP"):
            events += ["NOT_SAFE_DIRECTION_MOVE_AFTER_BOMB_DROP"]
        
        if self.transitions[-1].action == "BOMB" and (old_state_features[2] == 2 and self_action != "RIGHT"):
            events += ["NOT_SAFE_DIRECTION_MOVE_AFTER_BOMB_DROP"]
        
        if self.transitions[-1].action == "BOMB" and (old_state_features[2] == 1 and self_action != "LEFT"):
            events += ["NOT_SAFE_DIRECTION_MOVE_AFTER_BOMB_DROP"]

        # check if agents moves into explosion
        explosions = get_explosions(old_pos, old_game_state["explosion_map"])
        all_moves = np.array(["DOWN", "UP", "RIGHT", "LEFT"])
        deadly_moves = all_moves[explosions == 1]
        
        if np.any(explosions) and not within_reach_old and not self_action in deadly_moves:
            events += ["AVOIDED_EXPLOSION"]

        if self_action in deadly_moves:
            events += ["MOVED_INTO_EXPLOSION"]        
        
        # check if agent moves into bomb about to explode
        bombs_about_to_explode = get_bombs_about_to_explode_directions(old_game_state)
        all_moves = np.array(["DOWN", "UP", "RIGHT", "LEFT"])
        into_bomb_moves = all_moves[bombs_about_to_explode == 1]
        
        if np.any(bombs_about_to_explode) and not within_reach_old and not self_action in into_bomb_moves:
            events += ["AVOIDED_BOMB_ABOUT_TO_EXPLODE"]
            
        if self_action in into_bomb_moves:
            events += ["MOVED_INTO_BOMB_ABOUT_TO_EXPLODE"]

        # bomb enemy if nearby
        if old_state_features[2] > 0 and old_state_features[5] > 0 and self_action == "BOMB": 
            events += ["BOMBED_ENEMY"]

        reward = reward_from_events(self, events)
            

        max_future_q = np.max(self.model[str(new_state_features)])
        curr_q = self.model[str(old_state_features)][ACTIONS.index(self_action)]

        new_q = (1 - LEARNING_RATE) * curr_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
        
        self.model[str(old_state_features)][ACTIONS.index(self_action)] = new_q

        self.delta += [np.abs(curr_q - new_q)]

    r = reward_from_events(self, events)
    self.rewards_per_step += [r]

    self.transitions.append(Transition(old_state_features, self_action, new_state_features, r))
    

def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')

    self.rounds_done += 1
    
    if not (e.GOT_KILLED in events or e.KILLED_SELF in events):
        events.append(e.SURVIVED_ROUND)

    last_state = state_to_features(self, last_game_state)
    r = reward_from_events(self, events)

    self.model[str(last_state)][ACTIONS.index(last_action)] = r
    
    self.rewards_per_step += [r]
    self.rewards_per_game += [sum(self.rewards_per_step)]
    self.rewards_per_step = []

    if self.rounds_done % EVERY == 0:
        self.stats["avg"] += [np.mean(self.rewards_per_game)]
        self.stats["max"] += [max(self.rewards_per_game)]
        self.stats["min"] += [min(self.rewards_per_game)]

        self.stats["q_table_len"] += [len(self.model)]

        self.stats["eps"] += [self.eps_threshold] 

        self.stats["delta"] += [np.mean(self.delta)]

        self.rewards_per_game = []
        self.delta = []

    self.transitions.append(Transition(last_state, last_action, None, r))

    if self.rounds_done % 1000 == 0:
        fig, ax = plt.subplots()
        ax.plot(np.arange(self.rounds_done // EVERY) * EVERY, self.stats["avg"], label="average")
        ax.plot(np.arange(self.rounds_done // EVERY) * EVERY, self.stats["min"], label="min")
        ax.plot(np.arange(self.rounds_done // EVERY) * EVERY, self.stats["max"], label="max")
        ax.yaxis.grid(True, which="major", linestyle='--')
        ax.set_title("Accumulated rewards per game")
        ax.set_xlabel("number of rounds")
        ax.set_ylabel("accumulated value")
        plt.legend()
        fig.savefig("rewards.png")

        fig, ax = plt.subplots()
        ax.plot(np.arange(self.rounds_done // EVERY) * EVERY, self.stats["q_table_len"], label="q table length")
        ax.yaxis.grid(True, which="major", linestyle='--')
        ax.set_title("Q table length")
        ax.set_xlabel("number of rounds")
        ax.set_ylabel("length")
        plt.legend()
        fig.savefig("q_table_len.png")
        
        fig, ax = plt.subplots()
        ax.plot(np.arange(self.rounds_done // EVERY) * EVERY, self.stats["eps"], label="eps")
        ax.yaxis.grid(True, which="major", linestyle='--')
        ax.set_title("Epsilon")
        ax.set_xlabel("Number of rounds")
        ax.set_ylabel("epsilon value")
        plt.legend()
        fig.savefig("eps.png")
        
        fig, ax = plt.subplots()
        ax.plot(np.arange(self.rounds_done // EVERY) * EVERY, self.stats["delta"], label="delta")
        ax.yaxis.grid(True, which="major", linestyle='--')
        ax.set_title("Temporal difference")
        ax.set_xlabel("number of rounds")
        ax.set_ylabel("delta value")
        plt.legend()
        fig.savefig("delta.png")

        plt.close("all")
    
        # Store the model
        with open("my-saved-model.pt", "wb") as file:
            pickle.dump(self.model, file)

    self.logger.debug("q_table_len: " + str(len(self.model)))
    self.logger.debug("eps: " + str(self.eps_threshold))


def reward_from_events(self, events: List[str]):
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        "MOVED_TO_SAFE_AREA": 0.1,
        "MOVED_AWAY_FROM_SAVE_AREA": -0.1, 
        "NOT_SAFE_DIRECTION_MOVE_AFTER_BOMB_DROP": -0.25,
        "SAFE_DIRECTION_MOVE_AFTER_BOMB_DROP": 0.25,
        "AVOIDED_EXPLOSION": 0.21,
        "MOVED_INTO_EXPLOSION": -0.21,
        "AVOIDED_BOMB_ABOUT_TO_EXPLODE": 0.21,
        "MOVED_INTO_BOMB_ABOUT_TO_EXPLODE": -0.21,
        "BLOCKED_MOVE": -0.2,
        "SAVE_CRATE_BOMB": 0.15,
        "MISSED_BOMB": -0.15,   
        "INEFFECTIVE_BOMB": -0.15,
        "INVALID_BOMB": -0.1,
        "CLOSER_TO_OBJECT_OF_INTEREST": 0.05,
        "NOT_CLOSER_TO_OBJECT_OF_INTEREST": -0.05,
        "BOMBED_ENEMY": 0.2,
        e.KILLED_OPPONENT: 0.5,
        "CLOSER_TO_ENEMY": 0.01,
        "FURTHER_FROM_ENEMY": -0.01,
        e.COIN_COLLECTED: 0.2,
    }
    
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return np.array([reward_sum], dtype=np.float32)


def get_distance(point_1, point_2):
    """
    Compute eudlidean distance between point_1 and point_2

    :param point_1: First point (x, y)
    :param point_2: Second point (x, y)
    :return: Eudlidean distance (float)
    """
    return np.linalg.norm(np.array(point_1) - np.array(point_2))
