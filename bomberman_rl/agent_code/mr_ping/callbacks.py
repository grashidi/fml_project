import os
import pickle
import random
from collections import defaultdict, deque, namedtuple
import numpy as np
import settings as s


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
Memory = namedtuple("Memory",
                    ("moves", "position", "bomb_save"))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 4  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

LEARNING_RATE = 0.05
DISCOUNT = 0.95
EVERY = 100

# EPS_START = 0.9
EPS_START = 0.05
EPS_END = 0.05
EPS_DECAY = 10000
MEMORY_SIZE = 6

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"


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

    self.rewards = []
    self.stats = {"avg": [], "max": [], "min": [], "q_table_len": [], "eps": []}


def init_action_space():
    return np.zeros(len(ACTIONS))


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
    self.memory = deque(maxlen=MEMORY_SIZE) 

    if self.train:
        self.eps_threshold = None

        if  not os.path.isfile("my-saved-model.pt"):
            self.logger.info("Setting up model from scratch.")
            self.model = defaultdict(init_action_space)

        if  os.path.isfile("my-saved-model.pt"):
            self.logger.info("Loading model from saved state.")

            with open("my-saved-model.pt", "rb") as file:
                self.model = pickle.load(file)
            
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

        :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # todo Exploration vs exploitation
    pos = game_state["self"][-1]
    
    if self.train:
        self.eps_threshold = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1

        if random.random() <= self.eps_threshold:

            self.logger.debug("Choosing action purely at random.")
            # 80%: walk in any direction. 10% wait. 10% bomb.
            return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])
        else:
            self.logger.debug("Querying model for action.")
            
            curr_state = state_to_features(self, game_state)
            a = ACTIONS[np.argmax(self.model[str(curr_state)])]
            
            return a

    else:
        curr_state = state_to_features(self, game_state)
        a = ACTIONS[np.argmax(self.model[str(curr_state)])]
        
        self.memory.append(Memory(a, game_state["self"][-1], curr_state[2]))

    
        return a


def state_to_features(self, game_state: dict):
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

    field_shape = game_state["field"].shape

    features = np.zeros((6,), dtype=np.float32)

    field = game_state["field"]
    all_paths = explore_paths(game_state)

    # observe surrounding tiles
    free_directions = get_free_directions(game_state["self"][-1], game_state)
    features[0] = sum(val*(2**idx) for idx, val in enumerate(reversed(free_directions)))

    # check distance from save area
    save_area_direction = get_direction_to_save_area(game_state, all_paths)
    features[1] = sum(val*(2**idx) for idx, val in enumerate(reversed(save_area_direction)))

    # check if an escape path after bomb drop exists
    bomb_direction = is_save_to_drop_bomb(game_state, all_paths)
    features[2] = sum(val*(2**idx) for idx, val in enumerate(reversed(bomb_direction)))

    # get the direction of interest
    crate_direction = get_direction_to_closest_crate(field, all_paths)
    coin_direction = get_direction_to_closest_coin(game_state["coins"], all_paths)
    direction_of_interest = coin_direction if not -1 in coin_direction else crate_direction
    features[3] = sum(val*(2**idx) for idx, val in enumerate(reversed(direction_of_interest)))

    # check for bombs about to explode
    bombs_about_to_explode_directions = get_bombs_about_to_explode_directions(game_state)
    features[4] = sum(val*(2**idx) for idx, val in enumerate(reversed(bombs_about_to_explode_directions)))

    # check for enemies in surrounding tiles
    enemy_direction = get_enemy_directions(game_state["self"][-1], game_state["others"])
    features[5] = sum(val*(2**idx) for idx, val in enumerate(reversed(enemy_direction)))
    
    return features


def get_distance_from_bomb(agent_pos, bomb_pos):
    """
    Computes distance between agent and bomb (or other object)

    :param agent_pos: Position of the agent (x, y)
    :param bomb_pos: Position of the bomb (x, y)
    :return: Distance to bomb (x, y).
    """
    x_dist = bomb_pos[0] - agent_pos[0]
    y_dist = bomb_pos[1] - agent_pos[1]

    return (x_dist, y_dist)


def bomb_is_within_reach(game_state):
    """
    Check if agent is within the reach of a bomb

    :param pos: Current position
    :param bomb_locs: Bombs ((pos), timer)
    :return: True if agent is within the reach of a bomb else false
    """
    pos = game_state["self"][-1]

    imminent_explosions = get_imminent_explosion_map(game_state, timer_limit=3)
    
    if imminent_explosions[pos[0], pos[1]]:
        return True

    return False


def get_free_directions(pos, game_state): 
    """
    Get unblocked directions. Possible blockages are:
    crates, bricks, bombs, explosions.
    One-hot encoded [below, above, right, left].

    :param pos: Current position in the field
    :param game_state: Current gamestate
    :return: One-hot encoded array. A one is indicating a blocked cell.
    """
    field = game_state["field"]
    bomb_locs = game_state["bombs"]
    explosion_map = game_state["explosion_map"]

    crate_indicator = get_crate_directions(pos, field)
    brick_indicator = get_brick_directions(pos, field)
    bomb_indicator = get_bomb_locations(pos, bomb_locs)
    explosion_indicator = get_explosions(pos, explosion_map)
    about_to_explode_indicator = get_bombs_about_to_explode_directions(game_state)
    enemy_indicator = get_enemy_directions(pos, game_state["others"])
  
    free_directions = crate_indicator | brick_indicator | bomb_indicator | explosion_indicator | about_to_explode_indicator | enemy_indicator

    return free_directions
 

def get_crate_directions(pos, field):
    """
    Get directions of crates surrounding the current position. 
    One-hot encoded [below, above, right, left].

    :param pos: Current position in the field
    :param game_state: Current gamestate
    :return: One-hot encoded array. A one is indicating a cell with a crate.
    """
    block_state = np.zeros(4, dtype=int)

    # check tile below
    if field[pos[0], pos[1] + 1] == 1:
        block_state[0] = 1
    
    # check tile above
    if field[pos[0], pos[1] - 1] == 1:
        block_state[1]= 1
    
    # check tile right
    if field[pos[0] + 1, pos[1]] == 1:
        block_state[2] = 1
    
    # check tile left
    if field[pos[0] - 1, pos[1]] == 1:
        block_state[3] = 1

    return block_state


def get_brick_directions(pos, field):
    """
    Get directions of bricks surrounding the current position. 
    One-hot encoded [below, above, right, left].

    :param pos: Current position in the field
    :param game_state: Current gamestate
    :return: One-hot encoded array. A one is indicating a cell with a brick.
    """
    block_state = np.zeros(4, dtype=int)

    # check tile below
    if field[pos[0], pos[1] + 1] == -1:
        block_state[0] = 1
    
    # check tile above
    if field[pos[0], pos[1] - 1] == -1:
        block_state[1]= 1
    
    # check tile right
    if field[pos[0] + 1, pos[1]] == -1:
        block_state[2] = 1
    
    # check tile left
    if field[pos[0] - 1, pos[1]] == -1:
        block_state[3] = 1

    return block_state


def get_bomb_locations(pos, bomb_locs):
    """
    Get directions of bombs surrounding the current position. 
    One-hot encoded [below, above, right, left].

    :param pos: Current position in the field
    :param game_state: Current gamestate
    :return: One-hot encoded array. A one is indicating a cell with a bomb.
    """
    block_state = np.zeros(4, dtype=int)

    for bomb in bomb_locs:
        bomb_pos = bomb[0]
        x_dist, y_dist = get_distance_from_bomb(pos, bomb_pos)

        # bomb below
        if y_dist == 1 and x_dist == 0:
            block_state[0] = 1
        
        # bomb above
        if y_dist == -1 and x_dist == 0:
            block_state[1] = 1

        # bomb right
        if x_dist == 1 and y_dist == 0:
            block_state[2] = 1

        # bomb left
        if x_dist == -1 and y_dist == 0:
            block_state[3] = 1
    
    return block_state


def get_explosions(pos, explosion_map): 
    """
    Get directions of explosions surrounding the current position. 
    One-hot encoded [below, above, right, left].

    :param pos: Current position in the field
    :param game_state: Current gamestate
    :return: One-hot encoded array. A one is indicating a cell with a explosion.
    """
    block_state = np.zeros(4, dtype=int)

    # check tile below
    if explosion_map[pos[0], pos[1] + 1] > 0:
        block_state[0] = 1
    
    # check tile above
    if explosion_map[pos[0], pos[1] - 1] > 0:
        block_state[1]= 1
    
    # check tile right
    if explosion_map[pos[0] + 1, pos[1]] > 0:
        block_state[2] = 1
    
    # check tile left
    if explosion_map[pos[0] - 1, pos[1]] > 0:
        block_state[3] = 1
    
    return block_state


def get_cycle_directions(self, pos):
    direction_state = np.zeros(4, dtype=int)
    positions = [m.position for m in self.memory]

    if positions.count((pos[0], pos[1])) > 2:

        if (pos[0], pos[1] + 1) in positions:
            direction_state[0] = 1

        if (pos[0], pos[1] - 1) in positions:
            direction_state[1] = 1

        if (pos[0] + 1, pos[1]) in positions:
            direction_state[2] = 1

        if (pos[0] - 1, pos[1]) in positions:
            direction_state[3] = 1

    return direction_state
    

def get_enemy_directions(pos, other_locs):
    """
    Get directions of enemies surrounding the current position. 
    One-hot encoded [below, above, right, left].

    :param pos: Current position in the field
    :param game_state: Current gamestate
    :return: One-hot encoded array. A one is indicating a cell with an enemy.
    """
    block_state = np.zeros(4, dtype=int)

    for enemy in other_locs:
        enemy_pos = enemy[-1]
        x_dist, y_dist = get_distance_from_bomb(pos, enemy_pos)

        # enemy below
        if y_dist == 1 and x_dist == 0:
            block_state[0] = 1
        
        # enemy above
        if y_dist == -1 and x_dist == 0:
            block_state[1] = 1

        # enemy right
        if x_dist == 1 and y_dist == 0:
            block_state[2] = 1

        # enemy left
        if x_dist == -1 and y_dist == 0:
            block_state[3] = 1
    
    return block_state


def get_bombs_about_to_explode_directions(game_state):
    """
    Get directions of bombs about to explode surrounding the agent's current position. 
    One-hot encoded [below, above, right, left].

    :param game_state: Current gamestate
    :return: One-hot encoded array. A one is indicating a cell with a bomb about to explode.
    """
    explosion_map = np.zeros_like(game_state["explosion_map"])
    size = game_state["field"].shape[0]
    agent_pos = game_state["self"][-1]
    impact_radius = s.BOMB_POWER + 1

    for bomb in game_state["bombs"]:
        bomb_pos = bomb[0]
        timer = bomb[1]
        
        if timer == 0:
            brick_indicator = get_brick_directions(bomb_pos, game_state["field"])
            x_dist, y_dist = get_distance_from_bomb(agent_pos, bomb_pos)
            
            if not brick_indicator[0]:
                if 0 > y_dist and abs(y_dist) <= impact_radius and agent_pos[0] == bomb_pos[0]:
                    explosion_map[bomb_pos[0], bomb_pos[1]:min(size - 1, bomb_pos[1] + abs(y_dist))] = 1.
                else:
                    explosion_map[bomb_pos[0], bomb_pos[1]:min(size - 1, bomb_pos[1] + impact_radius)] = 1.

            if not brick_indicator[1]:
                if 0 < y_dist and abs(y_dist) <= impact_radius and agent_pos[0] == bomb_pos[0]:
                    explosion_map[bomb_pos[0], max(1, bomb_pos[1] - y_dist):bomb_pos[1]] = 1.
                else:
                    explosion_map[bomb_pos[0], max(1, bomb_pos[1] - impact_radius):bomb_pos[1]] = 1.

            if not brick_indicator[2]:
                if 0 > x_dist and abs(x_dist) <= impact_radius and agent_pos[1] == bomb_pos[1]:
                    explosion_map[bomb_pos[0]:min(size - 1, bomb_pos[0] + abs(x_dist)), bomb_pos[1]] = 1.
                else:
                    explosion_map[bomb_pos[0]:min(size - 1, bomb_pos[0] + impact_radius), bomb_pos[1]] = 1.

            if not brick_indicator[3]:
                if 0 < x_dist and abs(x_dist) <= impact_radius and agent_pos[1] == bomb_pos[1]:
                    explosion_map[max(1, bomb_pos[0] - x_dist): bomb_pos[0], bomb_pos[1]] = 1.
                else:
                    explosion_map[max(1, bomb_pos[0] - impact_radius): bomb_pos[0], bomb_pos[1]] = 1.

    return get_explosions(game_state["self"][-1], explosion_map)


def get_impact_map(game_state):
    """
    Creates an impact map which is a combination of current explosions
    and potential explosions of currently deployed bombs. 

    :param game_state: Current gamestate
    :return: 2D numpy array. A one indicates a tile with an explosion.
    """
    current_explosions = game_state["explosion_map"]
    imminent_explosions = get_imminent_explosion_map(game_state)

    impact_map = np.logical_or(current_explosions, imminent_explosions)

    return impact_map


def get_imminent_explosion_map(game_state, timer_limit=3):
    """
    Creates an impact map of potential explosions of currently deployed bombs. 

    :param game_state: Current gamestate
    :return: 2D numpy array. A one indicates a tile with an explosion.
    """
    explosion_map = np.zeros_like(game_state["explosion_map"])
    size = game_state["field"].shape[0]
    agent_pos = game_state["self"][-1]
    impact_radius = s.BOMB_POWER + 1

    for bomb in game_state["bombs"]:
        bomb_pos = bomb[0]
        timer = bomb[1]
        
        if timer <= timer_limit:
            brick_indicator = get_brick_directions(bomb_pos, game_state["field"])
            x_dist, y_dist = get_distance_from_bomb(agent_pos, bomb_pos)
            
            if not brick_indicator[0]:
                explosion_map[bomb_pos[0], bomb_pos[1]:min(size - 1, bomb_pos[1] + impact_radius)] = 1.

            if not brick_indicator[1]:
                explosion_map[bomb_pos[0], max(1, bomb_pos[1] - impact_radius):bomb_pos[1]] = 1.

            if not brick_indicator[2]:
                explosion_map[bomb_pos[0]:min(size - 1, bomb_pos[0] + impact_radius), bomb_pos[1]] = 1.

            if not brick_indicator[3]:
                explosion_map[max(1, bomb_pos[0] - impact_radius): bomb_pos[0], bomb_pos[1]] = 1.

    return explosion_map


def get_direction_to_save_area(game_state, all_paths):
    """
    Get the direction of tile unaffected by an explosion.

    :param game_state: Current gamestate.
    :param all_paths: All available paths from the current position
                      sorted in ascending order by length.
    :return: One-hot encoded array. A one is indicating a tile leading into 
             a safe direction.
    """
    save_paths = []
    if bomb_is_within_reach(game_state):
        impact_map = get_impact_map(game_state)

        for path in all_paths:
            for i, cell in enumerate(path[:4]):
                if not impact_map[cell[0], cell[1]]:
                    save_paths += [(i, path)]
                    break
 
        if save_paths:
            shortest_save_path = sorted(save_paths, key=lambda k: k[0])[0][-1]
            return get_path_direction(shortest_save_path)
    
    return np.zeros(4, dtype=int)
        

def is_save_to_drop_bomb(game_state, all_paths):
    """
    Checks if there is an available escape path if the agent
    would drop a bomb.

    :param game_state: Current gamestate.
    :param all_paths: All available paths from the current position
                      sorted in ascending order by length.
    :return: One-hot encoded array. A one is indicating a tile leading into 
             a safe direction.
    """
    if not bomb_is_within_reach(game_state):
        for path in all_paths:
            if path_goes_around_corner(path[:4]) and not bomb_is_in_path(path[1:6], game_state["bombs"]):
                return get_path_direction(path)
        
        for path in all_paths:
            if len(path) > (s.BOMB_POWER) and not bomb_is_in_path(path[1:6], game_state["bombs"]):
                return get_path_direction(path)

    return np.zeros(4, dtype=int)


def get_direction_to_closest_crate(field, all_paths):
    """
    Gets the direction to the closest crate.

    :param game_state: Current gamestate.
    :param all_paths: All available paths from the current position
                      sorted in ascending order by length.
    :return: One-hot encoded array. A one is indicating a tile leading to 
             a crate.
    """
    distances = []

    for path in all_paths:
        if field[path[0][0], path[0][1] + 1] == 1:
            return [0, 0, 0, 0]
        
        if field[path[0][0], path[0][1] - 1] == 1:
            return [0, 0, 0, 0]
        
        if field[path[0][0] + 1, path[0][1]] == 1:
            return [0, 0, 0, 0]
        
        if field[path[0][0] - 1, path[0][1]] == 1:
            return [0, 0, 0, 0] 
        
        for i, cell in enumerate(path[1:]):
            if field[cell[0], cell[1] + 1] == 1:
                distances += [(i, get_path_direction(path))]
                break
            
            if field[cell[0], cell[1] - 1] == 1:
                distances += [(i, get_path_direction(path))]
                break
            
            if field[cell[0] + 1, cell[1]] == 1:
                distances += [(i, get_path_direction(path))]
                break
            
            if field[cell[0] - 1, cell[1]] == 1:
                distances += [(i, get_path_direction(path))]
                break

    return sorted(distances, key=lambda k: k[0])[0][1] if len(distances) > 0 else [-1]


def get_direction_to_closest_coin(coin_locs, all_paths):
    """
    Gets the direction to the closest coin.

    :param game_state: Current gamestate.
    :param all_paths: All available paths from the current position
                      sorted in ascending order by length.
    :return: One-hot encoded array. A one is indicating a tile leading to 
             a coin.
    """
    distances = []

    for path in all_paths:
        for i, cell in enumerate(path):
            for coin in coin_locs:
                if cell == coin:
                    distances += [(i, get_path_direction(path))]
                    break
                
    return sorted(distances, key=lambda k: k[0])[0][1] if len(distances) > 0 else [-1]
    

def isNotVisited(x, path):
    """
    Helper function. Checks if tile is already included in the path.

    :param x: Current tile.
    :param path: The current path.
    :return: True if tile is not yet in the path.
    """
    for i in range(len(path)):
        if (path[i] == x):
            return 0
             
    return 1


def explore_paths(game_state, max_len=10):
    """
    Gets all available unblocked paths from the agent's current position.
    Paths are sorted in ascending order by length.

    :param game_state: Current gamestate.
    :param max_len: Maximal length of the paths returned
    :return: List with list of tuple containing the coordinates of the paths.
    """
    src = game_state["self"][-1]
    grid = game_state["field"]
    bomb_locs = game_state["bombs"]
    explosion_map = game_state["explosion_map"]
                   
    q = deque()
 
    all_paths = []
    path = []
    path.append(src)
    q.append(path.copy())
     
    while q:
        path = q.popleft()
        last = path[-1]

        if (len(path) >= max_len):
            all_paths += [path]
            continue
 
        free_directions = get_free_directions(last, game_state) 

        dead_end = True
        for i, cell in enumerate([(last[0], last[1] + 1), (last[0], last[1] - 1),
                                  (last[0] + 1, last[1]), (last[0] - 1, last[1])]):

            if isNotVisited(cell, path) and not free_directions[i]:
                dead_end = False
                newpath = path.copy()
                newpath += [cell]
                q.append(newpath)

        if dead_end:
            all_paths += [path]
    
    return sorted(all_paths, key=lambda k: len(k)) 


def path_goes_around_corner(path):
    """
    Checks if given path changes its direction.

    :param path: Path to be checked.
    :return: True if paths changes its direction.
    """
    source = path[0]

    for cell in path[1:]:
        if cell[0] != source[0] and cell[1] != source[1]:
            return True

    return False


def get_path_direction(path):
    """
    Gets the initial direction of the path. The initial
    direction is based on the path's first two tiles.

    :param path: Path to be checked.
    :return: One-hot encoded array indicating the path's inital direction.
             ([down, up, right, left])
    """
    direction = np.zeros(4, dtype=int) # below, above, right, left
    
    if len(path) > 1:
        x_dist = path[0][0] - path[1][0]
        y_dist = path[0][1] - path[1][1]
        
        if y_dist == -1:
            direction[0] = 1

        if y_dist == 1:
            direction[1] = 1

        if x_dist == -1:
            direction[2] = 1

        if x_dist == 1:
            direction[3] = 1
    
    return direction
    

def bomb_is_in_path(path, bomb_locs):
    """
    Checks if a bomb is located int the given path.

    :param path: Path to be checked.
    :param bomb_locs: Bombs taken from the game state ([(x, y), timer]).
    :return: True if bomb is in path.
    """
    for cell in path:
        for bomb in bomb_locs:
            bomb_pos = bomb[0]
            if cell == bomb_pos:
                return True
    
    return False
    
