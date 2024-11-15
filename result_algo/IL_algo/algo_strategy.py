import gamelib
import random
import math
import warnings
from sys import maxsize
import json
from data_preprocess import *
from model import *
import numpy as np
import torch

"""
Most of the algo code you write will be in this file unless you create new
modules yourself. Start by modifying the 'on_turn' function.

Advanced strategy tips: 

  - You can analyze action frames by modifying on_action_frame function

  - The GameState.map object can be manually manipulated to create hypothetical 
  board states. Though, we recommended making a copy of the map to preserve 
  the actual current map state.
"""

class AlgoStrategy(gamelib.AlgoCore):
    def __init__(self):
        super().__init__()
        seed = random.randrange(maxsize)
        random.seed(seed)
        gamelib.debug_write('Random seed: {}'.format(seed))

    def on_game_start(self, config):
        """ 
        Read in config and perform any initial setup here 
        """
        gamelib.debug_write('Configuring your custom algo strategy...')
        self.config = config
        global WALL, SUPPORT, TURRET, SCOUT, DEMOLISHER, INTERCEPTOR, MP, SP
        WALL = config["unitInformation"][0]["shorthand"]
        SUPPORT = config["unitInformation"][1]["shorthand"]
        TURRET = config["unitInformation"][2]["shorthand"]
        SCOUT = config["unitInformation"][3]["shorthand"]
        DEMOLISHER = config["unitInformation"][4]["shorthand"]
        INTERCEPTOR = config["unitInformation"][5]["shorthand"]
        MP = 1
        SP = 0
        self.agent = ILAgent(state_dim=427, action_dim=1680)
        self.agent.load_state_dict(torch.load('IL-algo/model_weights.pth', weights_only=True))
        self.agent.eval()
        self.hist_action = np.zeros((3, 210))
        # self.hist_action[2][45:53] = 1
        self.hist_action[1][34:42] = 1
        for pos in [131, 109, 89, 71, 55]:
            self.hist_action[2][pos] = 1
        

    def on_turn(self, turn_state):
        """
        This function is called every turn with the game state wrapper as
        an argument. The wrapper stores the state of the arena and has methods
        for querying its state, allocating your current resources as planned
        unit deployments, and transmitting your intended deployments to the
        game engine.
        """
        game_state = gamelib.GameState(self.config, turn_state)
        gamelib.debug_write('Performing turn {} of your custom algo strategy'.format(game_state.turn_number))
        game_state.suppress_warnings(True)  #Comment or remove this line to enable warnings.
        turn_frame = json.loads(turn_state)
        state = get_state(turn_frame, 1)
        action = self.agent(torch.tensor(state, dtype=torch.float32))
        if game_state.turn_number > 2:
            self.recover_construction(game_state)
        if game_state.turn_number < 2:
            self.update_hist_construction(action)
        self.deploy_action(action, game_state)
        self.add_attack(game_state)
        game_state.submit_turn()

    """
    NOTE: All the methods after this point are part of the sample starter-algo
    strategy and can safely be replaced for your custom algo.
    """

    def deploy_action(self, action, game_state):
        action = torch.reshape(action, (8, -1))
        for i in range(action.shape[0]):
            unit_action = action[i].detach().numpy()
            action_pos_array = np.where(unit_action>0.5)[0]
            if i <= 2:
                for pos in action_pos_array:
                    loc_x, loc_y = action_pos_to_loc(pos)
                    if (int(loc_x), int(loc_y)) != (22, 11) and (int(loc_x), int(loc_y)) != (20, 8):
                        game_state.attempt_spawn(self.index_to_unit(i), [int(loc_x), int(loc_y)])
            elif i <= 5:
                for pos in action_pos_array:
                    loc_x, loc_y = action_pos_to_loc(pos)
                    game_state.attempt_spawn(self.index_to_unit(i), [int(loc_x), int(loc_y)], 3)
            elif i == 6:
                for pos in action_pos_array:
                    loc_x, loc_y = action_pos_to_loc(pos)
                    game_state.attempt_remove([int(loc_x), int(loc_y)])
            elif i == 7:
                for pos in action_pos_array:
                    loc_x, loc_y = action_pos_to_loc(pos)
                    game_state.attempt_upgrade([int(loc_x), int(loc_y)])
        return

    def recover_construction(self, game_state):
        # recover construction from hist action
        for i in range(3):
            unit_action = self.hist_action[i]
            action_pos_array = np.where(unit_action>0)[0]
            for pos in reversed(list(action_pos_array)):
                loc_x, loc_y = action_pos_to_loc(pos)
                game_state.attempt_spawn(self.index_to_unit(i), [int(loc_x), int(loc_y)])             
        return

    def add_attack(self, game_state):
        # if MP resource available, add attack
        start = [[9,4],[18,4]]
        if game_state.get_resource(MP) and game_state.turn_number % 3 == 1:
            attack_units = np.random.randint(6, size=2)
            game_state.attempt_spawn(self.index_to_unit(attack_units[1]%3 + 3), start[attack_units[1]%2], 1000)
        return
                
    def update_hist_construction(self, action):
        # update hist construction
        action = torch.reshape(action, (8, -1))
        for i in range(3):
            unit_action = unit_action = action[i].detach().numpy()
            unit_action[unit_action>0.5] = 1
            unit_action[unit_action<0.5] = 0
            self.hist_action[i] = self.hist_action[i] + unit_action
        self.hist_action[self.hist_action>0] = 1
        return
        
        # attack action need to be adjusted
        # imitate other algo?

    def index_to_unit(self, index):
        return self.config["unitInformation"][index]["shorthand"]

    def on_action_frame(self, turn_string):
        """
        This is the action frame of the game. This function could be called 
        hundreds of times per turn and could slow the algo down so avoid putting slow code here.
        Processing the action frames is complicated so we only suggest it if you have time and experience.
        Full doc on format of a game frame at in json-docs.html in the root of the Starterkit.
        """
        # state = json.loads(turn_string)
        # events = state["events"]
        # breaches = events["breach"]
        # for breach in breaches:
        #     location = breach[0]
        #     unit_owner_self = True if breach[4] == 1 else False
        #     if not unit_owner_self:
        #         gamelib.debug_write("Got scored on at: {}".format(location))
        return


if __name__ == "__main__":
    algo = AlgoStrategy()
    algo.start()