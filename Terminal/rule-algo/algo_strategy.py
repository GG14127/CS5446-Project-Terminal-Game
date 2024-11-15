import gamelib
import random
import math
import warnings
from sys import maxsize
import json


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

        # basic structures
        self.turret_locations = [[1, 12], [2, 12], [25, 12], [26, 12], [7, 9], [8, 9], [19, 9], [20, 9], [13, 6], [14, 6]]
        self.support_locations = []
        self.wall_locations = [[0, 13], [27,13]]
        for location in self.turret_locations:
            self.wall_locations.append([location[0],location[1]+1])

        # add on structures
        self.add_turret_locations = {
            'L': [[x, 12] for x in range(4, 14, 2)],
            'R': [[x, 12] for x in range(23, 13, -2)]
        }
        self.add_support_locations = {
            'L': [[x, 10] for x in range(3, 13)],
            'R': [[x, 10] for x in range(24, 14, -1)]
        }
        self.add_wall_locations = {
            'L': [[24, 12], [23, 11], [22, 10], [21, 10], [18, 9], [17, 8], [16, 7], [15, 7]],
            'R': [[3, 12], [4, 11], [5, 10], [6, 10], [9, 9], [10, 8], [11, 7], [12, 7]]
        }

        # attack_start location
        self.attack_location = {
            'L': [2, 11],
            'R': [25, 11]
        }
        self.attack_period = 2
        self.attack_edge = 'R'

        # # reactive structures
        # self.reactive_turret = []
        # self.reactive_support = []
        # self.reactive_wall = []

        # last turn damaged info
        self.damaged_suport = []
        self.damaged_turret = []

        # last turn score info
        self.last_scored_turn = -1
        self.dangered_location = []
        self.scored_on_locations = []

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

        self.analyze_turn_state(game_state, turn_state)

        self.strategy(game_state)

        game_state.submit_turn()

    def on_action_frame(self, turn_string):
        """
        This is the action frame of the game. This function could be called 
        hundreds of times per turn and could slow the algo down so avoid putting slow code here.
        Processing the action frames is complicated so we only suggest it if you have time and experience.
        Full doc on format of a game frame at in json-docs.html in the root of the Starterkit.
        """
        # Let's record at what position we get scored on
        state = json.loads(turn_string)
        events = state["events"]
        breaches = events["breach"]
        for breach in breaches:
            location = breach[0]
            unit_owner_self = True if breach[4] == 1 else False
            # When parsing the frame data directly, 
            # 1 is integer for yourself, 2 is opponent (StarterKit code uses 0, 1 as player_index instead)
            if not unit_owner_self:
                gamelib.debug_write("Got scored on at: {}".format(location))
                self.scored_on_locations.append(location)
                gamelib.debug_write("All locations: {}".format(self.scored_on_locations))
            else:
                self.last_scored_turn = state["turnInfo"][1]


    def analyze_turn_state(self, game_state, turn_state):
        """
        Analyze turn/game state and adjust defence and attack layout
        """
        self.damaged_turret = []
        self.damaged_suport = []

        return

    def strategy(self, game_state):
        """
        build defences and set attackers
        """
        # Check and Place defenses: Turret, Wall, Support
        self.build_defences(game_state)
        # Check and Place Attackers: Demolisher, Interceptor, Scout
        self.set_attackers(game_state)

        return

    def build_defences(self, game_state):
        """
        Check and build basic defenses based on recorded locations.
        Expand defences if needed and available
        """

        # Place structure units accoding to current layout
        game_state.attempt_spawn(TURRET, self.turret_locations)
        game_state.attempt_spawn(WALL, self.wall_locations)
        game_state.attempt_upgrade(self.wall_locations)

        if game_state.turn_number - self.last_scored_turn > self.attack_period:
            if game_state.get_resource(SP):
                self.expand_defences(game_state)
        
        if game_state.get_resource(SP):
            self.upgrade_defences(game_state)

    def expand_defences(self, game_state):
        """
        add defences to build way to support attackers
        add defences for reaction
        """
        for wall_location in self.add_wall_locations[self.attack_edge]:
            game_state.attempt_spawn(WALL, wall_location)
        for turret_location in self.add_turret_locations[self.attack_edge]:
            if game_state.attempt_spawn(TURRET, turret_location) or game_state.contains_stationary_unit(turret_location):
                game_state.attempt_spawn(WALL, [turret_location[0], turret_location[1]+1])
                game_state.attempt_spawn(WALL, [turret_location[0]+1, turret_location[1]])
                game_state.attempt_spawn(WALL, [turret_location[0]+1, turret_location[1]])
                
                for support_location in self.add_support_locations[self.attack_edge]:
                    if (self.attack_edge == 'R' and support_location[0] >= turret_location[0]) or (self.attack_edge == 'L' and support_location[0] <= turret_location[0]):
                        game_state.attempt_spawn(SUPPORT, support_location)
                    else:
                        break
            else:
                break
    
        # Place reactive structures
        # for location in self.scored_on_locations:
        #     # Build turret one space above so that it doesn't block our own edge spawn locations
        #     build_location = [location[0], location[1]+1]
        #     game_state.attempt_spawn(TURRET, build_location)

    def upgrade_defences(self, game_state):
        # Upgrade structures
        for turret_location in self.add_turret_locations[self.attack_edge]:
            game_state.attempt_upgrade(turret_location)
        # for turret_location in self.turret_locations:
        #     game_state.attempt_upgrade(turret_location)
        return


    def set_attackers(self, game_state):
        if game_state.turn_number % self.attack_period == 1:
            game_state.attempt_spawn(DEMOLISHER, self.attack_location[self.attack_edge], 2)
            game_state.attempt_spawn(INTERCEPTOR, self.attack_location[self.attack_edge], 100)


if __name__ == "__main__":
    algo = AlgoStrategy()
    algo.start()
