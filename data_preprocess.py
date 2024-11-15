import json
import numpy as np
from tqdm import tqdm
import math

def read_replay_data(file):
    data = []
    with open(file, 'r') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data

def loc_to_state_pos(x, y):
    pos = 0
    if y >= 14:
        pos += 213
        y = 27 - y
        x = 27 - x
    if y > 0:
        pos += y * (y + 1)
    pos += x - 13 + y
    return pos

def action_pos_to_loc(pos):
    y = math.isqrt(pos+14) - 1
    x = pos - y**2 - 2*y + 13
    while x < 13 - y:
        y -= 1
        x = x = pos - y**2 - 2*y + 13
    return x, y

def get_state(turn_frame, player):
    """
    from turn_frame get state, including stationary units
    """
    state = np.zeros(427)
    state[-1] = turn_frame["turnInfo"][1]
    state[210:213] = turn_frame["p1Stats"][:3]
    state[423:426] = turn_frame["p2Stats"][:3]
    for unit_type_i, units_list in enumerate(turn_frame["p1Units"]):
        if 3 <= unit_type_i <= 6:
            continue 
        for j, unit_info in enumerate(units_list):
            loc_x, loc_y = unit_info[0], unit_info[1]
            state_pos = loc_to_state_pos(loc_x, loc_y)
            if unit_type_i < 3:
                state[state_pos] = unit_type_i + 1
            else:
                state[state_pos] += 3
    for unit_type_i, units_list in enumerate(turn_frame["p2Units"]):
        if 3 <= unit_type_i <= 6:
            continue
        for j, unit_info in enumerate(units_list):
            loc_x, loc_y = unit_info[0], unit_info[1]
            state_pos = loc_to_state_pos(loc_x, loc_y)
            if unit_type_i < 3:
                state[state_pos] = unit_type_i + 1
            else:
                state[state_pos] += 3
    if player != 1:
        state[:213], state[213:426] = state[213:426], state[:213]
    return state


def get_action(turn_frame, action_frame, player):
    """
    From turn_frame and the first action_frame get action for player
    """
    action = np.zeros((8, 210))
    key = "p" + str(player) + "Units"
    for unit_type_i in range(len(turn_frame[key])):
        old_unit_list = turn_frame[key][unit_type_i]
        new_unit_list = action_frame[key][unit_type_i]
        old_unit_set_i = set([unit[3] for unit in old_unit_list])
        for j, unit_info in enumerate(new_unit_list):
            loc_x, loc_y, identifier = unit_info[0], unit_info[1], unit_info[3]
            if identifier not in old_unit_set_i:
                if loc_y >= 14:
                    loc_y = 27 - loc_y
                    loc_x = 27 - loc_x
                pos = loc_to_state_pos(loc_x, loc_y)
                action[unit_type_i][pos] = 1
    return action


def parse_replay_data(data, player):
    states = []
    actions = []
    frame_i = 1
    turn_frame = None
    action_frame = None
    while frame_i < len(data):
        turnInfo = data[frame_i]["turnInfo"]
        if turnInfo[0] == 0:
            turn_frame = data[frame_i]
            action_frame = data[frame_i+1]
            states.append(get_state(turn_frame, player))
            actions.append(get_action(turn_frame, action_frame, player))
        frame_i += 1
    return states, actions