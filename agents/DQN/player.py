import os
import numpy as np
import time
import json
import gym
import gym_everglades
import random

class PlayerHelper:
    def __init__(self, action_space, player_num, map_name):
        self.action_space = action_space
        self.num_groups = 12
        self.player_num = player_num
        with open('../../config/' + map_name) as fid:
            self.map_dat = json.load(fid)
        
        self.nodes_array = []
        for i, in_node in enumerate(self.map_dat['nodes']):
            self.nodes_array.append(in_node['ID'])
            
        self.num_nodes = len(self.map_dat['nodes'])
        self.num_actions = action_space
        
        self.shape = (self.num_actions, 2)
        self.action_choices = self.get_action_choices(
            (self.num_groups * len(self.nodes_array), 2))

        self.unit_config = {
            0: [('controller', 1), ('striker', 5)],
            1: [('controller', 3), ('striker', 3), ('tank', 3)],
            2: [('tank', 5)],
            3: [('controller', 2), ('tank', 4)],
            4: [('striker', 10)],
            5: [('controller', 4), ('striker', 2)],
            6: [('striker', 4)],
            7: [('controller', 1), ('striker', 2), ('tank', 3)],
            8: [('controller', 3)],
            9: [('controller', 2), ('striker', 4)],
            10: [('striker', 9)],
            11: [('controller', 20), ('striker', 8), ('tank', 2)]
        }
    
    # Listing of all possible actions
    def get_action_choices(self, shape):
        action_choices = np.zeros(shape)
        group_id = 0
        node_id = 1
        for i in range(0, action_choices.shape[0]):
            if i > 0 and i % 11 == 0:
                group_id += 1
                node_id = 1
            action_choices[i] = [group_id, node_id]
            node_id += 1
        return action_choices
    
    def get_action(self, obs):
        action = np.zeros(self.shape)
        action_idx = random.sample(
            range(0, len(self.action_choices)), self.num_actions)        
        for i in range(0, self.num_actions):
            action[i] = self.action_choices[action_idx[i]]
        return (action_idx, action)
    
    def legal_moves(self, obs):
        group_status = np.array([[45, 47, 48], [50, 52, 53], [55, 57, 58], [60, 62, 63],
                                 [65, 67, 68], [70, 72, 73], [75, 77, 78], [80, 82, 83],
                                 [85, 87, 88], [90, 92, 93], [95, 97, 98], [100, 102, 103]])
        group_legal = np.full((132,), False, dtype=bool)
        i = 0
        for group in group_status:
            if ((obs[group[1]] != 0) and (obs[group[2]] == 0)):
                node = int(obs[group[0]])
                connected_nodes = [j['ConnectedID'] for j in self.map_dat['nodes'][node-1]['Connections']]
                connected_nodes.append(node)
                for k in range(len(connected_nodes)):
                    group_legal[i*11 + (connected_nodes[k]-1)] = True
            i += 1
        return group_legal