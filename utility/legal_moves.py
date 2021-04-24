import numpy as np
import json

class Legal_Moves:
    def __init__(self, map_name):
        with open('../../config/' + map_name) as fid:
            self.map_dat = json.load(fid)
 
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