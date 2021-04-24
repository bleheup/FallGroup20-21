from scipy.ndimage.filters import gaussian_filter1d
from matplotlib import animation
import os
import numpy as np
import pandas as pd
import torch
import random

import matplotlib.pyplot as plt
plt.style.use('ggplot')


def build_action_table(num_groups, num_nodes):
    action_choices = np.zeros((num_groups * num_nodes, 2))
    group_id = 0
    node_id = 1
    for i in range(0, action_choices.shape[0]):
        if i > 0 and i % num_nodes == 0:
            group_id += 1
            node_id = 1
        action_choices[i] = [group_id, node_id]
        node_id += 1
    return action_choices


def _save(agent, rewards, env_name, output_dir, model_type):

    path = './runs/{}/'.format(output_dir)
    try:
        os.makedirs(path)
    except:
        pass

    torch.save(agent.policy_network.state_dict(),
               os.path.join(path, 'model_state_dict'+model_type))

    plt.cla()
    plt.plot(rewards, c='#bd0e3a', alpha=0.3)
    plt.plot(gaussian_filter1d(rewards, sigma=5), c='#bd0e3a', label='Winrate')
    plt.xlabel('Episodes')
    plt.ylabel('Winrate')
    plt.title('Branching DDQN ({}): {}'.format(agent.td_target, env_name))
    plt.savefig(os.path.join(path, 'reward.png'))

    pd.DataFrame(rewards, columns=['Reward']).to_csv(
        os.path.join(path, 'rewards.csv'), index=False)



def save_checkpoint(agent, rewards, env_name, output_dir):
    _save(agent, rewards, env_name, output_dir, "_last")


def save_best(agent, rewards, env_name, output_dir):
    _save(agent, rewards, env_name, output_dir, "_best")
