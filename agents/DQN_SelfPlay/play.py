import importlib
import torch
import os
import gym_everglades
import utils
import gym
from config import Configuration
from models import BranchingQNetwork
import numpy as np
import sys
#sys.path.append('/mnt/d/everglades-ai-wargame/')

if __name__ == "__main__":

    # Get configuration
    config_file = sys.argv[1]
    config = Configuration(config_file)

    # Prepare environment
    env = gym.make('everglades-v0')
    players = {}
    names = {}

    # Global initialization
    torch.cuda.init()
    device = torch.device(
        config.device if torch.cuda.is_available() else "cpu")

    # Information about environments
    observation_space = env.observation_space.shape[0]
    action_space = env.num_actions_per_turn
    action_bins = env.num_groups * env.num_nodes

    # Prepare agent
    map_name = config.map_file
    rand_agent_file = "../random_actions"
    rand_agent_name, rand_agent_extension = os.path.splitext(rand_agent_file)
    rand_agent_mod = importlib.import_module(
        rand_agent_name.replace('../', 'agents.'))
    rand_agent_class = getattr(
        rand_agent_mod, os.path.basename(rand_agent_name))
    rand_player = rand_agent_class(env.num_actions_per_turn, 0, map_name)

    bdqn_player = BranchingQNetwork(
        observation_space=observation_space,
        action_space=action_space,
        action_bins=action_bins,
        hidden_dim=config.hidden_dim,
    )
    bdqn_player_num = 1
    bdqn_player.load_state_dict(torch.load(
        './runs/Newton-train-2-7-21-res/model_state_dict_best'))
    bdqn_player.to(device)
    players[0] = rand_player
    players[1] = bdqn_player
    names[0] = rand_player.__class__.__name__
    names[1] = bdqn_player.__class__.__name__
    action_table = utils.build_action_table(env.num_groups, env.num_nodes)
    # Play
    total_wins = 0
    game_played = 0
    winrate = 0
    for episode in range(config.max_episodes):
        state = env.reset(
            players=players,
            config_dir=config.env_config,
            map_file=config.map_file,
            unit_file=config.unit_file,
            output_dir=config.env_output_dir,
            pnames=names,
            debug=config.debug
        )
        done = False
        action = {}
        while not done:
            for pid in players:
                if pid != rand_player.player_num:
                    state[pid] = torch.from_numpy(
                        state[pid]).float().to(device)
                    with torch.no_grad():
                        action_idx = bdqn_player(state[pid]).squeeze(0)
                        # .numpy().reshape(-1)
                        action_idx = torch.argmax(
                            action_idx, dim=1).reshape(-1)
                    action_idx = action_idx.detach().cpu().numpy()  # .reshape(-1)
                    action[pid] = np.zeros(
                        (env.num_actions_per_turn, 2))
                    # print(action_idx)
                    for n in range(0, len(action_idx)):
                        action[pid][n][0] = action_table[action_idx[n]][0]
                        action[pid][n][1] = action_table[action_idx[n]][1]

                else:
                    action[pid] = rand_player.get_action(state[pid])
            print(action)
            state, reward, done, info = env.step(action)

            if done:
                if reward[bdqn_player_num] == 1:
                    total_wins += 1
                game_played += 1
                winrate = (total_wins/game_played) * 100
        print("Game result: {}".format(reward))
        print("Winrate for Episode {}/{}: {:.2f}%".format(episode,
                                                          config.max_episodes, winrate))
