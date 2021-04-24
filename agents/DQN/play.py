import importlib
import torch
import os
import gym_everglades
import utils
import gym
from OneHotEncode import OneHotEncode
from config import Configuration
from models import BranchingQNetwork, BranchingDQN
from everglades_renderer import Renderer
import numpy as np
import sys
sys.path.append('/mnt/d/everglades-ai-wargame/')

if __name__ == "__main__":

    # Get configuration
    config_file = sys.argv[1]
    config = Configuration(config_file)
    path = sys.argv[2]
    model = sys.argv[3]
    # Prepare environment
    env = gym.make('everglades-v0')
    players = {}
    names = {}

    renderer = Renderer(config.map_file)
    # Global initialization
    torch.cuda.init()
    device = torch.device(
        config.device if torch.cuda.is_available() else "cpu")

    # Information about environments
    observation_space = 105
    action_space = env.num_actions_per_turn
    action_bins = env.num_groups * env.num_nodes

    # Prepare agent
    map_name = config.map_file
    agent_name = "random_actions"
    rand_agent_file = "../"+agent_name
    rand_agent_name, rand_agent_extension = os.path.splitext(rand_agent_file)
    rand_agent_mod = importlib.import_module(
        rand_agent_name.replace('../', 'agents.'))
    rand_agent_class = getattr(
        rand_agent_mod, os.path.basename(rand_agent_name))
    rand_player = rand_agent_class(env.num_actions_per_turn, 0)

    bdqn_player = BranchingDQN(
        observation_space=observation_space,
        action_space=action_space,
        action_bins=action_bins,
        target_update_freq=config.target_update_freq,
        learning_rate=config.lr,
        gamma=config.gamma,
        hidden_dim=config.hidden_dim,
        td_target=config.td_target,
        device=device,
        exploration_method=config.exploration_method,
        architecture=config.architecture
    )
    bdqn_player_num = 1
    bdqn_player.policy_network.load_state_dict(torch.load(
        path + model))
    bdqn_player.eval()
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
            renderer.render(state)
            for pid in players:
                if pid != rand_player.player_num:
                    action[pid] = bdqn_player.get_action(state[pid])
                    # encode_state = state[pid]
                    # encode_state = torch.from_numpy(
                    #     encode_state).float().to(device)
                    # with torch.no_grad():
                    #     action_idx = bdqn_player(encode_state).squeeze(0)
                    #     # .numpy().reshape(-1)
                    #     action_idx = torch.argmax(
                    #         action_idx, dim=1).reshape(-1)
                    # action_idx = action_idx.detach().cpu().numpy()  # .reshape(-1)
                    # action[pid] = np.zeros(
                    #     (env.num_actions_per_turn, 2))
                    
                    # for n in range(0, len(action_idx)):
                    #     action[pid][n][0] = action_table[action_idx[n]][0]
                    #     action[pid][n][1] = action_table[action_idx[n]][1]
                    # print(action[pid])
                else:
                    action[pid] = rand_player.get_action(state[pid])
            #print(action)
            state, reward, done, scores = env.step(action)
            # print(scores)
            if done:
                if reward[bdqn_player_num] == 1:
                    total_wins += 1
                game_played += 1
                winrate = (total_wins/game_played) * 100
        with open(os.path.join(path, "rewards-{}-{}.txt".format(model, agent_name)), 'a') as fout:
                    fout.write("Winrate last 100: {}.\n".format(winrate))
        print("Game result: {}".format(reward))
        print("Winrate for Episode {}/{}: {:.2f}%".format(episode,
                                                          config.max_episodes, winrate))