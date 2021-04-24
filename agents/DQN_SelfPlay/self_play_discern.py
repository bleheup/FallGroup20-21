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
import imageio

# Change this line to be the path to the project
sys.path.append('D:\School Work\Spring 2021\COP 4935\Fall2020-Group19-noisy-net')

if __name__ == "__main__":

    Renderer_mod = importlib.import_module('everglades_renderer')
    Renderer = getattr(Renderer_mod, 'Renderer')

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

    # Picks an agent from the discernible actions to play against
    opponent_choice = 1
    rand_agent_class = None
    if opponent_choice == 1:
        rand_agent_mod = importlib.import_module('agents.random_actions')
        rand_agent_class = getattr(rand_agent_mod, 'random_actions')
    elif opponent_choice == 2:
        rand_agent_mod = importlib.import_module('agents.cycle_target_node11P2')
        rand_agent_class = getattr(rand_agent_mod, 'cycle_targetedNode11P2')
    elif opponent_choice == 3:
        rand_agent_mod = importlib.import_module('agents.swarm_agent')
        rand_agent_class = getattr(rand_agent_mod, 'SwarmAgent')
    elif opponent_choice == 4:
        rand_agent_mod = importlib.import_module('agents.same_commands')
        rand_agent_class = getattr(rand_agent_mod, 'same_commands')
    elif opponent_choice == 5:
        rand_agent_mod = importlib.import_module('agents.all_cycle')
        rand_agent_class = getattr(rand_agent_mod, 'all_cycle')
    elif opponent_choice == 6:
        rand_agent_mod = importlib.import_module('agents.base_rush_v1')
        rand_agent_class = getattr(rand_agent_mod, 'base_rushV1')
    elif opponent_choice == 7:
        rand_agent_mod = importlib.import_module('agents.cycle_target_node1')
        rand_agent_class = getattr(rand_agent_mod, 'cycle_targetedNode1')
    
    if opponent_choice == 1:
        rand_player = rand_agent_class(env.num_actions_per_turn, 0, map_name)
    else:
        rand_player = rand_agent_class(env.num_actions_per_turn, 0)

    
    bdqn_player = BranchingQNetwork(
        observation_space=observation_space,
        action_space=action_space,
        action_bins=action_bins,
        hidden_dim=config.hidden_dim,
        exploration_method = config.exploration_method,
    )
    bdqn_player_num = 1

    """Load Pre-Saved Model"""
    #bdqn_player.load_state_dict(torch.load(
    #    './runs/Michael_Local/model_state_dict_last'))
    #bdqn_player.load_state_dict(torch.load(
    #    './agents/bd3qn/runs/Newton-train-2-7-21-res/model_state_dict_best'))
    bdqn_player.load_state_dict(torch.load(
        './runs/Michael_Local_2/model_state_dict_last'))
    bdqn_player.to(device)
    players[0] = rand_player
    players[1] = bdqn_player
    names[0] = rand_player.__class__.__name__
    names[1] = bdqn_player.__class__.__name__
    action_table = utils.build_action_table(env.num_groups, env.num_nodes)

    to_render = True
    r = Renderer(map_name,frame_collection=True)
    gif_frames = []
    
    # Play
    total_wins = 0
    game_played = 0
    winrate = 0
    for episode in range(1):
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
        if to_render == True:
            gif_frames.append(r.render(state))
        while not done:
            for pid in players:
                if pid != rand_player.player_num:
                    state[pid] = torch.from_numpy(
                        state[pid]).float().to(device)
                    with torch.no_grad():
                        action_idx = bdqn_player(state[pid]).squeeze(0)
                        action_idx = torch.argmax(
                            action_idx, dim=1).reshape(-1)
                    action_idx = action_idx.detach().cpu().numpy()
                    action[pid] = np.zeros(
                        (env.num_actions_per_turn, 2))
                    for n in range(0, len(action_idx)):
                        action[pid][n][0] = action_table[action_idx[n]][0]
                        action[pid][n][1] = action_table[action_idx[n]][1]
                else:
                    action[pid] = rand_player.get_action(state[pid])
            #print(action)
            state, reward, done, info = env.step(action)

            if to_render == True:
                gif_frames.append(r.render(state))

            if done:
                if reward[bdqn_player_num] == 1:
                    total_wins += 1
                game_played += 1
                winrate = (total_wins/game_played) * 100
        print("Game result: {}".format(reward))
        print("Winrate for Episode {}/{}: {:.2f}%".format(episode,
                                                          config.max_episodes, winrate))
        if to_render == True:
            imageio.mimsave('./selfplay_vs_{}.gif'.format(rand_agent_class.__name__), gif_frames, duration = 0.3)
            to_render = False
