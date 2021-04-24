import importlib
import torch
import gym
import utils

from config import Configuration
from per_buffer import PERBuffer
from models import BranchingDQN
from trainer import Trainer
import os
import importlib
import gym_everglades

import sys
sys.path.append('D:\School Work\Spring 2021\COP 4935\Fall2020-Group19-noisy-net')
if __name__ == '__main__':

    # Get configuration
    config_file = sys.argv[1]
    config = Configuration(config_file)

    # Specific Imports

    # Prepare environment
    env = gym.make('everglades-v0')
    players = {}
    names = {}

    # Global initialization
    torch.cuda.init()
    device = torch.device(
        config.device if torch.cuda.is_available() else "cpu")
    torch.autograd.set_detect_anomaly(True)

    # Information about environments

    observation_space = env.observation_space.shape
    action_space = env.num_actions_per_turn
    action_bins = env.num_groups * env.num_nodes
    # Prepare Experience Memory Replay. TODO: Fix hardcoded action space
    memory = PERBuffer(observation_space, (action_space,), config.capacity)

    # Prepare agent
    # agent = BranchingDQN(
    #     observation_space=observation_space,
    #     action_space=action_space,
    #     action_bins=action_bins,
    #     target_update_freq=config.target_update_freq,
    #     learning_rate=config.lr,
    #     gamma=config.gamma,
    #     hidden_dim=config.hidden_dim,
    #     td_target=config.td_target,
    #     device=device
    # )
    map_name = config.map_file
    rand_agent_file = "../random_actions"
    rand_agent_name, rand_agent_extension = os.path.splitext(rand_agent_file)
    rand_agent_mod = importlib.import_module(
        rand_agent_name.replace('../', 'agents.'))
    rand_agent_class = getattr(
        rand_agent_mod, os.path.basename(rand_agent_name))
    rand_player = rand_agent_class(env.num_actions_per_turn, 0, map_name)

    bdqn_player = BranchingDQN(
        observation_space=observation_space[0],
        action_space=action_space,
        action_bins=action_bins,
        target_update_freq=config.target_update_freq,
        learning_rate=config.lr,
        gamma=config.gamma,
        hidden_dim=config.hidden_dim,
        td_target=config.td_target,
        device=device,
        exploration_method=config.exploration_method
    )

    players[0] = rand_player
    names[0] = rand_player.__class__.__name__
    players[1] = bdqn_player
    names[1] = bdqn_player.__class__.__name__

    # Prepare Trainer
    trainer = Trainer(
        model=bdqn_player,
        env=env,
        memory=memory,
        max_steps=config.max_steps,
        max_episodes=config.max_episodes,
        epsilon_start=config.epsilon_start,
        epsilon_final=config.epsilon_final,
        epsilon_decay=config.epsilon_decay,
        start_learning=config.start_learning,
        batch_size=config.batch_size,
        save_update_freq=config.save_update_freq,
        exploration_method=config.exploration_method,
        output_dir=config.output_dir,
        players=players,
        player_num=1,
        config_dir=config.env_config,
        map_file=config.map_file,
        unit_file=config.unit_file,
        env_output_dir=config.env_output_dir,
        pnames=names,
        debug=config.debug,
    )

    # Train
    trainer.loop()
