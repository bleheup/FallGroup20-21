import numpy as np

from torch.utils import tensorboard

import copy

from utils import save_checkpoint, save_best, build_action_table
from datetime import datetime
import os


class Trainer:
    def __init__(self, model,
                 env,
                 memory,
                 max_steps,
                 max_episodes,
                 epsilon_start,
                 epsilon_final,
                 epsilon_decay,
                 start_learning,
                 batch_size,
                 save_update_freq,
                 exploration_method,
                 output_dir,
                 players,
                 player_num,
                 config_dir,
                 map_file,
                 unit_file,
                 env_output_dir,
                 pnames,
                 debug):
        self.model = model
        self.env = env
        self.memory = memory
        self.max_steps = max_steps
        self.max_episodes = max_episodes
        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        self.start_learning = start_learning
        self.batch_size = batch_size
        self.save_update_freq = save_update_freq
        self.output_dir = output_dir
        self.action_table = build_action_table(env.num_groups, env.num_nodes)
        self.players = players
        self.player_num = player_num
        self.config_dir = config_dir
        self.map_file = map_file
        self.unit_file = unit_file
        self.env_output_dir = env_output_dir
        self.pnames = pnames
        self.debug = debug
        self.exploration_method = exploration_method
        self.nodes_array = []
        for i in range(1, self.env.num_nodes + 1):
            self.nodes_array.append(i)
        self._create_opponent_pool()
        self.opp_num = 0
        self.episode_cnt = 0
        self.opp_save_freq = 10 # How many games before saving oppoent
        self.opp_choose_freq = 0.2 # How often [0,1] to sample from the opponent pool

    def _exploration(self, step):
        return self.epsilon_final + (self.epsilon_start - self.epsilon_final) * np.exp(-1. * step / self.epsilon_decay)

    # Initializes an opponent pool for agent to sample from.
    # Can be optionally initialized with a list of opponents, otherwise uses self
    def _create_opponent_pool(self, opp_list = None):
        self.opponent_pool = []
        self.opponent_count = 0
        # Adds a copy of self to pool
        if opp_list == None:
            self.add_self()
        else:
            for opponent in opp_list:
                self.add_opponent(opponent)
        return

    def choose_opponent(self):
        choice = np.random.random()
        # Samples from all available opponents
        if choice < self.opp_choose_freq:
            self.opp_num = np.random.randint(0,self.opponent_count)
            self.players[int(not self.player_num)] = self.opponent_pool[self.opp_num]
        # Uses last saved state
        else:
            self.players[int(not self.player_num)] = self.opponent_pool[self.opponent_count-1]
        return

    # Adds a new opponent to the pool (by default it will add the current active player)
    def add_opponent(self, opponent = None):
        # Adds a copy of the 
        if opponent == None:
            self.add_self()
        else:
            self.opponent_pool.append(opponent)
            self.opponent_count += 1
        return
    
    # Adds a copy of the current active player to opponent pool
    def add_self(self):
        self_opponent = copy.deepcopy(self.players[self.player_num])
        self.opponent_pool.append(self_opponent)
        self.opponent_count += 1
        return

    def loop(self):
        #state = self.env.reset()
        state = self.env.reset(
            players=self.players,
            config_dir=self.config_dir,
            map_file=self.map_file,
            unit_file=self.unit_file,
            output_dir=self.env_output_dir,
            pnames=self.pnames,
            debug=self.debug
        )
        num_of_wins = 0
        episode_winrate = 0
        total_games_played = 0
        all_winrate = []
        highest_winrate = 0
        w = tensorboard.SummaryWriter()
        time = datetime.now().strftime('%Y%m%d_%H%M%S')
        path = './runs/{}/'.format(self.output_dir)
        try:
            os.makedirs(path)
        except:
            pass

        for step in range(self.max_steps):
            epsilon = self._exploration(step)
            action_idx = []
            action = {}
            for pid in self.players:
                if pid == self.player_num:
                    # If noisy network or non-random actions are chosen
                    if self.exploration_method == "Noisy" or np.random.random_sample() > epsilon:
                        # The action indexes are needed for updating, thus get_action does not suffice
                        action_idx = self.model.get_action_idx(state[pid])
                        action[pid] = np.zeros(
                            (self.env.num_actions_per_turn, 2))
                        for n in range(0, len(action_idx)):
                            action[pid][n][0] = self.action_table[action_idx[n]][0]
                            action[pid][n][1] = self.action_table[action_idx[n]][1]
                    # Performs a random action
                    else:
                        #print("not here")
                        action_idx = np.random.choice(
                            len(self.action_table), size=7)
                        action[pid] = np.zeros(
                            (self.env.num_actions_per_turn, 2))
                        for n in range(0, len(action_idx)):
                            action[pid][n][0] = self.action_table[action_idx[n]][0]
                            action[pid][n][1] = self.action_table[action_idx[n]][1]
                else:
                    action[pid] = self.players[pid].get_action(state[pid])


            next_state, reward, done, infos = self.env.step(action)

            if done:
                # Adds copies of self to list of opponents at certain episode counts
                self.episode_cnt += 1
                if self.episode_cnt % self.opp_save_freq == 0:
                    print('ADDING NEW OPPONENT')
                    self.add_opponent()

                self.choose_opponent()
                next_state = self.env.reset(
                    players=self.players,
                    config_dir=self.config_dir,
                    map_file=self.map_file,
                    unit_file=self.unit_file,
                    output_dir=self.env_output_dir,
                    pnames=self.pnames,
                    debug=self.debug
                )
                if reward[self.player_num] == 1:
                    num_of_wins += 1
                total_games_played += 1
                print("Result on game {}: {}".format(
                    len(all_winrate), reward))
                episode_winrate = (num_of_wins/total_games_played) * 100
                all_winrate.append(episode_winrate)
                with open(os.path.join(path, "rewards-{}.txt".format(time)), 'a') as fout:
                    fout.write("{}\n".format(episode_winrate))
                print("Current winrate: {}%".format(episode_winrate))
                w.add_scalar("winrate",
                             episode_winrate, global_step=len(all_winrate))
                if episode_winrate > highest_winrate:
                    highest_winrate = episode_winrate
                    save_best(self.model, all_winrate,
                              "Evergaldes", self.output_dir)

            self.memory.store(
                state[self.player_num],
                action_idx,
                reward[self.player_num],
                next_state[self.player_num],
                done
            )
            state = next_state

            if step > self.start_learning:
                loss = self.model.update_policy(
                    self.memory.miniBatch(self.batch_size), self.memory)
                with open(os.path.join(path, "loss-{}.txt".format(time)), 'a') as fout:
                    fout.write("{}\n".format(loss))
                w.add_scalar("loss/loss", loss, global_step=step)

            if step % self.save_update_freq == 0:
                save_checkpoint(self.model, all_winrate,
                                "Evergaldes", self.output_dir)

            if len(all_winrate) == self.max_episodes:
                save_checkpoint(self.model, all_winrate,
                                "Evergaldes", self.output_dir)
                break

        w.close()
