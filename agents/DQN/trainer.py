import numpy as np
from OneHotEncode import OneHotEncode
from torch.utils import tensorboard
import random
from utils import save_checkpoint, save_best, build_action_table
from datetime import datetime
import os
import importlib
from player import PlayerHelper

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
                 debug,
                 renderer,
                 isNSteps):
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
        self.player_helper = PlayerHelper(7,1, "../config/DemoMap.json")
        self.players = players
        self.player_num = player_num
        self.config_dir = config_dir
        self.map_file = map_file
        self.unit_file = unit_file
        self.env_output_dir = env_output_dir
        self.pnames = pnames
        self.player_name="random_actions"
        self.debug = debug
        self.exploration_method = exploration_method
        self.nodes_array = []
        self.renderer = renderer
        if isNSteps > 0:
            self.isNSteps = True
        else:
            self.isNSteps = False
        for i in range(1, self.env.num_nodes + 1):
            self.nodes_array.append(i)

    def _exploration(self, step):
        return self.epsilon_final + (self.epsilon_start - self.epsilon_final) * np.exp(-1. * step / self.epsilon_decay)

    def _changePlayer(self, player, pid):
        rand_agent_file = "./{}".format(player)
        rand_agent_name, rand_agent_extension = os.path.splitext(rand_agent_file)
        rand_agent_mod = importlib.import_module(
            rand_agent_name.replace('./', 'agents.'))
        rand_agent_class = getattr(
            rand_agent_mod, os.path.basename(rand_agent_name))
        rand_player = rand_agent_class(self.env.num_actions_per_turn, 0)
        self.players[pid] = rand_player
        self.pnames[pid] = rand_player.__class__.__name__
    
    def _get_random(self, obs):
        rand_agent_file = "./random_actions"
        rand_agent_name, rand_agent_extension = os.path.splitext(rand_agent_file)
        rand_agent_mod = importlib.import_module(
            rand_agent_name.replace('./', 'agents.'))
        rand_agent_class = getattr(
            rand_agent_mod, os.path.basename(rand_agent_name))
        rand_player = rand_agent_class(self.env.num_actions_per_turn, 0)
        return rand_player.get_action(obs)
    def get_num_wins(self, all_reward):
        sums = 0
        for i in all_reward:
            if i == 1:
                sums += 1
        return sums

    def loop(self):
        player_list = {
            'random_actions': 1, 
            'base_rushV1': 0,
            'Cycle_BRush_Turn25': 0, 
            'Cycle_BRush_Turn50': 0,
            'Cycle_Target_Node': 0,
            'cycle_targetedNode1': 0,
            'cycle_targetedNode11': 0,
            'cycle_targetedNode11P2': 0,
            'same_commands': 0,
            'SwarmAgent': 0
            }
        plist = []

        for p in list(player_list.keys()):
            for i in range(0, player_list[p]):
                plist.append(p)

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
        all_reward = []
        highest_winrate = 0
        w = tensorboard.SummaryWriter()
        time = datetime.now().strftime('%Y%m%d_%H%M%S')
        path = './runs/{}/'.format(self.output_dir)
        try:
            os.makedirs(path)
        except:
            pass
        
        total_turn_played = 0
        turn_played_by_network = 0
        
        for step in range(self.max_steps):
            epsilon = self._exploration(step)
            self.renderer.render(state)
            # print(epsilon)
            
            action_idx = []
            action = {}
            total_turn_played += 1
            for pid in self.players:
                if pid == self.player_num:
                    if self.exploration_method == "Noisy" or np.random.random_sample() > epsilon:
                        action[pid] = self.model.get_action(state[pid])
                        turn_played_by_network += 1
                        
                    else:
                        
                        legal_moves = self.player_helper.legal_moves(state[pid])
                        actions_final = self._get_random(state[pid])

                        for i in range(len(actions_final)):
                            compute_idx = actions_final[i][0]*11 + (actions_final[i][1] - 1)
                            compute_idx = compute_idx.astype(int)
                            if legal_moves[compute_idx] == False:
                                actions_final[i] = [0,0]
                        
                        action[pid] = actions_final
                        
                else:
                    action[pid] = self.players[pid].get_action(state[pid])
            #print(action)
            next_state, reward, done, scores = self.env.step(action)
            
            other_player_id = 0
            if self.player_num == 0:
                other_player_id = 1
            if done:
                if len(all_reward) > 100:
                    all_reward.pop(0)
                all_reward.append(reward[self.player_num])
                for pid in self.players:
                    if pid != self.player_num:
                        #print(plist)
                        self.player_name = random.choice(plist)
                        self._changePlayer(self.player_name, pid)
                        print("Training with {}".format(self.player_name))
                
                if self.isNSteps:
                    self.model.finish_nstep(self.memory)
                next_state = self.env.reset(
                    players=self.players,
                    config_dir=self.config_dir,
                    map_file=self.map_file,
                    unit_file=self.unit_file,
                    output_dir=self.env_output_dir,
                    pnames=self.pnames,
                    debug=self.debug
                )
                print("Result on game {}: {}. Number of moves made by the network: {}/{}. Agents: {}".format(
                    len(all_winrate), reward, turn_played_by_network, total_turn_played, self.player_name))
                # if reward[self.player_num] == 1:
                #     reward[self.player_num] = scores[self.player_num] + 3001
                # else:
                #     reward[self.player_num] = scores[self.player_num] - scores[other_player_id] - 3001
                total_games_played += 1
                num_of_wins = self.get_num_wins(all_reward)
                episode_winrate = (num_of_wins/len(all_reward)) * 100
        
                all_winrate.append(episode_winrate)
                with open(os.path.join(path, "rewards-{}.txt".format(time)), 'a') as fout:
                    fout.write("Winrate last 100: {}. Number of moves made by the network: {}/{}. Agents: {}\n".format(episode_winrate, turn_played_by_network, total_turn_played, self.player_name))
                print("Current winrate last 100: {}%".format(episode_winrate))
                w.add_scalar("winrate",
                             episode_winrate, global_step=len(all_winrate))
                turn_played_by_network = 0
                total_turn_played = 0
                if episode_winrate > highest_winrate:
                    highest_winrate = episode_winrate
                    save_best(self.model, all_winrate,
                              "Evergaldes", self.output_dir)
            # else:
            #     reward[self.player_num] = scores[self.player_num] - scores[other_player_id]
            if self.isNSteps:
                self.model.append_to_replay(self.memory,
                    state[self.player_num],
                    action[self.player_num],
                    reward[self.player_num],
                    next_state[self.player_num],
                    done
                )
            else:
                self.memory.add(
                    state[self.player_num],
                    action[self.player_num],
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
