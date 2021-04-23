import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
my_module_path = '/lustre/fs0/home/cop4935.student2/everglades/modules/'
sys.path.append(my_module_path + 'gym-everglades')
sys.path.append(my_module_path + 'everglades-server')
import tensorflow as tf
import gym
import gym_everglades
from everglades_server import server
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model, load_model
import tensorflow.keras.backend as K
from player import Player
import numpy as np
from datetime import datetime
from OneHotEncode import OneHotEncode

# Opponents
from swarm_agent import SwarmAgent
from all_cycle import all_cycle
from base_rush_v1 import base_rushV1
from cycle_rush_turn25 import Cycle_BRush_Turn25
from cycle_rush_turn50 import Cycle_BRush_Turn50
from cycle_target_node11P2 import cycle_targetedNode11P2
from random_actions import random_actions
from same_commands import same_commands
from cycle_target_node import Cycle_Target_Node

class Memory:
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []
        
        self.batch_size = batch_size
        
    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]
        
        return np.array(self.states),\
                np.array(self.actions),\
                np.array(self.probs),\
                np.array(self.vals),\
                np.array(self.rewards),\
                np.array(self.dones),\
                batches
    
    def store(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)
        
    def clear(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []

class PPOAgent(Player):
    def __init__(self, action_space, player_num, map_name):
        super().__init__(action_space, player_num, map_name)
        
        # variables here
        self.EPISODES = 20000
        self.EPISODES_TEST = 1000
        self.TRAINING_BATCH_SIZE = 5
        self.test_checkpoint = 5000
        self.clipping = 0.2
        self.gamma = 0.99
        self.lmbda = 0.95
        self.critic_discount = 0.5
        
        self.n_epochs = 4
        self.samples_filled = 0
        self.enemy_transit = [0] * 12
        self.time_filename = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        self.win_list = []
        self.iteration = 0
    
    def set_init_state(self, env, players, config_dir, map_file, unit_file, output_dir, names, debug):
        self.env = env
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.action_choices.shape[0]
        self.config_dir = config_dir
        self.map_file = map_file
        self.unit_file = unit_file
        self.output_dir = output_dir
        self.names = names
        self.debug = debug
        self.players = players
        
        # Dummy values for calculations
        self.dummy_ones = np.zeros((1, 1))
        self.dummy_n = np.zeros((1, self.action_size))
        
        # Batch memorization
        self.memory = Memory(self.TRAINING_BATCH_SIZE)
        
        # Used for training
        self.advantage = None
        self.old_prediction = None
        self.returns = None
        self.critic_values = None
        self.is_training = False
        
        # Current working model
        self.actor_model = self.build_actor_model()
        self.critic_model = self.build_critic_model()
        
    def build_actor_model(self):
        state = Input(shape=(249,))
        
        dense = Dense(512, activation='relu')(state)
        dense = Dense(256, activation='relu')(dense)
        
        policy = Dense(self.action_size, activation='softmax')(dense)
        
        # Custom loss function for PPO
        def ppo_loss(y_true, y_pred):
            old_prediction = None
            advantage = None
            returns = None
            critic_values = None
            
            if self.is_training:
                old_prediction = tf.convert_to_tensor(self.old_prediction, np.float32)
                advantage = tf.convert_to_tensor(self.advantage, np.float32)
                returns = tf.convert_to_tensor(self.returns, np.float32)
                critic_values = tf.convert_to_tensor(self.critic_values, np.float32)
            else:
                old_prediction  = tf.convert_to_tensor(self.dummy_n, np.float32)
                advantage = tf.convert_to_tensor(self.dummy_ones, np.float32)
                returns = tf.convert_to_tensor(self.dummy_ones, np.float32)
                critic_values = tf.convert_to_tensor(self.dummy_ones, np.float32)
            
            prob = y_pred * y_true
            old_prob = old_prediction * y_true
            ratio = K.exp(K.log(prob + 1e-10) - K.log(old_prob + 1e-10))
            clip_ratio = K.clip(ratio, min_value = 1 - self.clipping, max_value = 1 + self.clipping)
            surrogate1 = ratio * advantage
            surrogate2 = clip_ratio * advantage
            actor_loss = -K.mean(K.minimum(surrogate1, surrogate2))
            critic_loss = K.mean(K.square(returns - critic_values))
            total_loss = self.critic_discount * critic_loss + actor_loss
            return total_loss
        
        actor_network = Model(inputs = [state], outputs=policy)
        actor_network.compile(optimizer=Adam(lr=1e-4), loss=ppo_loss)
        
        # actor_network.summary()
        return actor_network
    
    def ppo_loss(self, y_true, y_pred):
            old_prediction = None
            advantage = None
            returns = None
            critic_values = None
            
            if self.is_training:
                old_prediction = tf.convert_to_tensor(self.old_prediction, np.float32)
                advantage = tf.convert_to_tensor(self.advantage, np.float32)
                returns = tf.convert_to_tensor(self.returns, np.float32)
                critic_values = tf.convert_to_tensor(self.critic_values, np.float32)
            else:
                old_prediction  = tf.convert_to_tensor(self.dummy_n, np.float32)
                advantage = tf.convert_to_tensor(self.dummy_ones, np.float32)
                returns = tf.convert_to_tensor(self.dummy_ones, np.float32)
                critic_values = tf.convert_to_tensor(self.dummy_ones, np.float32)
            
            prob = y_pred * y_true
            old_prob = old_prediction * y_true
            ratio = K.exp(K.log(prob + 1e-10) - K.log(old_prob + 1e-10))
            clip_ratio = K.clip(ratio, min_value = 1 - self.clipping, max_value = 1 + self.clipping)
            surrogate1 = ratio * advantage
            surrogate2 = clip_ratio * advantage
            actor_loss = -K.mean(K.minimum(surrogate1, surrogate2))
            critic_loss = K.mean(K.square(returns - critic_values))
            total_loss = self.critic_discount * critic_loss + actor_loss
            return total_loss
    
    def build_critic_model(self):
        state = Input(shape=(249,))
        
        dense = Dense(512, activation='relu')(state)
        dense = Dense(256, activation='relu')(dense)
        
        V = Dense(1, activation='linear')(dense)
        
        critic_network = Model(inputs=state, outputs=V)
        critic_network.compile(optimizer=Adam(lr=1e-4),loss='mse')
        
        # critic_network.summary()
        return critic_network
    
    def new_distribution(self, legal_moves, pred):
        result = np.where(legal_moves == True)
        new_pred = np.zeros((132,))
        aux_sum = np.zeros((12,))
        new_pred[result] = pred[result]
        norm = new_pred.sum()
        
        # Fix the 0%-legal-move issue
        num_probs = np.count_nonzero(new_pred)
        if len(result[0]) > num_probs:
            if num_probs == 0:
                avg_prob = 1 / len(result[0])
            else:
                avg_prob = norm / num_probs
            for i in range(len(result[0])):
                if new_pred[result[0][i]] == 0:
                    new_pred[result[0][i]] = avg_prob
            norm = new_pred.sum()

        if len(result[0] != 0):
            new_pred = new_pred/norm

        new_pred = np.reshape(new_pred, (12, 11))
        
        for i in range(aux_sum.size):
            sum_of_array = new_pred[i].sum()
            aux_sum[i] = sum_of_array
            
            # Destructively modify choice array, must be analyzed row-by-row
            if sum_of_array > 0:
                new_pred[i] = new_pred[i]/sum_of_array
        
        legal_groups = np.count_nonzero(aux_sum)
        return new_pred, legal_groups, aux_sum
    
    def get_action(self, state):
        action = np.zeros(self.shape)
        input_state = np.reshape(OneHotEncode(state), [1, 249])
        pred = self.actor_model.predict(input_state).flatten()
        value = self.critic_model.predict(input_state).flatten()

        norm, legal_groups, aux_sum = self.new_distribution(self.legal_moves(state), pred)
        
        min_select = min(self.num_actions, legal_groups)
        choice = []
        if (legal_groups == 0):
            select = [-1] * 7
        else:
            select = np.random.choice(12, min_select, replace=False, p=aux_sum)
            
            for i in select:
                select_one = np.random.choice(11, 1, p=norm[i])
                choice.append(i*11 + select_one)
            
        for i in range(0, min_select):
            action[i] = self.action_choices[choice[i]]
            
        # Fill out select array for completeness
        for i in range(len(select), self.num_actions):
            select = np.append(select, -1)
        
        return action, select, pred, value
        
    def train_network(self):
        self.is_training = True
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr,\
            reward_arr, dones_arr, batches = \
                self.memory.generate_batches()
            
            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)
            
            for t in range(len(reward_arr)-1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr)-1):
                    a_t += discount*(reward_arr[k] + self.gamma*values[k+1]*\
                                     (1-int(dones_arr[k])) - values[k])
                    discount *= self.gamma*self.lmbda
                advantage[t] = a_t
        
            for batch in batches:
                batch_state = np.vstack(state_arr[batch])            
                batch_old_prob = np.vstack(old_prob_arr[batch])
        
                action_final = np.zeros(shape=(len(action_arr[batch]), self.action_size))
        
                # action batch has size (16, 7)
                # final action batch has size (16, 132)
                for i in range(len(action_arr[batch])):
                    for j in range(len(action_arr[batch][i])):
                        index = action_arr[batch][i][j]
                        action_final[i, index] = 1
        
                self.advantage = np.vstack(advantage[batch])
                self.old_prediction = batch_old_prob
                self.returns = advantage[batch] + values[batch]
                self.critic_values = self.get_v(batch_state)
                
                self.actor_model.fit(x=batch_state, y=action_final, verbose=0)
                self.critic_model.fit(x=batch_state, y=self.returns, verbose=0)
        
                self.advantage = None
                self.old_prediction = None
                self.returns = None
                self.critic_values = None
        self.is_training = False
    
    def store_transition(self, state, action, probs, vals, reward, done):
        self.memory.store(state, action, probs, vals, reward, done)
        
    def get_v(self, state):
        s = np.reshape(state, (-1, 249))
        v = self.critic_model.predict(s)
        return v
    
    def save(self, actor_name, critic_name):
        self.actor_model.save(actor_name)
        self.critic_model.save(critic_name)
        
    def load_models(self):
        self.actor_model = load_model('../everglades-ppo-actor-20210323_191444-Yami Bakura-e5000.h5',\
                                      custom_objects={'ppo_loss': self.ppo_loss})
        self.critic_model = load_model('../everglades-ppo-critic-20210323_191444-Yami Bakura-e5000.h5')
        
    def reward_shape(self, obs, opp_actions, scores):
        initial_reward = scores[self.player_num] - scores[1]
        
        # Unit indices
        # unit_locations = [45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
        # unit_types = [46, 51, 56, 61, 66, 71, 76, 81, 86, 91, 96, 101]
        # unit_health = [47, 52, 57, 62, 67, 72, 77, 82, 87, 92, 97, 102]
        # units_transit = [48, 53, 58, 63, 68, 73, 78, 83, 88, 93, 98, 103]
        # units_alive = [49, 54, 59, 64, 69, 74, 79, 84, 89, 94, 99, 104]
        
        location_remap = [11, 8, 9, 10, 5, 6, 7, 2, 3, 4, 1]
        
        # Node indices
        # Mild danger [12, 20, 24, 28]
        # Severe danger [8, 16]
        # Critical danger [4]
        
        nodes = [0] * 11
        node_marked = [False] * 7
        
        # Total enemy units left
        enemy_unit_count = 0
        j = 0
        for i in range(4, 48, 4):
            enemy_unit_count += obs[0][i]
            nodes[j] += obs[0][i]
            j += 1
        
        # Update enemy unit locations based on transits
        # Iterate through action array
        for i in range(7):
            group = int(opp_actions[i][0])
            # Was the move legal (Did it provoke a transit and wasn't already in transit? Also, must be alive)
            if (obs[1][48+(5*group)] == 1 and self.enemy_transit[group] == 0 and obs[1][47+(5*group)] != 0):
                self.enemy_transit[int(opp_actions[i][0])] = int(opp_actions[i][1])
        
        # Iterate through enemy groups
        for i in range(12):
            # Are the units in transit and alive
            if (obs[1][48+(5*i)] == 1 and obs[1][47+(5*group)] != 0):
                # Take units out of where they're "located" and put it where their destination is
                nodes[location_remap[int(obs[1][45+(5*i)])-1]-1] -= obs[1][49+(5*i)]
                nodes[location_remap[self.enemy_transit[i]-1]-1] += obs[1][49+(5*i)]
            else:
                # Keep units where they are and make sure transit index is 0
                self.enemy_transit[i] = 0
        
        # Subtract number of friendly units at defensive node locations
        for i in range(7):
            
            # Sneak in a check for where enemy units are located near the base
            if (nodes[i] > 0):
                node_marked[i] = True
                
            # Not moving and not dead (and are within defensive range)
            if ((obs[0][48+(5*i)] != 1 and obs[0][49+(5*i)] != 0) and obs[0][45+(5*i)] < 8):
                if (obs[0][46+(5*i)] == 1 or obs[0][46+(5*i)] == 2):
                    nodes[int(obs[0][45+(5*i)])-1] -= obs[0][49+(5*i)] * (obs[0][47+(5*i)] * 0.01)
                else:
                    # Having controller try to defend is deemed "not very valuable"
                    nodes[int(obs[0][45+(5*i)])-1] -= (obs[0][49+(5*i)] * (obs[0][47+(5*i)] * 0.01) * 0.5)
        
        # Give weighted sum of differences of nodes ONLY at locations where enemy units were located
        sum_of_unit_difference = 0.0
        for i in range(7):
            if node_marked[i] == True:
                sum_of_unit_difference += abs(nodes[i])
        
        return initial_reward - (abs(initial_reward) * (sum_of_unit_difference/enemy_unit_count))
    
    def run(self, opponent_list, opponent_distribution, opponent_name):
        for e in range(self.EPISODES):
        
            # Testing
            if ((e+1) % self.test_checkpoint == 0):
                save_dir = '../models/PPOTrial/'
                actor_name = save_dir + 'everglades-ppo-actor-{1}-{0}-e{2}.h5'.format(opponent_name, self.time_filename, (e+1))
                critic_name = save_dir + 'everglades-ppo-critic-{1}-{0}-e{2}.h5'.format(opponent_name, self.time_filename, (e+1))
                self.save(actor_name, critic_name)
                
                if (self.test(opponent_list, opponent_distribution, False, opponent_name) >= 75):
                    # End Session with opponent
                    break
            
            opponent = np.random.choice(opponent_list, 1, p=opponent_distribution)
            if (isinstance(opponent[0], list)):
                opponent = np.random.choice(opponent[0], 1)
            self.players[1] = opponent[0]
            self.names[1] = opponent[0].__class__.__name__
            
            state = env.reset(
                players=self.players,
                config_dir=self.config_dir,
                map_file=self.map_file,
                unit_file=self.unit_file,
                output_dir=self.output_dir,
                pnames=self.names,
                debug=self.debug
            )
            
            done = False
            
            while not done:
                agent_move = None
                actions = {}
                for pid in self.players:
                    if pid == self.player_num:
                        # agent makes move
                        agent_move = self.get_action(state[pid])
                                
                        actions[pid] = agent_move[0]
                    else:
                        actions[pid] = self.players[pid].get_action(state[pid])

                s_, r, done, scores = env.step(actions)
                
                if r[self.player_num] == 1:
                    r[self.player_num] += 3000
                elif r[self.player_num] == -1:
                    r[self.player_num] -= 3000
                    
                # Reward shaping
                r[self.player_num] += self.reward_shape(s_, actions[1], scores)

                # Save only the states and rewards that the agent can see
                encoded_state = np.reshape(OneHotEncode(state[self.player_num]), [1, 249])
                self.store_transition(encoded_state, agent_move[1], agent_move[2], agent_move[3], r[self.player_num], done)
                self.samples_filled += 1
                
                if self.samples_filled % 20 == 0 and self.samples_filled != 0:
                    self.train_network()
                    self.memory.clear()
                    self.samples_filled = 0
                
                state = s_
                if done:
                    self.enemy_transit = [0] * 12
         
    def test(self, opponent_list, opponent_distribution, final_test, opponent_name):
        
        # Winrate
        win = 0
        
        for e in range(self.EPISODES_TEST):
            
            if final_test:
                opponent = np.random.choice(opponent_list, 1)
            else:
                opponent = np.random.choice(opponent_list, 1, p=opponent_distribution)
            if (isinstance(opponent[0], list)):
                opponent = np.random.choice(opponent[0], 1)
            self.players[1] = opponent[0]
            self.names[1] = opponent[0].__class__.__name__
            
            state = env.reset(
                players=self.players,
                config_dir=self.config_dir,
                map_file=self.map_file,
                unit_file=self.unit_file,
                output_dir=self.output_dir,
                pnames=self.names,
                debug=self.debug
            )
            
            done = False
            
            while not done:
                agent_move = None
                
                actions = {}
                for pid in self.players:
                    if pid == self.player_num:
                        agent_move = self.get_action(state[pid])
                        
                        actions[pid] = agent_move[0]
                    else:
                        actions[pid] = self.players[pid].get_action(state[pid])
                
                s_, r, done, scores = env.step(actions)
                state = s_
                
                if done:
                    if r[self.player_num] == 1:
                        win += 1
                    time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    self.win_list.append(int(100 * win/(e+1)))
                    
                    winpercent = int(100 * win/(e+1))
                    with open('{}_test_status.txt'.format(opponent_name), 'a') as fout:
                        fout.write(time + ' | episode: {}/{}, win: {} ({}%)\n'.format(e+1, self.EPISODES_TEST, win, winpercent))
        
        with open('{}_final_status_{}.txt'.format(opponent_name, self.iteration), 'w') as fout:
            fout.write(','.join(map(str, self.win_list)))
        self.iteration += 1
        self.win_list = []
        return int(100 * win/(e+1))
                
if __name__ == "__main__":
    map_name = "DemoMap.json"
    
    config_dir = '../../../config/'
    map_file = config_dir + map_name
    setup_file = config_dir + 'GameSetup.json'
    unit_file = config_dir + 'UnitDefinitions.json'
    output_dir = '../../../game_telemetry/'
    
    debug = 1
    
    env = gym.make('everglades-v0')
    players = {}
    names = {}
    
    # Reinforcement Learning Agent
    Popo = PPOAgent(env.num_actions_per_turn, 0, map_name)
    
    # Opponents
    Raptor = random_actions(env.num_actions_per_turn, 1)
    Tristan = SwarmAgent(env.num_actions_per_turn, 1)
    Duke = same_commands(env.num_actions_per_turn, 1)
    Yami_Bakura = all_cycle(env.num_actions_per_turn, 1)
    Pegasus = [Cycle_BRush_Turn25(env.num_actions_per_turn, 1), Cycle_BRush_Turn50(env.num_actions_per_turn, 1)]
    Yugi = base_rushV1(env.num_actions_per_turn, 1)
    Kaiba = [cycle_targetedNode11P2(env.num_actions_per_turn, 1), Cycle_Target_Node(env.num_actions_per_turn, 1)]
    
    opponents = [Raptor, Tristan, Duke, Yami_Bakura, Pegasus, Yugi, Kaiba]
    opponent_probability = [
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.05, 0.95, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.05, 0.05, 0.9, 0.0, 0.0, 0.0, 0.0],
        [0.05, 0.05, 0.05, 0.85, 0.0, 0.0, 0.0],
        [0.05, 0.05, 0.05, 0.05, 0.8, 0.0, 0.0],
        [0.05, 0.05, 0.05, 0.05, 0.05, 0.75, 0.0],
        [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.7]
    ]
    opponent_names = ['Raptor', 'Tristan', 'Duke', 'Yami Bakura', 'Pegasus', 'Yugi', 'Kaiba']
    
    players[0] = Popo
    names[0] = Popo.__class__.__name__
    players[1] = opponents[0]
    names[1] = opponents[0].__class__.__name__
    
    Popo.set_init_state(env, players,
                              config_dir, map_file, unit_file, output_dir, names, debug)
    # Popo.load_models()
    
    for i in range(1, 7):
        print('New opponent! Enter:', opponent_names[i])
        Popo.run(opponents[0:i+1], opponent_probability[i][0:i+1], opponent_names[i])
    
    Popo.test(opponents[1:], None, True, 'All')
