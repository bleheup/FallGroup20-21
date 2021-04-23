import os
import sys
import numpy as np
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)
import tensorflow.keras as keras
from tensorflow.keras.layers import Input, Dense
import tensorflow_probability as tfp
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
# Append custom module paths
MY_MODULE_PATH = 'C:/Users/aperr/Desktop/Senior Design/Newton Test/'  #'/lustre/fs0/home/cop4935.student2/everglades/modules/'
sys.path.append(MY_MODULE_PATH + 'gym-everglades')
sys.path.append(MY_MODULE_PATH + 'everglades-server')
sys.path.append(MY_MODULE_PATH + 'agents')
import gym
import gym_everglades
from everglades_server import server
from player import Player

# False/0 = No debug output; True/1 = Basic debug output; 2 = Lots of debug output
DEBUG = 0

# OPTIONAL: set default float precision
#tf.keras.backend.set_floatx('float64')  

NUM_BOTS = 12
NUM_NODES = 11
NUM_ACTION_COMBOS = NUM_BOTS * NUM_NODES  # In Everglades, our "action" consists of a selected bot and the node it will move to.
                                          # This equates to 12*11 = 132 action combinations.
                                          # Note we will only choose 7 of these, and each choice must have a different selected group as we can't move the same group more than once per step.
NUM_ACTIONS = 7  # Number of actions (action space); should be manually passed when instantiating agent
                 # 7 is Everglades' action space, since we can only make 7 moves per step
INPUT_SPACE = 249  # Input dimensions (observation space); should be manually passed when instantiating agent
                   # 249 is Everglades' observation space after one-hot-encoding

BATCH_SIZE = 64

MODEL_VERSION = '2.0.0'
CHECKPOINT_PATH = 'C:/Users/aperr/Desktop/Senior Design/SAC/models/' + MODEL_VERSION + '/'  # TODO: DEPRECATED -- Remove later!
PREV_MODEL_PATH = 'C:/Users/aperr/Desktop/Senior Design/SAC/models/' + MODEL_VERSION + '/'


class ReplayBuffer:
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_counter = 0
        self.state_memory = np.zeros( (self.mem_size, *input_shape) )
        self.new_state_memory = np.zeros( (self.mem_size, *input_shape) )
        self.action_memory = np.zeros( (self.mem_size, 7, 2) )
        self.reward_memory = np.zeros( (self.mem_size) )
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, done):
        i = self.mem_counter % self.mem_size

        self.state_memory[i] = state
        self.new_state_memory[i] = state_
        self.action_memory[i] = action
        self.reward_memory[i] = reward
        self.terminal_memory[i] = done

        self.mem_counter += 1

    # Uniform sampling; maybe implement PER later?
    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_counter, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]
        
        return states, actions, rewards, states_, dones


class CriticNetwork(keras.Model):
    def __init__(self, fc1_dims=BATCH_SIZE, fc2_dims=BATCH_SIZE, name='critic', n_actions=NUM_ACTION_COMBOS, cp_path=CHECKPOINT_PATH):
        super().__init__()
        cp_dir = os.path.dirname(cp_path)
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.model_name = name
        self.checkpoint_dir = cp_dir
        self.checkpoint_file = os.path.join(cp_dir, name+'_sac')

        self.fc1 = Dense(self.fc1_dims, activation='relu', name='fc1')
        self.fc2 = Dense(self.fc2_dims, activation='relu', name='fc2')
        self.q = Dense(self.n_actions, activation=None, name='q')  # Q-value for each possible action 
    
    def call(self, state):
        action_value = self.fc1( state )
        action_value = self.fc2( action_value )
        q = self.q(action_value)
        return q


class ActorNetwork(keras.Model):
    def __init__(self, noise=1e-6, fc1_dims=BATCH_SIZE, fc2_dims=BATCH_SIZE, name='actor', n_action_choices=NUM_ACTION_COMBOS, cp_path=CHECKPOINT_PATH):
        super().__init__()
        cp_dir = os.path.dirname(cp_path)
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.model_name = name
        self.checkpoint_dir = cp_dir
        self.checkpoint_file = os.path.join(cp_dir, name+'_sac')
        self.n_action_choices = tf.Variable(n_action_choices, trainable=False, name="n_action_choices")  # Track this variable for use in saving/loading
        self.noise = tf.Variable(noise, trainable=False, name="noise")  # Track this variable for use in saving/loading

        self.fc1 = Dense(self.fc1_dims, activation='relu', name='fc1')
        self.fc2 = Dense(self.fc2_dims, activation='relu', name='fc2')
        self.out = Dense(self.n_action_choices, activation='softmax', name='out')

    # Pass in state thru first two layers of our network and store in prob
    # Then use prob to return probability distribution tensor
    def call(self, state):
        probs1 = self.fc1(state)
        probs2 = self.fc2(probs1)
        probs_out = self.out(probs2)

        min_prob = self.noise
        max_prob = 1 - self.noise*tf.cast(self.n_action_choices, tf.float32)
        return tf.clip_by_value( probs_out, min_prob, max_prob )  # This clipping leads to improper probability distributions (AKA sum != 1), but avoids nan results

    # Sample the categorical action distribution outputted by the policy network
    @tf.function( input_signature=[tf.TensorSpec(shape=(None,INPUT_SPACE), dtype=tf.float32)] )  # Define tf.function signature to use when saving/loading model
    def sample_action_dist(self, state):
        probs = self(state)
        log_probs = tf.math.log( probs )

        # This agent does not use the returned actions for loss/cost calculation,
        # so we don't have to worry about numpy functions not being in scope of the tf computation graph
        # (Not even sure if it would matter anyways since it's just a probability sample, but good to know anyways)

        # TODO: FIGURE OUT HOW TO CONVERT PROBS TO NP ARRAY FOR SAMPLING WITHOUT REPLACEMENT
        #tf.print("probs numpy:", probs)
        #np_action = np.random.choice(probs, replace=False)  #, size=(7,)
        #tf.print("action taken:", np_action)
        #action_tensor = tf.convert_to_tensor(np_action)
        #tf.print("action_tensor:", action_tensor)

        # Action sampling
        prob_dist = tfp.distributions.Categorical(probs=probs)
        action = prob_dist.sample(NUM_ACTIONS)
        log_prob = prob_dist.log_prob(action)

        # Returns sampled action, log prob of that sampled action, probabilities of all actions, and log probs of all actions
        log_prob = []  # unused
        return action, log_prob, probs, log_probs


class AgentDiscreteSAC(Player):
    # Two learning rates: alpha for actor network, beta for critic network
    def __init__(self, lr_alpha=0.0003, lr_beta=0.0015, input_dims=[INPUT_SPACE], n_actions=NUM_ACTIONS, env=None, gamma=0.99, max_size=1000000, tau=0.0025,
                 layer1_size=BATCH_SIZE, layer2_size=BATCH_SIZE, batch_size=BATCH_SIZE, temperature=0.95, target_entropy=0.98*(np.log(NUM_ACTIONS)),
                 action_space=None, player_num=None, map_name=None):
        super().__init__(action_space, player_num, map_name)
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.env = env
        
        self.alpha = temperature
        self.target_entropy = target_entropy  # Temperature not yet implemented, so ignore this for now

        if load_prev_model:
            self.load(filepath=PREV_MODEL_PATH)
        else:
            self.critic_1 = CriticNetwork(name='critic_1')
            self.critic_2 = CriticNetwork(name='critic_2')
            self.critic_target_1 = CriticNetwork(name='critic_target_1')
            self.critic_target_2 = CriticNetwork(name='critic_target_2')
            self.actor = ActorNetwork(name='actor')

            self.actor.compile( optimizer=Adam(lr=lr_alpha) )
            self.critic_1.compile( optimizer=Adam(lr=lr_beta) )
            self.critic_2.compile( optimizer=Adam(lr=lr_beta) )
            self.critic_target_1.set_weights(self.critic_1.get_weights())
            self.critic_target_2.set_weights(self.critic_2.get_weights())

        self.learn_counter = 0
        self.is_learning = False

    def choose_action(self, observation):
        state = tf.convert_to_tensor([observation], dtype=tf.float32)
        if self.is_learning:
            # Sample action-combos
            action_combos, _, _, _ = self.actor.sample_action_dist( tf.squeeze(state, axis=0) )  # don't care about log probs at this point

            # Convert action-combos to valid [group, node] actions
            action = action_combo_to_arr(action_combos.numpy())
            action = np.concatenate( (action[0], action[1]), axis=1 )
        else:
            action = self.env.action_space.sample()

        return action

    def remember(self, state, action, reward, new_state, done):
        # We don't want to directly access values of the memory class from Agent class
        # So, we use this as an interface function
        self.memory.store_transition(state, action, reward, new_state, done)

    def update_target_network_parameters(self, tau=None):
        # When we call with tau=1, we are facilitating hard network weight copy
        if tau is None:
            # If None, use default value for tau
            tau = self.tau
        weights_1 = []
        weights_2 = []
        targets_1 = self.critic_target_1.weights
        targets_2 = self.critic_target_2.weights
        for i, weight in enumerate(self.critic_1.weights):
            weights_1.append(weight*tau + targets_1[i]*(1-tau))
        for i, weight in enumerate(self.critic_2.weights):
            weights_2.append(weight*tau + targets_2[i]*(1-tau))
        
        self.critic_target_1.set_weights(weights_1)
        self.critic_target_2.set_weights(weights_2)

    def learn(self, n_steps=1):
        # If we haven't filled up an entire batch_size worth of memories, just return
        if (self.memory.mem_counter < self.batch_size):
            self.learn_counter += 1
            return

        if self.is_learning == False:
            self.is_learning = True

        for _ in range(n_steps):
            state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)
            # state shape = (BATCH_SIZE, 249)
            # action shape = (BATCH_SIZE, 7, 2)  # each action is a list of 7 [group, node] lists
            # reward shape = (BATCH_SIZE,)
            # new_state shape = (BATCH_SIZE, 249)
            # done shape = (BATCH_SIZE,)

            # Convert to tensors with specified dtype to ensure precision
            states = tf.convert_to_tensor(state, dtype=tf.float32)
            states_ = tf.convert_to_tensor(new_state, dtype=tf.float32)
            rewards = tf.convert_to_tensor(reward, dtype=tf.float32)
            actions = tf.convert_to_tensor(action, dtype=tf.int32)

            # For debugging
            critic_1_loss = None
            critic_2_loss = None
            actor_loss = None

            # -- Critic networks --
            # Use persistent=True because loss has two components
            # If this is not passed, only keeps track of stuff for the application of a single set of gradients
            # AKA, would only update 1 critic networks instead of both
            # This lets us apply gradients twice
            with tf.GradientTape(persistent=True) as tape:
                # Sample next-state action probs from memory
                _, _, next_probs, next_log_probs = self.actor.sample_action_dist(states_)

                # Get expected q-values for sampled next-states (= q_hat)
                q1_ = tf.squeeze( self.critic_target_1(states_) )
                q2_ = tf.squeeze( self.critic_target_2(states_) )
                q_ = tf.minimum(q1_, q2_)
                critic_q_hat = next_probs * (q_ - self.alpha*next_log_probs)
                critic_q_hat = tf.reduce_sum(critic_q_hat, axis=1)  # Sum estimated "entropy" (?) of all next-state actions

                q_hat = rewards + self.gamma*critic_q_hat*tf.cast(1-done, tf.float32)  # We multiply by (1-done) so we don't get next state rewards if this was the final state of an episode

                # Get q-values for the actions taken at the sampled states (= q)
                critic1_reshaped = tf.reshape( self.critic_1(states), [BATCH_SIZE, NUM_BOTS, NUM_NODES] )  # critic values shape = (64, 132) => (64, 12, 11) reshaped
                critic2_reshaped = tf.reshape( self.critic_2(states), [BATCH_SIZE, NUM_BOTS, NUM_NODES] )
                
                # Very inefficiently sum the critic values for each of the 7 actions
                # Annoyingly, tf does not allow tensor-based indexing so I have to iterate through each action sample one-by-one
                q1 = tf.zeros(shape=[BATCH_SIZE])
                q2 = tf.zeros(shape=[BATCH_SIZE])
                for action_num in range(NUM_ACTIONS):
                    group = actions[:, action_num, 0]
                    node = actions[:, action_num, 1]
                    for g, n in zip(group, node):
                        q1 += critic1_reshaped[:, g, n-1]  # Everglades nodes use 1-based indexing, so convert to 0-based to access correct index
                        q2 += critic2_reshaped[:, g, n-1]

                # Loss calculations
                critic_1_loss = keras.losses.MSE(q1, q_hat)
                critic_2_loss = keras.losses.MSE(q2, q_hat)

            # Calculate gradients
            critic_1_network_gradient = tape.gradient(critic_1_loss, self.critic_1.trainable_variables)
            critic_2_network_gradient = tape.gradient(critic_2_loss, self.critic_2.trainable_variables)
            self.critic_1.optimizer.apply_gradients( zip(critic_1_network_gradient, self.critic_1.trainable_variables) )
            self.critic_2.optimizer.apply_gradients( zip(critic_2_network_gradient, self.critic_2.trainable_variables) )

            # Update critic target network
            self.update_target_network_parameters()

            # -- Actor network --
            with tf.GradientTape() as tape:
                # Sample action probs from memory
                _, _, policy_probs, policy_log_probs = self.actor.sample_action_dist(states)

                q1_policy = self.critic_1(states)
                q2_policy = self.critic_2(states)
                q_policy = tf.minimum(q1_policy, q2_policy)

                actor_loss = policy_probs * (self.alpha*policy_log_probs - q_policy)
                actor_loss = tf.reduce_sum(actor_loss, axis=1) + 1e-6  # Sum estimated "entropy" (?) of all next-state actions
                actor_loss = tf.reduce_mean(actor_loss)

            # Gradient of our loss w/ respect to our actor's trainable variables
            actor_network_gradient = tape.gradient(actor_loss, self.actor.trainable_variables)
            self.actor.optimizer.apply_gradients( zip(actor_network_gradient, self.actor.trainable_variables) )

    # Save model state
    def save(self, filepath):
        print('... saving models (', filepath, ') ...')
        #self.actor.summary()
        self.actor.save(filepath=filepath+'actor')
        #self.critic_1.summary()
        self.critic_1.save(filepath=filepath+'critic_1')
        #self.critic_2.summary()
        self.critic_2.save(filepath=filepath+'critic_2')
        #self.critic_target_1.summary()
        self.critic_target_1.save(filepath=filepath+'critic_target_1')
        #self.critic_target_2.summary()
        self.critic_target_2.save(filepath=filepath+'critic_target_2')

    # Load model state from file directory
    def load(self, filepath):
        # load_model only loads in the info regarding the MODEL, not the python object holding that model.
        # Meaning, object vars/methods/etc are not saved when calling the Model.save() function.
        # This must be done manually with tf.Variable or @tf.function.
        print('... loading models (', filepath, ') ...')
        self.actor = load_model(filepath=filepath+'actor')
        self.critic_1 = load_model(filepath=filepath+'critic_1')
        self.critic_2 = load_model(filepath=filepath+'critic_2')
        self.critic_target_1 = load_model(filepath=filepath+'critic_target_1', compile=False)
        self.critic_target_2 = load_model(filepath=filepath+'critic_target_2', compile=False)


### MAIN ###
from datetime import datetime
import pybullet_envs
import matplotlib.pyplot as plt


# Plots score over episodes
def plot_learning_curve(x, scores, figure_file, title=None):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, scores)
    plt.plot(x, running_avg)
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.legend(['Score', 'Running Average (prev 100)'])
    if title is None:
        plt.title("Model v" + MODEL_VERSION + " performance on " + env_name)
    else:
        plt.title(title)
    plt.savefig(figure_file)
    plt.clf()

# Plots win-rate over episodes
def plot_winrate_curve(x, scores, figure_file, title=None):
    outcomes = [100*(s+1)/2 for s in scores]  # convert scores to 0-100 range
    winrate = np.zeros(len(outcomes))
    for i in range(len(winrate)):
        winrate[i] = np.mean(outcomes[max(0, i-100):(i+1)])
    plt.plot(x, winrate)
    plt.xlabel("Episode")
    plt.ylabel("Winrate")
    plt.legend(['Winrate (prev 100)'])
    if title is None:
        plt.title("Model v" + MODEL_VERSION + " performance on " + env_name)
    else:
        plt.title(title)
    plt.savefig(figure_file)
    plt.clf()

# One-hot encodes Everglades' observation space
def ohc_observation(observation):
    new_observation = np.zeros(249, dtype=int)
    for i in range(45):
        new_observation[i] = observation[i]
    _i = 0
    for j in range(45, 105, 5):
        # Translate what node location
        index_loc = int(j + 12*_i + observation[j])
        new_observation[index_loc] = 1
        # Translate what type of class
        index_class = int(j+ 12*_i + 11 + observation[j+1])
        new_observation[index_class] = 1
        # Translate average health
        index_avg_health = int(j+ 12*_i + 14)
        new_observation[index_avg_health] = observation[j+2]
        # Translate transit status
        index_transit = int(j+ 12*_i + 15)
        new_observation[index_transit] = observation[j+3]
        # Translate number of units alive
        index_num_alive = int(j+ 12*_i + 16)
        new_observation[index_num_alive] = observation[j+4]
        _i += 1
    return new_observation

# Converts 14-tuple action to [7,2] shaped array (for Everglades compatibility)
def action_tuple_to_arr(action_tuple):
    return np.asarray(action_tuple, dtype=np.float32).reshape((7, 2))

# Converts a combined group-node action (of value 0-131) to a [2] shaped array (with values [0-11, 1-11])
def action_combo_to_arr(action_combo):
    group = action_combo//NUM_NODES
    node = 1+(action_combo % NUM_NODES)  # Everglades nodes use 1-based indexing, so convert node from 0-based to 1-based for action
    return np.array([group, node])


# Run parameters
#load_checkpoint = True  # Load previous model weights/network params?  # TODO: DEPRECATED -- Remove later!
load_prev_model = False  # Load previously saved models for all of our agent's networks?
do_learn        = True  # Update network parameters?
do_save         = False  # Save updated model(s)?
do_plot         = False  # Plot model performance?
do_render       = False  # Display rendered environment?
do_save_render  = False  # Save a rendered gif of gameplay?
N_EPISODES = 1
MAX_STEPS = -1  # Set to -1 to let environment decide max steps
#LEARN_DELAY = -1  # Number of episodes before model starts sampling memory and learning  # TODO: Implement learning delay to optionally fill replay buffer with more experience before learning
SAVE_DELAY = 0  # Number of episodes before model decides to start saving
SAVE_PERIOD = 10  # Number of episodes between each periodic model save

## RENDERER ##
from everglades_renderer import Renderer
import imageio


if __name__ == '__main__':
    env_name = 'everglades-v0'
    env = gym.make(env_name)

    # Everglades parameters
    map_name = "DemoMap.json"
    config_dir = MY_MODULE_PATH + 'config/'
    map_file = config_dir + map_name
    setup_file = config_dir + 'GameSetup.json'
    unit_file = config_dir + 'UnitDefinitions.json'
    output_dir = MY_MODULE_PATH + 'game_telemetry/'
    input_shape = [249]
    n_actions = env.num_actions_per_turn
    player_num = 1
    rand_player = Player(n_actions, 0, map_name)
    dqn_player = AgentDiscreteSAC( input_dims=input_shape, env=env, n_actions=n_actions, action_space=n_actions, player_num=player_num, map_name=map_name )
    players = {
        0: rand_player,
        1: dqn_player
    }
    names = [rand_player.__class__.__name__, dqn_player.__class__.__name__]

    agent = dqn_player

    # Console output for verifying agent parameters
    print("\n===== Starting env:", env_name, "=====")
    print('action space =', env.action_space)
    print('env num_actions_per_turn =', env.num_actions_per_turn)
    print('random action sample =', env.action_space.sample(), "\n")
    print('observation space =', env.observation_space)
    print('observation space shape =', env.observation_space.shape)
    print('observation space shape[0] =', env.observation_space.shape[0])
    print('one-hot encoded observation space shape = 249', "\n")
    print('* do_learn =', do_learn)
    print('* do_save =', do_save)
    print('* do_plot =', do_plot)
    print('* do_render =', do_render)
    print('N_EPISODES =', N_EPISODES, "| MAX_STEPS =", MAX_STEPS, "| SAVE_DELAY", SAVE_DELAY, "| SAVE_PERIOD =", SAVE_PERIOD)
    print("==============================================\n")

    best_score = env.reward_range[0]
    score_history = []

    # Filename string for output file
    fig_path = 'C:/Users/aperr/Desktop/Senior Design/Newton Test/plots/' + env_name + '/'
    time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    fname = MODEL_VERSION + '_' + time_str + '.png'
    fig_file =  fig_path + fname

    ## RENDERER ##
    gif_frames = []
    if do_save_render:
        renderer = Renderer(map_file, frame_collection=True)

    # Run
    for i in range(N_EPISODES):
        observation = env.reset(
            players=players,
            config_dir=config_dir,
            map_file=map_file,
            unit_file=unit_file,
            output_dir=output_dir,
            pnames=names,
            debug=int(DEBUG > 0)
        )

        ## RENDERER ##
        if do_save_render:
            gif_frames.append(renderer.render(observation))

        observation[player_num] = ohc_observation( observation[player_num] )  # One-hot encode observation
        if do_render:
            env.render(mode="human")
        done = False
        score = 0
        steps = 0

        while not done:
            actions = {}
            for pid in players:
                if pid == player_num:
                    observation[pid] = np.reshape( observation[pid], [1, input_shape[0]] )
                    actions[pid] = action_tuple_to_arr( agent.choose_action(observation[pid]) )
                else:
                    actions[pid] = players[pid].get_action(observation[pid])
            observation_, reward, done, _ = env.step(actions)

            ## RENDERER ##
            if do_save_render:
                gif_frames.append(renderer.render(observation_))

            observation_[player_num] = ohc_observation( observation_[player_num] )  # One-hot encode new observation
            score += reward[player_num]
            agent.remember( observation[player_num], actions[player_num], reward[player_num], observation_[player_num], done )

            if do_learn:
                agent.learn()
                
            observation = observation_

            if do_render:
                env.render(mode="human")

            steps += 1

            if MAX_STEPS != -1:
                if steps > MAX_STEPS:
                    # Stop episode early
                    break

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        # Periodically save model and plot performance
        if (i+1) % SAVE_PERIOD == 0:
            if do_save and agent.is_learning:
                agent.save(filepath=PREV_MODEL_PATH + time_str + '_'+str(i+1)+'eps/')
            if do_plot:
                if not os.path.exists(fig_path):
                    os.makedirs(fig_path)
                x = [j+1 for j in range(i+1)]
                plot_winrate_curve(x, score_history, fig_file)

        # Save model
        if i > SAVE_DELAY and avg_score > best_score:
            best_score = avg_score
            if do_save and agent.is_learning:
                agent.save(filepath=PREV_MODEL_PATH)
            
        print('episode', i, '::  score = %.1f' % score, 'avg_score = %.1f' % avg_score)

    if do_plot:
        x = [i+1 for i in range(N_EPISODES)]
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)
        plot_winrate_curve(x, score_history, fig_file)

    ## RENDERER ##
    if do_save_render:
        render_save_path = 'C:/Users/aperr/Desktop/Senior Design/Newton Test/renders/'
        if not os.path.exists(render_save_path):
            os.makedirs(render_save_path)
        render_fprefix = MODEL_VERSION + '_' + time_str
        render_fig_file =  render_save_path + render_fprefix + '_{}_vs_{}.gif'.format("DiscreteSAC", "RandomAgent")
        imageio.mimsave(render_fig_file, gif_frames, duration=0.3)

