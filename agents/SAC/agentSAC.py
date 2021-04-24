import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense
import tensorflow_probability as tfp
from tensorflow.keras.optimizers import Adam

# False/0 = No debug output; True/1 = Basic debug output; 2 = Lots of debug output
DEBUG = 0

# OPTIONAL: set default float precision
#tf.keras.backend.set_floatx('float64')  

#NUM_BOT_ACTIONS = 7
#BOT_ACTION_SPACE = 2
#NUM_ACTIONS = NUM_BOT_ACTIONS * BOT_ACTION_SPACE
INPUT_SPACE = 8  # Input dimensions (action space); should be set when instantiating agent


class ReplayBuffer:
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_counter = 0
        self.state_memory = np.zeros( (self.mem_size, *input_shape) )
        self.new_state_memory = np.zeros( (self.mem_size, *input_shape) )
        # Everglades uses a discrete action space, not continuous, so action_memory might need modification to work
        print("ReplayBuffer: n_actions =", n_actions) if DEBUG else None
        print("ReplayBuffer: input_shape =", input_shape) if DEBUG else None
        self.action_memory = np.zeros( (self.mem_size, n_actions) )  #*input_shape) )  #NUM_BOT_ACTIONS, BOT_ACTION_SPACE) )  # continuous: (self.mem_size, n_actions)  
        print("ReplayBuffer: action_memory.shape =", self.action_memory.shape) if DEBUG else None
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
    def __init__(self, n_actions, fc1_dims=256, fc2_dims=256, name='critic', cp_path='tmp/sac'):
        super().__init__()
        cp_dir = os.path.dirname(cp_path)
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.model_name = name
        self.checkpoint_dir = cp_dir
        self.checkpoint_file = os.path.join(cp_dir, name+'_sac')

        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.q = Dense(1, activation=None)
    
    def call(self, state, action):
        print("== Critic Network ==") if DEBUG else None
        print("state.shape =", state.shape) if DEBUG == 2 else None
        print("first 5 state =", state[:5]) if DEBUG == 2 else None
        print("action.shape =", action.shape) if DEBUG == 2 else None
        print("first 5 action =", action[:5]) if DEBUG == 2 else None
        state_action_pair = tf.concat([state, action], 1)
        print("state_action_pair.shape =", tf.concat([state, action], 1).shape) if DEBUG == 2 else None
        print("first 5 state_action_pair =", state_action_pair[:5]) if DEBUG == 2 else None
        action_value = self.fc1(state_action_pair)
        print("1st action_value.shape =", action_value.shape) if DEBUG == 2 else None
        action_value = self.fc2( action_value )
        print("2nd action_value.shape =", action_value.shape) if DEBUG == 2 else None
        q = self.q(action_value)
        print("q: max/min =", str(np.max(q)) + "/" + str(np.min(q)), "| med =", np.median(q), "| mean =", np.mean(q), " ||  q.shape", q.shape) if DEBUG else None
        return q
    

class ValueNetwork(keras.Model):
    def __init__(self, fc1_dims=256, fc2_dims=256, name='value', cp_path='tmp/sac'):
        super().__init__()
        cp_dir = os.path.dirname(cp_path)
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.model_name = name
        self.checkpoint_dir = cp_dir
        self.checkpoint_file = os.path.join(cp_dir, name+'_sac')

        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.v = Dense(1, activation=None)
    
    # Value function only cares about current value of state, not state-action pairs
    def call(self, state):
        state_value = self.fc1(state)
        state_value = self.fc2(state_value)
        v = self.v(state_value)
        return v
    

class ActorNetwork(keras.Model):
    def __init__(self, max_action, noise=1e-6, fc1_dims=256, fc2_dims=256, name='actor', n_actions=2, cp_path='tmp/sac'):
        super().__init__()
        cp_dir = os.path.dirname(cp_path)
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.model_name = name
        self.checkpoint_dir = cp_dir
        self.checkpoint_file = os.path.join(cp_dir, name+'_sac')
        self.max_action = max_action
        self.noise = noise

        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.mu = Dense(self.n_actions, activation=None)
        self.sigma = Dense(self.n_actions, activation=None)

    # Pass in state thru first two layers of our network and store in prob
    # Then use prob to return mean and std for our distribution
    def call(self, state):
        prob = self.fc1(state)
        prob = self.fc2(prob)
        mu = self.mu(prob)
        sigma = self.sigma(prob)  # std of distribution
        sigma = tf.clip_by_value(sigma, self.noise, 1)  # clip std from 1e-6 to 1
        return mu, sigma
    
    # Sample the distribution returned above to get the next action for agent
    # No reparameterization (implement if necessary)
    def sample_normal(self, state):
        mu, sigma = self.call(state)
        probs = tfp.distributions.Normal(mu, sigma)

        actions = probs.sample()  # include reparam here if implemented
        log_probs = probs.log_prob(actions)

        action = tf.math.tanh(actions) * self.max_action
        log_probs -= tf.math.log(1 - tf.math.pow(action, 2) + self.noise)
        log_probs = tf.math.reduce_sum(log_probs, axis=1, keepdims=True)

        return action, log_probs
    

# NOTE: 132 actions. input space= 105 or 261


class Agent:
    # Two learning rates: alpha for actor network, beta for value/critic network
    def __init__(self, alpha=0.0003, beta=0.0003, input_dims=[INPUT_SPACE], env=None, gamma=0.99, n_actions=2, max_size=1000000, tau=0.005,
                 layer1_size=256, layer2_size=256, batch_size=256, reward_scale=2):
        self.gamma = gamma
        self.tau = tau
        print("Agent: input_dims =", input_dims) if DEBUG else None
        print("Agent: n_actions =", n_actions) if DEBUG else None
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions

        self.actor = ActorNetwork(n_actions=n_actions, name='actor', max_action=env.action_space.high)
        self.critic_1 = CriticNetwork(n_actions=n_actions, name='critic_1')
        self.critic_2 = CriticNetwork(n_actions=n_actions, name='critic_2')
        self.value = ValueNetwork(name='value')
        self.target_value = ValueNetwork(name='target_value')

        self.actor.compile( optimizer=Adam(lr=alpha) )
        self.critic_1.compile( optimizer=Adam(lr=beta) )
        self.critic_2.compile( optimizer=Adam(lr=beta) )
        self.value.compile( optimizer=Adam(lr=beta) )
        # Target value network does no optimization, copies weights with soft-update rule. Necessary for model to compile.
        self.target_value.compile( optimizer=Adam(lr=beta) )

        self.scale = reward_scale
        self.update_network_parameters(tau=1)  # Hard copy of params from online (?) network to target value network

        ## DEBUGGING ##
        self.learn_counter = 0
        self.is_learning = False

    def choose_action(self, observation):
        state = tf.convert_to_tensor([observation])
        actions, _ = self.actor.sample_normal(state) # don't care about log probs at this point

        return actions[0]   # return np array from tensor

    def remember(self, state, action, reward, new_state, done):
        # We don't want to directly access values of the memory class from Agent class
        # So, we use this as an interface function
        self.memory.store_transition(state, action, reward, new_state, done)

    def update_network_parameters(self, tau=None):
        # When we call with tau=1, we are facilitating hard network weight copy
        if tau is None:
            # If None, use default value for tau
            tau = self.tau
        weights = []
        targets = self.target_value.weights
        for i, weight in enumerate(self.value.weights):
            weights.append(weight*tau + targets[i]*(1-tau))
        
        self.target_value.set_weights(weights)

    def save_models(self):
        print('... saving models ...')
        self.actor.save_weights(self.actor.checkpoint_file)
        self.critic_1.save_weights(self.critic_1.checkpoint_file)
        self.critic_2.save_weights(self.critic_2.checkpoint_file)
        self.value.save_weights(self.value.checkpoint_file)
        self.target_value.save_weights(self.target_value.checkpoint_file)

    def load_models(self):
        print('... loading models ...')
        self.actor.load_weights(self.actor.checkpoint_file)
        self.critic_1.load_weights(self.critic_1.checkpoint_file)
        self.critic_2.load_weights(self.critic_2.checkpoint_file)
        self.value.load_weights(self.value.checkpoint_file)
        self.target_value.load_weights(self.target_value.checkpoint_file)

    def learn(self):
        # If we haven't filled up an entire batch_size worth of memories, just return
        if (self.memory.mem_counter < self.batch_size):
            self.learn_counter += 1
            return

        if self.is_learning == False:
            self.is_learning = True
            print("Took", self.learn_counter, "steps to start learning.") if DEBUG else None

        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

        # Convert to tensors with specified dtype to ensure precision
        states = tf.convert_to_tensor(state)#, dtype=tf.float32)
        states_ = tf.convert_to_tensor(new_state)#, dtype=tf.float32)
        rewards = tf.convert_to_tensor(reward)#, dtype=tf.float32)
        actions = tf.convert_to_tensor(action)#, dtype=tf.float32)

        # -- Value network -- Perform gradient descent using GradientTape
        with tf.GradientTape() as tape:
            print("\n==== Value Network Gradient ====") if DEBUG else None
            value = tf.squeeze(self.value(states), 1)  # Squeeze states to remove batch dimenstionality
            value_ = tf.squeeze(self.target_value(states_))  # Same for new states, now by passing target_value network

            current_policy_actions, log_probs = self.actor.sample_normal(states)
            log_probs = tf.squeeze(log_probs, 1)
            q1_new_policy = self.critic_1(states, current_policy_actions)
            q2_new_policy = self.critic_2(states, current_policy_actions)

            critic_value = tf.squeeze( tf.math.minimum(q1_new_policy, q2_new_policy), 1 )

            value_target = critic_value - log_probs  # from paper
            value_loss = 0.5 * keras.losses.MSE(value, value_target)

        print("Value network gradient...") if DEBUG else None
        value_network_gradient = tape.gradient(value_loss, self.value.trainable_variables)
        print("... calculated") if DEBUG else None
        self.value.optimizer.apply_gradients( zip(value_network_gradient, self.value.trainable_variables) )
        print("... gradients applied!") if DEBUG else None

        # - Actor network -
        with tf.GradientTape() as tape:
            # In the paper, they reparameterize here. We don't implement this so it's just the usual action.
            print("\n==== Actor Network Gradient ====") if DEBUG else None
            new_policy_actions, log_probs = self.actor.sample_normal(states)
            log_probs = tf.squeeze(log_probs, 1)
            q1_new_policy = self.critic_1(states, new_policy_actions)
            q2_new_policy = self.critic_2(states, new_policy_actions)

            critic_value = tf.squeeze( tf.math.minimum(q1_new_policy, q2_new_policy), 1 )

            actor_loss = log_probs - critic_value
            actor_loss = tf.math.reduce_mean(actor_loss)

        # Gradient of our loss w/ respect to our actor's trainable variables
        print("Actor network gradient...") if DEBUG else None
        print("Num trainable variables =", len(self.actor.trainable_variables)) if DEBUG else None
        for var in self.actor.trainable_variables:
            print(":: '{}' = {}".format(var.name, var[:5])) if DEBUG == 2 else None
        actor_network_gradient = tape.gradient(actor_loss, self.actor.trainable_variables)
        print("... calculated") if DEBUG else None
        self.actor.optimizer.apply_gradients( zip(actor_network_gradient, self.actor.trainable_variables) )
        print("... gradients applied!") if DEBUG else None

        # Critic networks
        # Use persistent=True because loss has two components
        # If this is not passed, only keeps track of stuff for the application of a single set of gradients
        # AKA, would only update 1 critic networks instead of both
        # This lets us apply gradients twice
        with tf.GradientTape(persistent=True) as tape:
            q_hat = self.scale*reward + self.gamma*value_*(1-done)  # value_ is out of scope but seems to work?

            # Sample actions from memory (hence the name "old policy")
            print("\n==== Critic Network Gradient ====") if DEBUG else None
            print("CNG: state.shape", state.shape) if DEBUG == 2 else None
            print("CNG: action.shape", action.shape) if DEBUG == 2 else None
            print("CNG: first 5 critic_1", self.critic_1(state, action)[:5]) if DEBUG == 2 else None
            print("CNG: first 5 critic_2", self.critic_2(state, action)[:5], "\n") if DEBUG == 2 else None
            q1_old_policy = tf.squeeze( self.critic_1(state, action), 1 )
            q2_old_policy = tf.squeeze( self.critic_2(state, action), 1 )

            critic_1_loss = 0.5 * keras.losses.MSE(q1_old_policy, q_hat)
            critic_2_loss = 0.5 * keras.losses.MSE(q2_old_policy, q_hat)

        # Calculate gradients
        print("Critic network gradients...") if DEBUG else None
        critic_1_network_gradient = tape.gradient(critic_1_loss, self.critic_1.trainable_variables)
        print("... 1") if DEBUG == 2 else None
        critic_2_network_gradient = tape.gradient(critic_2_loss, self.critic_2.trainable_variables)
        print("... 2") if DEBUG == 2 else None
        self.critic_1.optimizer.apply_gradients( zip(critic_1_network_gradient, self.critic_1.trainable_variables) )
        print("... 1st gradients applied!") if DEBUG == 2 else None
        self.critic_2.optimizer.apply_gradients( zip(critic_2_network_gradient, self.critic_2.trainable_variables) )
        print("... 2nd gradients applied!") if DEBUG == 2 else None

        self.update_network_parameters()
        print("Network params updated.") if DEBUG else None


### MAIN 
from datetime import datetime
import gym
import pybullet_envs
import matplotlib.pyplot as plt

N_EPISODES = 100
#MAX_STEPS = 1000

def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running avg of prev 100 scores')
    plt.savefig(figure_file)

if __name__ == '__main__':
    env_name = 'BipedalWalker-v3'
    env = gym.make(env_name)
    print("\n===== Starting env:", env_name, "=====")
    print('action space = ', env.action_space)
    print('action space shape = ', env.action_space.shape)
    print('*** action space shape[0] = ', env.action_space.shape[0], "\n")
    print('observation space = ', env.observation_space)
    print('*** observation space shape = ', env.observation_space.shape)
    print('observation space shape[0] = ', env.observation_space.shape[0], "\n")
    agent = Agent( input_dims=env.observation_space.shape, env=env, n_actions=env.action_space.shape[0] )

    # Filename string for output file
    time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    fname = env_name + '_' + time_str + '.png'
    fig_file = 'C:/Users/aperr/Desktop/Senior Design/CartPole SAC/plots/' + fname

    best_score = env.reward_range[0]
    score_history = []

    load_checkpoint = False
    if load_checkpoint:
        agent.load_models()
        env.render(mode='human')  # (???) Use for evaluating performance
    
    for i in range(N_EPISODES):
        observation = env.reset()
        done = False
        score = 0
        step = 0

        while not done:
            print("step", step) if step % 100 == 0 else None
            step += 1
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.remember(observation, action, reward, observation_, done)
            if not load_checkpoint:
                agent.learn()  # Since we are evaluating performance, we want a static agent
            observation = observation_

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()  # Since we are evaluating performance, we want a static agent
            
        print('episode', i, '::  score = %.1f' % score, 'avg_score = %.1f' % avg_score)

    if not load_checkpoint:
        x = [i+1 for i in range(N_EPISODES)]
        plot_learning_curve(x, score_history, fig_file)



