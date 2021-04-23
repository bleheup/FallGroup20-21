import os
import numpy as np
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)
import tensorflow.keras as keras
from tensorflow.keras.layers import Input, Dense
import tensorflow_probability as tfp
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

# False/0 = No debug output; True/1 = Basic debug output; 2 = Lots of debug output
DEBUG = 0

# OPTIONAL: set default float precision
#tf.keras.backend.set_floatx('float64')  

#NUM_BOTS = 12
#NUM_BOT_ACTIONS = 7
#BOT_ACTION_SPACE = 11
#NUM_ACTIONS = NUM_BOTS * BOT_ACTION_SPACE
NUM_ACTIONS = 4  # Number of actions (action space); should be manually passed when instantiating agent
                 # 7 is Everglades' action space, since we can only make 7 moves per step
INPUT_SPACE = 8  # Input dimensions (observation space); should be manually passed when instantiating agent
                   # 249 is Everglades' observation space after one-hot-encoding

BATCH_SIZE = 256

MODEL_VERSION = '1.2.0'
CHECKPOINT_PATH = 'C:/Users/aperr/Desktop/Senior Design/SAC/models/' + MODEL_VERSION + '/'  # TODO: DEPRECATED -- Remove later!
PREV_MODEL_PATH = 'C:/Users/aperr/Desktop/Senior Design/SAC/models/' + MODEL_VERSION + '/'


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
    def __init__(self, fc1_dims=BATCH_SIZE, fc2_dims=BATCH_SIZE, name='critic', n_actions=NUM_ACTIONS, cp_path=CHECKPOINT_PATH):
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
        self.q = Dense(n_actions, activation=None, name='q')  # Q-value for each possible action 
    
    # TODO DISCRETE: *(i) It is now more efficient to have the soft Q-function output the Q-value of each possible action, rather than simply the action provided as an input,
    #                i.e. our Q function moves from (Q : S × A → R) to (Q : S → R^|A|).
    def call(self, state):
        action_value = self.fc1( state )
        action_value = self.fc2( action_value )
        q = self.q(action_value)
        print("q: max/min =", str(np.max(q)) + "/" + str(np.min(q)), "| med =", np.median(q), "| mean =", np.mean(q), " ||  q.shape", q.shape) if DEBUG==2 else None
        return q
    

class ActorNetwork(keras.Model):
    def __init__(self, noise=1e-6, fc1_dims=BATCH_SIZE, fc2_dims=BATCH_SIZE, name='actor', n_action_choices=NUM_ACTIONS, cp_path=CHECKPOINT_PATH):
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
    # Then use prob to return mean and std for our distribution
    # TODO DISCRETE: *(ii) There is now no need for our policy to output the mean and covariance of our action distribution, instead it can directly output our action distribution.
    #                The policy therefore changes from (π : S → R^(2|A|)) to (π : S → [0, 1]^|A|) where now we are applying a softmax function in the final layer of the
    #                policy to ensure it outputs a valid probability distribution.
    def call(self, state):
        probs1 = self.fc1(state)
        #tf.print("fc1:", probs1)
        #tf.print("fc1 is nan? ", tf.reduce_sum( tf.cast(tf.math.is_nan(probs1), tf.int32) ))

        probs2 = self.fc2(probs1)
        #tf.print("fc2:", probs2)
        #tf.print("fc2 is nan? ", tf.reduce_sum( tf.cast(tf.math.is_nan(probs2), tf.int32) ))

        probs_out = self.out(probs2)
        #tf.print("out:", probs_out)
        #tf.print("out is nan? ", tf.reduce_sum( tf.cast(tf.math.is_nan(probs_out), tf.int32) ))

        min_prob = self.noise
        max_prob = 1 - self.noise*tf.cast(self.n_action_choices, tf.float32)
        return tf.clip_by_value( probs_out, min_prob, max_prob )  # This clipping leads to improper probability distributions (AKA sum != 1), but avoids nan results

    # Sample the categorical action distribution outputted by the policy network
    @tf.function( input_signature=[tf.TensorSpec(shape=(None,INPUT_SPACE), dtype=tf.float32)] )  # Define tf.function signature to use when saving/loading model
    def sample_action_dist(self, state):
        probs = self(state)
        log_probs = tf.math.log( probs )  # Add noise if probs==0 to avoid undefined log(0)

        prob_dist = tfp.distributions.Categorical(probs=probs)
        action = prob_dist.sample()
        log_prob = prob_dist.log_prob(action)
        #tf.print("action taken:", actions[0])

        # Returns sampled action, log prob of that sampled action, probabilities of all actions, and log probs of all actions
        return action, log_prob, probs, log_probs


class Agent:
    # Two learning rates: alpha for actor network, beta for value/critic network
    def __init__(self, lr_alpha=0.0003, lr_beta=0.0015, input_dims=[INPUT_SPACE], n_actions=NUM_ACTIONS, env=None, gamma=0.99, max_size=1000000, tau=0.0025,
                 layer1_size=BATCH_SIZE, layer2_size=BATCH_SIZE, batch_size=BATCH_SIZE, temperature=0.4, target_entropy=0.98*(np.log(NUM_ACTIONS))):
        self.gamma = gamma
        self.tau = tau
        print("Agent: input_dims =", input_dims) if DEBUG else None
        print("Agent: n_actions =", n_actions) if DEBUG else None
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.env = env
        
        # Temperature not yet implemented, so ignore this for now
        # (iv) Similarly, we can make the same change as (iii) to our calculation of the temperature loss to also reduce the variance of that estimate.
        self.alpha = temperature
        self.target_entropy = target_entropy

        if load_prev_model:
            self.load(filepath=PREV_MODEL_PATH)
        else:
            self.critic_1 = CriticNetwork(n_actions=NUM_ACTIONS, name='critic_1')
            self.critic_2 = CriticNetwork(n_actions=NUM_ACTIONS, name='critic_2')
            self.critic_target_1 = CriticNetwork(n_actions=NUM_ACTIONS, name='critic_target_1')
            self.critic_target_2 = CriticNetwork(n_actions=NUM_ACTIONS, name='critic_target_2')
            self.actor = ActorNetwork(n_action_choices=NUM_ACTIONS, name='actor')

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
            action, _, _, _ = self.actor.sample_action_dist(state)  # don't care about log probs at this point
            action = action[0].numpy()   # return np array from tensor
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
            print("Took", self.learn_counter, "steps to start learning.") # if DEBUG else None

        for _ in range(n_steps):
            state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

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
                print("\n==== Critic Network Gradient ====") if DEBUG==2 else None
                print("CNG: states.shape", states.shape) if DEBUG == 2 else None
                print("CNG: states_.shape", states_.shape) if DEBUG == 2 else None
                print("CNG: critic_1", self.critic_1( states )) if DEBUG == 2 else None
                print("CNG: critic_2", self.critic_2( states )) if DEBUG == 2 else None
                print("CNG: critic_target_1", self.critic_target_1( states_ )) if DEBUG == 2 else None
                print("CNG: critic_target_2", self.critic_target_2( states_ ), "\n") if DEBUG == 2 else None

                # TODO DISCRETE: *(iii) Before, in order to minimise the soft Q-function cost J_Q(θ) we had to plug in our sampled actions from the replay buffer to form a monte-carlo estimate
                #                of the soft state-value function (because estimating the soft state-value function involved taking an expectation over the action distribution).
                #                However, now, because our action set is discrete, we can fully recover the action distribution and so there is no need to form a monte-carlo estimate
                #                and instead we can calculate the expectation directly.

                # Sample next-state action probs from memory
                _, _, next_probs, next_log_probs = self.actor.sample_action_dist(states_)
                # Get expected q-values for sampled next-states (= q_hat)
                q1_ = tf.squeeze( self.critic_target_1(states_) )
                q2_ = tf.squeeze( self.critic_target_2(states_) )
                q_ = tf.minimum(q1_, q2_)
                #print("q_ =", q_) if DEBUG else None
                critic_q_hat = next_probs * (q_ - self.alpha*next_log_probs)
                #print("critic_q_hat =", critic_q_hat) if DEBUG else None
                critic_q_hat = tf.reduce_sum(critic_q_hat, axis=1)  # Sum estimated "entropy" (?) of all next-state actions
                #print("critic_q_hat reduced =", critic_q_hat) if DEBUG else None
                q_hat = rewards + self.gamma*critic_q_hat*tf.cast(1-done, tf.float32)  # We multiply by (1-done) so we don't get next state rewards if this was the final state of an episode
                #print("q_hat =", q_hat) if DEBUG else None

                # Get q-values for the actions taken at the sampled states (= q)
                actions_mask = tf.convert_to_tensor( keras.utils.to_categorical(actions, dtype='int32') )  # one-hot encode actions to provide a boolean mask when accessing q-values
                q1 = tf.boolean_mask( self.critic_1(states), actions_mask )
                q2 = tf.boolean_mask( self.critic_2(states), actions_mask )
                #print("q1 =", q1) if DEBUG else None
                #print("q2 =", q2) if DEBUG else None

                # Loss calculations
                critic_1_loss = keras.losses.MSE(q1, q_hat)
                #print("critic_1_loss =", critic_1_loss) if DEBUG else None
                critic_2_loss = keras.losses.MSE(q2, q_hat)
                #print("critic_2_loss =", critic_2_loss) if DEBUG else None

            # Calculate gradients
            print("Critic network gradients...") if DEBUG==2 else None
            critic_1_network_gradient = tape.gradient(critic_1_loss, self.critic_1.trainable_variables)
            critic_2_network_gradient = tape.gradient(critic_2_loss, self.critic_2.trainable_variables)
            self.critic_1.optimizer.apply_gradients( zip(critic_1_network_gradient, self.critic_1.trainable_variables) )
            self.critic_2.optimizer.apply_gradients( zip(critic_2_network_gradient, self.critic_2.trainable_variables) )
            print("... gradients applied!") if DEBUG == 2 else None
            # Update critic target network
            self.update_target_network_parameters()

            # -- Actor network --
            with tf.GradientTape() as tape:
                print("\n==== Actor Network Gradient ====") if DEBUG==2 else None
                # Sample action probs from memory
                _, _, policy_probs, policy_log_probs = self.actor.sample_action_dist(states)
                q1_policy = self.critic_1(states)
                q2_policy = self.critic_2(states)
                q_policy = tf.minimum(q1_policy, q2_policy)
                #print("q_policy =", q_policy) if DEBUG else None
                
                # TODO DISCRETE: ??? (v) Before, to minimise Jπ(φ) we had to use the reparameterisation trick to allow gradients to pass through the expectations operator.
                #                However, now our policy outputs the exact action distribution we are able to calculcate the expectation directly.

                actor_loss = policy_probs * (self.alpha*policy_log_probs - q_policy)
                #print("actor_loss =", actor_loss) if DEBUG else None
                actor_loss = tf.reduce_sum(actor_loss, axis=1) + 1e-6  # Sum estimated "entropy" (?) of all next-state actions
                #print("actor_loss sum =", actor_loss) if DEBUG else None
                actor_loss = tf.reduce_mean(actor_loss)
                #print("actor_loss mean =", actor_loss) if DEBUG else None
                
            print("Actor network gradient...") if DEBUG==2 else None

            # Gradient of our loss w/ respect to our actor's trainable variables
            actor_network_gradient = tape.gradient(actor_loss, self.actor.trainable_variables)

            # DEBUG #####################
            #for g,v in zip(actor_network_gradient, self.actor.trainable_variables):
            #    tf.print(v.name, '\t', tf.reduce_min(g), '\t', tf.reduce_mean(g), '\t', tf.reduce_max(g))

            self.actor.optimizer.apply_gradients( zip(actor_network_gradient, self.actor.trainable_variables) )

            print("... gradients applied!") if DEBUG==2 else None

            print("critic_1 loss = {}\tcritic_2 loss = {}\tactor loss = {}".format(critic_1_loss.numpy(), critic_2_loss.numpy(), actor_loss.numpy())) if DEBUG else None
            if tf.math.is_nan(critic_1_loss) or tf.math.is_nan(critic_2_loss) or tf.math.is_nan(actor_loss):
                print("critic_1 loss is nan? =", tf.math.is_nan(critic_1_loss))
                print("critic_2 loss is nan? =", tf.math.is_nan(critic_2_loss))
                print("actor loss is nan? =", tf.math.is_nan(actor_loss))
                print("states =", states)
                #print("states is nan? =", tf.math.is_nan(states))
                print("states_ =", states_)
                #print("states_ is nan? =", tf.math.is_nan(states_))
                print("q_ =", q_)
                print("next_probs =", next_probs)
                print("next_log_probs =", next_log_probs)
                print("q_ - self.alpha*next_log_probs =", q_ - self.alpha*next_log_probs)
                print("next_probs * (q_ - self.alpha*next_log_probs) =", next_probs * (q_ - self.alpha*next_log_probs))
                print("critic_q_hat reduced =", critic_q_hat)
                print("q_hat =", q_hat)
                print("q1 =", q1)
                print("q2 =", q2)
                print("q_policy =", q_policy)
                print("policy_probs =", policy_probs)
                print("policy_log_probs =", policy_probs)
                print("self.alpha*policy_log_probs - q_policy =", self.alpha*policy_log_probs - q_policy)
                print("policy_probs * (self.alpha*policy_log_probs - q_policy) =", policy_probs * (self.alpha*policy_log_probs - q_policy))
                raise SystemExit()

    # Save model state
    def save(self, filepath, input_shape):
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
import gym
import pybullet_envs
import matplotlib.pyplot as plt


def plot_learning_curve(x, scores, figure_file, title=None):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, scores)
    plt.plot(x, running_avg)
    #plt.grid(color="#DDDDDD")
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.legend(['Score', 'Running Average (prev 100)'])
    if title is None:
        plt.title("Model v" + MODEL_VERSION + " performance on " + env_name)
    else:
        plt.title(title)
    plt.savefig(figure_file)
    plt.clf()


# Run parameters
#load_checkpoint = True  # Load previous model weights/network params?
load_prev_model = False  # Load previously saved models for all of our agent's networks?
do_learn        = True  # Update network parameters?
do_save         = True  # Save updated model(s)?
do_plot         = True  # Plot model performance?
do_render       = False  # Display rendered environment?
N_EPISODES = 10000
MAX_STEPS = -1  # Set to -1 to let environment decide max steps
SAVE_DELAY = 0  # Number of episodes before model decides to start saving
SAVE_PERIOD = 7  # Number of episodes between each periodic model save

if __name__ == '__main__':
    env_name = 'LunarLander-v2'
    env = gym.make(env_name)
    input_shape = env.observation_space.shape
    # A single discrete space means there is only one action selected
    n_actions = 1 if type(env.action_space) is gym.spaces.Discrete else env.action_space.shape
    print("\n===== Starting env:", env_name, "=====")
    print('action space =', env.action_space)
    print('*** action space n =', env.action_space.n)
    print('random action sample =', env.action_space.sample(), "\n")
    print('observation space =', env.observation_space)
    print('*** observation space shape =', env.observation_space.shape)
    print('observation space shape[0] =', env.observation_space.shape[0], "\n")
    print('* do_learn =', do_learn)
    print('* do_save =', do_save)
    print('* do_plot =', do_plot)
    print('* do_render =', do_render)
    print('N_EPISODES =', N_EPISODES, "| MAX_STEPS =", MAX_STEPS, "| SAVE_DELAY", SAVE_DELAY, "| SAVE_PERIOD =", SAVE_PERIOD, "\n")
    agent = Agent( input_dims=input_shape, env=env, n_actions=n_actions )

    best_score = env.reward_range[0]
    score_history = []

    # Filename string for output file
    fig_path = 'C:/Users/aperr/Desktop/Senior Design/SAC/plots/' + env_name + '/'
    time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    fname = MODEL_VERSION + '_' + time_str + '.png'
    fig_file =  fig_path + fname

    if do_render:
        env.reset()
        env.render(mode="human")

    # Run
    for i in range(N_EPISODES):
        observation = env.reset()
        done = False
        score = 0
        steps = 0

        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.remember(observation, float(action), reward, observation_, done)
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
                agent.save(filepath=PREV_MODEL_PATH + time_str + '_'+str(i+1)+'eps/', input_shape=input_shape)
            if do_plot:
                if not os.path.exists(fig_path):
                    os.makedirs(fig_path)
                x = [j+1 for j in range(i+1)]
                plot_learning_curve(x, score_history, fig_file)

        # Save model
        if i > SAVE_DELAY and avg_score > best_score:
            best_score = avg_score
            if do_save and agent.is_learning:
                agent.save(filepath=PREV_MODEL_PATH, input_shape=input_shape)
            
        print('episode', i, '::  score = %.1f' % score, 'avg_score = %.1f' % avg_score)

    if do_plot:
        x = [i+1 for i in range(N_EPISODES)]
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)
        plot_learning_curve(x, score_history, fig_file)

