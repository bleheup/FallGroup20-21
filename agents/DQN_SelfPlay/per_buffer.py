import numpy as np
import random
from segment_tree import SegmentTree, MinSegmentTree, SumSegmentTree

# Holds gameplay memory that can be sampled from and updated


class PERBuffer:

    # Simple class that holds the different types of memory
    class Memory:
        # Expects all shapes to be tuples, size to be an integer
        def __init__(self, state_shape, action_shape, size):
            self.states = np.zeros((size,) + state_shape)
            self.actions = np.zeros((size,) + action_shape)
            self.rewards = np.zeros(size)
            self.next_states = np.zeros((size,) + state_shape)
            self.dones = np.zeros(size)
            self.size = size

        # memory[i] will return a tuple of the entire memory @ i
        def __getitem__(self, key):
            return (self.states[key], self.actions[key], self.rewards[key],
                    self.next_states[key], self.dones[key])

        # Provides a quick way of updating multiple
        # parts of memory at a specific index
        def update(self, indx, state=None, action=None,
                   reward=None, next_state=None, done=None):
            self.states[indx] = state
            self.actions[indx] = action
            self.rewards[indx] = reward
            self.next_states[indx] = next_state
            self.dones[indx] = done

        # An alternative to __getitem__, returns dict instead
        def get(self, key):
            rtn = {"states": self.states[key],
                   "actions": self.actions[key],
                   "rewards": self.rewards[key],
                   "next_states": self.next_states[key],
                   "dones": self.dones[key]}
            return rtn

    # Creates the replay buffer
    def __init__(self, state_shape, action_shape, size,
                 alpha=0.6, beta=0.4, beta_delta=0.001, epsilon=0.01):
        self.memory = self.Memory(state_shape, action_shape, size)
        self.counter = 0
        self.size = self.memory.size
        # Segment trees
        self.sum_tree = SumSegmentTree(self.size)
        self.min_tree = MinSegmentTree(self.size)
        # P.E.R. hyperparameters
        self.alpha = alpha
        self.beta = beta
        self.beta_delta = beta_delta
        self.epsilon = epsilon
        self.max_priority = 1.0

    # Samples the indexes from memory in accordance to their priority
    def sample_indexes(self, batch_size, max_memory):
        sample_indexes = np.zeros(shape=batch_size)
        # Gets the total probability of all used memory
        prob_total_norm = self.sum_tree.sum(0, max_memory-1) / batch_size
        # Gets indexes using probability
        for i in range(batch_size):
            # ---VAL MAY NEED TO BE CHANGED---
            val = random.random() * prob_total_norm + i * prob_total_norm
            indx = self.sum_tree.find_prefixsum_idx(val)
            sample_indexes[i] = indx
        return sample_indexes

    # Stores new memory at looping index
    def store(self, state, action, reward, next_state, done):
        indx = self.counter % self.size
        self.memory.update(indx, state, action, reward, next_state, done)
        # Gets the priority alpha for the newly added sample
        priority_alpha = self.max_priority ** self.alpha
        # Adds this to the sum and min trees
        self.sum_tree[indx] = priority_alpha
        self.min_tree[indx] = priority_alpha
        # Updates the counter
        self.counter += 1

    # Samples the memory from filled parts of the buffer
    # Returns a tuple (states, actions, rewards, next_states, dones, weights)
    def miniBatch(self, batch_size):
        max_memory = min(self.counter, self.size)
        # Samples the indexes according to their importance
        batch_indxs = self.sample_indexes(batch_size, max_memory)
        batch_indxs = np.int_(batch_indxs)
        # Gets the weights
        weights = np.zeros(shape=batch_size)
        prob_min = self.min_tree.min() / (self.sum_tree.sum() + self.epsilon)
        max_weight = (prob_min * max_memory) ** (-self.beta)
        for i in range(0, len(batch_indxs)):
            prob = self.sum_tree[batch_indxs[i]] / \
                (self.sum_tree.sum() + self.epsilon)
            weight = (prob * max_memory) ** (-self.beta)
            weight_norm = weight / (max_weight + self.epsilon)
            weights[i] = weight_norm
        # Updates beta
        self.beta = min(1.0, self.beta + self.beta_delta)
        # Returns memory and weights and idxs
        return self.memory[batch_indxs] + (weights,) + (batch_indxs,)

    # for given indexes and priorities, updates the trees and max
    def update_priorities(self, indxs, priorities):
        for indx, priority in zip(indxs, priorities):
            priority_alpha = priority ** self.alpha
            self.sum_tree[indx] = priority_alpha[0]
            self.min_tree[indx] = priority_alpha[0]
            # Sets the max priority to be newest max
            self.max_priority = max(self.max_priority, priority)
