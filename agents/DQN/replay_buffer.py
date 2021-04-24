import numpy as np

# Holds gameplay memory that can be sampled from and updated 
class ReplayBuffer:

    # Simple class that holds the different types of memory
    class Memory:
        def __init__(self,state_shape,action_shape,size):
            self.states = np.zeros((size,state_shape))
            self.actions = np.zeros((size, action_shape[0], action_shape[1]))
            self.rewards = np.zeros(size)
            self.next_states = np.zeros((size,state_shape))
            self.dones = np.zeros(size)
            self.size = size

        # memory[i] will return a tuple of the entire memory @ i
        def __getitem__(self,key):
            return (self.states[key],self.actions[key],self.rewards[key],
                    self.next_states[key],self.dones[key])

        # Provides a quick way of updating multiple
        # parts of memory at a specific index
        def update(self,indx,state=None,action=None,
                   reward=None,next_state=None,done=None):
            self.states[indx] = state

            self.actions[indx] = action

            self.rewards[indx] = reward

            self.next_states[indx] = next_state

            self.dones[indx] = done

        # An alternative to __getitem__, returns dict instead
        def get(self,key):
            rtn = {"states": self.states[key],
                   "actions": self.actions[key],
                   "rewards": self.rewards[key],
                   "next_states": self.next_states[key],
                   "dones": self.dones[key]}
            return rtn

    # Creates the replay buffer
    def __init__(self,state_shape,action_shape,size):
        self.memory = self.Memory(state_shape,action_shape,size)
        self.counter = 0
        self.size = self.memory.size

    # Stores new memory at looping index
    def store(self,state,action,reward,next_state, done):
        indx = self.counter % self.size
        self.memory.update(indx, state, action, reward, next_state, done)
        self.counter += 1

    # Samples the memory from filled parts of the buffer
    def miniBatch(self,batch_size):
        max_memory = min(self.counter,self.size)
        batch_indxs = np.random.choice(max_memory,batch_size)
        return self.memory[batch_indxs]
        