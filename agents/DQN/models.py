import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from noisy_net import NoisyLinear, NoisyFactorizedLinear
from player import PlayerHelper
from utils import build_action_table
import math

class BranchingQNetwork(nn.Module):
    def __init__(self, observation_space, action_space, action_bins, hidden_dim, exploration_method, architecture = "Double"):
        super().__init__()
        self.exploration_method = exploration_method
        self.architecture = architecture
        if self.exploration_method == "Noisy":
            self.model = nn.Sequential(
                NoisyLinear(observation_space, hidden_dim*4),
                nn.ReLU(),
                NoisyLinear(hidden_dim*4, hidden_dim*2),
                nn.ReLU(),
                NoisyLinear(hidden_dim*2, hidden_dim),
                nn.ReLU()
            )
            if architecture == "Dueling":
                self.value_head = NoisyLinear(hidden_dim, 1)
                self.adv_heads = NoisyLinear(hidden_dim, action_bins)
            else:
                self.out = NoisyLinear(hidden_dim, action_bins)
        else:
            self.model = nn.Sequential(
                nn.Linear(observation_space, hidden_dim*4),
                nn.ReLU(),
                nn.Linear(hidden_dim*4, hidden_dim*2),
                nn.ReLU(),
                nn.Linear(hidden_dim*2, hidden_dim),
                nn.ReLU()
            )
            if architecture == "Dueling":
                self.value_head = nn.Linear(hidden_dim, 1)
                self.adv_heads = nn.Linear(hidden_dim, action_bins)
            else:
                self.out = nn.Linear(hidden_dim, action_bins)

    def forward(self, x):
        first_layer = self.model(x)
        
        if self.architecture == "Dueling":
            value = self.value_head(first_layer)
            advs = self.adv_heads(first_layer)
            q_val = value + advs - advs.mean()
        else:
            out = self.out(first_layer)
            q_val = out
        return q_val

    def sample_noise(self):
        self.model[0].sample_noise()
        self.model[2].sample_noise()
        self.model[4].sample_noise()
        if self.architecture == "Dueling":
            self.value_head.sample_noise()
            self.adv_heads.sample_noise()
        else:
            self.out.sample_noise()


class BranchingDQN(nn.Module):
    def __init__(self, 
    observation_space, 
    action_space,
    action_bins, 
    target_update_freq, 
    learning_rate, 
    gamma, 
    hidden_dim, 
    td_target, 
    device, 
    exploration_method, 
    architecture="Double",
    isPer=False,
    nSteps=3):
        super().__init__()

        self.observation_space = observation_space
        self.isPer = isPer
        self.action_space = action_space
        self.action_bins = action_bins
        self.gamma = gamma
        self.exploration_method = exploration_method
        self.architecture = architecture
        self.player_helper = PlayerHelper(7,1, "../config/DemoMap.json")
        self.policy_network = BranchingQNetwork(
            observation_space, action_space, action_bins, hidden_dim, exploration_method, architecture)
        if architecture != "Base":
            self.target_network = BranchingQNetwork(
                observation_space, action_space, action_bins, hidden_dim, exploration_method, architecture)
            self.target_network.load_state_dict(self.policy_network.state_dict())
            self.target_network.to(device)

        self.optim = optim.Adam(
            self.policy_network.parameters(), lr=learning_rate)

        self.action_choices = build_action_table()

        self.policy_network.to(device)
        
        self.device = device

        self.target_update_freq = target_update_freq
        self.update_counter = 0

        self.td_target = td_target

        self.nSteps = nSteps

        self.nBuffer = []

    def get_action(self, x):
        #turn = x[0]
        legal_obs = self.player_helper.legal_moves(x)
        x = torch.from_numpy(x).float().to(self.device)
        with torch.no_grad():
            if self.exploration_method == "Noisy":
                self.policy_network.sample_noise()

            preds = self.policy_network(x)
            actions = self.legal_move_decider(legal_obs, preds)[0]
           
        return actions # action.numpy()
    
    def legal_move_decider(self, legal_obs, preds):
        preds = preds.reshape(12,11)
        legal_obs = legal_obs.reshape(12,11)
        for i in range(len(legal_obs)):
            for j in range(len(legal_obs[0])):
                if legal_obs[i][j] == False:
                    preds[i][j] = 0

        actions = preds.max(1)
        actions_max = actions.values.sort(descending=True)
        actions_max_values = actions_max.values[:7]
        actions_max_idx = actions_max.indices[:7]
        location = actions.indices.gather(0, actions_max_idx)
        group = actions_max_idx
        location = location + 1
        actions = np.array([[g, l] for g, l in zip(group.detach().cpu().numpy(), location.detach().cpu().numpy())])
        #print(preds)
        for i in range(len(actions)):
            if actions_max_values[i] == 0:
                actions[i] = [0,0]
        return (actions, actions_max_values) #tensor

    def append_to_replay(self, memory, s, a, r, n, d):
        self.nBuffer.append((s,a,r,n,d))
        if len(self.nBuffer) < self.nSteps:
            return
        R = sum([self.nBuffer[i][2] * (self.gamma**i) for i in range(self.nSteps)])
        state, action, _, nextState, done = self.nBuffer.pop(0)

        memory.add(state, action, R, nextState, done)

    def finish_nstep(self, memory):
        while len(self.nBuffer) > 0:
            R =  sum([self.nBuffer[i][2] * (self.gamma**i) for i in range(len(self.nBuffer))])
            state, action, _, nextState, done = self.nBuffer.pop(0)
            memory.add(state, action, R, nextState, done)

    def update_policy(self, batch, memory):
        sample, batch_indxs, batch_weights = batch
        
        batch_states = sample[0]
        batch_actions = sample[1]
        batch_rewards = sample[2]
        batch_next_states = sample[3]
        batch_done = sample[4]

        states = torch.tensor(batch_states).float().to(self.device)
        rewards = torch.tensor(batch_rewards).float().to(self.device).unsqueeze(1)
        batch_legal_moves = []
        for nState in batch_next_states:
            legal_nState = self.player_helper.legal_moves(nState)
            batch_legal_moves.append(legal_nState)
        #print("batch action choices", batch_action_choices)
        next_states = torch.tensor(batch_next_states).float().to(self.device)
        if self.exploration_method == "Noisy":
            self.policy_network.sample_noise()
        current_Q = self.policy_network(states)
        
        new_current_Q = []
        for idx, action_group in enumerate(batch_actions):
            b = []
            for action in action_group:
                compute_idx = action[0]*11 + (action[1] - 1)
                compute_idx = compute_idx.astype(int)
                if compute_idx < 0:
                    with torch.no_grad():
                        b.append(torch.tensor(0).to(self.device))
                else:
                    b.append(current_Q[idx][compute_idx])
            b = torch.stack(b)
            new_current_Q.append(b)
            
        new_current_Q = torch.stack(new_current_Q)

        #print("Current Q", current_Q)
        with torch.no_grad():
            if self.architecture != "Base":
                if self.exploration_method == "Noisy":
                    self.target_network.sample_noise()
                next_Q = self.target_network(next_states)
            else:
                if self.exploration_method == "Noisy":
                    self.policy_network.sample_noise()
                next_Q = self.policy_network(next_states)
            next_Q_final = []
            for i, n in enumerate(next_Q):
                next_Q_final.append(self.legal_move_decider(batch_legal_moves[i], next_Q[i])[1])
            next_Q_final = torch.stack(next_Q_final)
            #print("next q", next_Q_final)
            
        expected_Q = rewards + next_Q_final * (self.gamma**self.nSteps)
        #print("Expect:", expected_Q, "Current:", new_current_Q)
        #print("Expect Q", expected_Q)
        #print("weights", batch_weights)
        batch_weights = torch.tensor(batch_weights).float().to(self.device)
        # print(batch_weights)
        # print(batch_indxs)
        loss = batch_weights * F.mse_loss(new_current_Q, expected_Q)
        #print("loss", loss)
        prios = loss + 1e-5
        loss = loss.mean()
        # print(self.policy_network)
        self.optim.zero_grad()
        loss.backward()
        # print(self.policy_network.parameters())
        # for p in self.policy_network.parameters():
        #     p.grad.data.clamp_(-1., 1.)
        self.optim.step()

        if self.isPer:
            memory.update_priorities(batch_indxs, prios.data.cpu().numpy())

        if self.architecture != "Base":
            self.update_counter += 1
            if self.update_counter % self.target_update_freq == 0:
                #print("Update target net")
                self.update_counter = 0
                self.target_network.load_state_dict(
                    self.policy_network.state_dict())

        return loss.detach().cpu()
