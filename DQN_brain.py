import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import collections
import random

# experience replay pool

class ReplayBuffer():
    def __init__(self, capacity):
        # create a queue to hold the experience
        self.buffer = collections.deque(maxlen=capacity)
    # add experience to the poor
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    # randomly sample a number of data from the pooor
    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done
    # the current size of queue
    def size(self):
        return len(self.buffer)
   
# build network for DQN

class Net(nn.Module):
    # build a network with a hidden layer
    def __init__(self, n_states, n_hidden, n_actions):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_states, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_actions)
    # forward propagation
    def forward(self, x):  # [b,n_states]
        x = self.fc1(x)
        x = self.fc2(x)
        return x

# define a DQN training model

class DQN(nn.Module):
    def __init__(self, n_states, n_hidden, n_actions,
                 learning_rate, gamma, epsilon,
                 target_update, device):
        super(DQN, self).__init__()
        self.n_states = n_states  
        self.n_hidden = n_hidden  
        self.n_actions = n_actions  
        self.learning_rate = learning_rate  
        self.gamma = gamma  
        self.epsilon = epsilon  # greedy policy
        self.target_update = target_update  # rate of update for target network
        self.device = device 
        self.count = 0          # record number of iterations
        
        # create two networks for DQN
        # create a network to be trained
        self.q_net = Net(self.n_states, self.n_hidden, self.n_actions)
        # create an instance of target network
        self.target_q_net = Net(self.n_states, self.n_hidden, self.n_actions)
        # choose optimizer for networks
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.learning_rate)

    # function to choose action
    def take_action(self, state):
        state = torch.Tensor(state[np.newaxis, :])
        # take epsilon-greedy policy
        if np.random.random() < self.epsilon:  
            actions_value = self.q_net(state)
            # choose the action with largest value
            action = actions_value.argmax().item()  
        else:
            # randomly choose an possible action
            action = np.random.randint(self.n_actions)
        return action

    # network updating
    def update(self, transition_dict):
        # get observations for current state
        states = torch.tensor(transition_dict['states'], dtype=torch.float)
        # get observations for current state
        actions = torch.tensor(transition_dict['actions']).view(-1,1)
        # get the resulting reward
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1,1)
        # get observations for next state
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float)
        # check if the termination is reached
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1,1)

        # input the current state and get different rewards according to actions
        q_values = self.q_net(states).gather(1, actions)  # [b,1]
        # pick the largest value for next state
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1,1)
        # the target value of current state
        q_targets = rewards + self.gamma * max_next_q_values * (1-dones)

        # the error between training and target networks
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))
        # initialize the gradient
        self.optimizer.zero_grad()
        # backpropagation
        dqn_loss.backward()
        # network updating
        self.optimizer.step()

        # copy the parameters from training network to the target network after n timesteps
        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(
                self.q_net.state_dict())
        
        self.count += 1