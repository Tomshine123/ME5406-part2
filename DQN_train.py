import numpy as np
import gym
import uw_robot
from DQN_brain import DQN, ReplayBuffer
import torch
import matplotlib.pyplot as plt

device = torch.device("cuda") if torch.cuda.is_available() \
        else torch.device("cpu")

# hyperparameters setting
num_episode = 5500
capacity = 500          # the capacity of experience pool
lr = 2e-3               # learning rate
gamma = 0.9             # discount factor
epsilon = 0.9           # greedy epsilon
target_update = 200     # updating rate for target network
batch_size = 32         # size of data sequence for training
n_hidden = 128          # number of units in each hidden layer
min_size = 200          # the minimum size of data in the pool
return_list = []        # record return of each episode
env_name = 'CustomEnv-v0' 

# load the environment
env = gym.make(env_name)
n_states = env.observation_space.shape[0]  
n_actions = env.action_space.n  

# create an instance of experience pool
replay_buffer = ReplayBuffer(capacity)
# create an instance of DQN model
agent = DQN(n_states=n_states,
            n_hidden=n_hidden,
            n_actions=n_actions,
            learning_rate=lr,
            gamma=gamma,
            epsilon=epsilon,
            target_update=target_update,
            device=device,
        )

# train the model
for i in range(num_episode):
    # initialize the world
    state = env.reset()
    episode_return = 0
    done = False
    while not done:
        # choose action according to policy
        action = agent.take_action(state)
        # update the environment
        next_state, reward, done, _ = env.step(action)
        # add experience to the pool
        replay_buffer.add(state, action, reward, next_state, done)
        # update the current state
        state = next_state
        episode_return += reward

        # begin training after getting enough experiences
        if replay_buffer.size() > min_size:
            # randomly sample from the pool
            s, a, r, ns, d = replay_buffer.sample(batch_size)
            # creat training dataset
            transition_dict = {
                'states': s,
                'actions': a,
                'next_states': ns,
                'rewards': r,
                'dones': d,
            }
            # update network
            agent.update(transition_dict)
    
    return_list.append(episode_return)

    # print reward for each episode
    print(f'iter:{i}, return:{np.mean(return_list[-10:])}')

# save the trained model
torch.save(agent.state_dict(),'DQN/dqn_test.pth')

# plot reward vs episode
episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DQN Returns')
plt.show()