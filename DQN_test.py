import gym
import uw_robot
from DQN_brain import DQN, ReplayBuffer
import torch

# GPU运算
device = torch.device("cuda") if torch.cuda.is_available() \
        else torch.device("cpu")

# hyperparameters setting
num_episode = 100
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
agent.load_state_dict(torch.load('DQN/dqn_test.pth'))   # load the trained model
agent.eval()                                            # turn to evaluation mode

obs = env.reset()                                       # initialize the game world
returns = 0
for _ in range(1000):
    action = agent.take_action(obs)                     # choose action by the model
    obs, reward, done, info = env.step(action)          # update the world
    returns += reward
    env.render()
    if done:
        obs = env.reset()
        print('reward in this episode is: %f' % returns)
