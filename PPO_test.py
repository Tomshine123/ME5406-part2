import os
import time
from datetime import datetime
import uw_robot
import torch
import numpy as np
import gym
from PPO_brain import PPO


def test():

    env_name = "CustomEnv-v0"
    max_ep_len = 1000           # max timesteps in one episode

    render = True               # render environment on screen
    frame_delay = 0             # if required; add delay b/w frames

    total_test_episodes = 10    # total num of testing episodes
    
    # hyperparameter settings
    K_epochs = 80               # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    gamma = 0.99                # discount factor

    lr_actor = 0.0003           # learning rate for actor
    lr_critic = 0.001           # learning rate for critic

    env = gym.make(env_name)

    # state space dimension
    state_dim = env.observation_space.shape[0]

    # action space dimension
    action_dim = env.action_space.n

    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip)

    # preTrained weights directory

    random_seed = 0             # set this to load a particular checkpoint trained on random seed
    run_num_pretrained = 0      # set this to load a particular checkpoint num

    directory = "PPO/"
    checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
    print("loading network from : " + checkpoint_path)

    # load a trained PPO model
    ppo_agent.load(checkpoint_path)

    test_running_reward = 0

    for ep in range(1, total_test_episodes+1):
        ep_reward = 0
        state = env.reset()

        for t in range(1, max_ep_len+1):
            action = ppo_agent.select_action(state)
            state, reward, done, _ = env.step(action)
            ep_reward += reward

            if render:
                env.render()
                time.sleep(frame_delay)

            if done:
                break

        # clear buffer
        ppo_agent.buffer.clear()

        test_running_reward +=  ep_reward
        print('Episode: {} \t\t Reward: {}'.format(ep, round(ep_reward, 2)))
        ep_reward = 0

    env.close()


    avg_test_reward = test_running_reward / total_test_episodes
    avg_test_reward = round(avg_test_reward, 2)
    print("average test reward : " + str(avg_test_reward))


if __name__ == '__main__':

    test()
