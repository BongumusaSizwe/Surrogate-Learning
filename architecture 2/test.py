import gym
import torch

import argparse
import random

from dqn.agent import DQNAgent
from dqn.replay_buffer import ReplayBuffer
from dqn.wrappers import *

import os
from skimage.color import rgb2gray
from PIL import Image
import numpy as np

# Save file
def path_name(env):
    directory = 'Models/'
    env_name = env.unwrapped.spec.id.lower()
    path = os.path.join(directory, env_name)
    dir_exist = os.path.isdir(path)
    dataset_name = env_name

    if not dir_exist:
        os.mkdir(path)

    count = 1
    for dirname, dirnames, filenames in os.walk(path):
        for filename in filenames:
            print(os.path.join(dirname, filename))
            count += 1
    #Create and save dataset
    loc = dataset_name.find('-')
    filepath = dataset_name[:loc]+str(count)
    
    return directory + env_name +'/' +  filepath


def make_video(env, TrainNet, generate_video= False):
    '''generate_video: Make video output.
        caution: not to generate 100 videos
    '''
    path = path_name(env)
    if generate_video:
        env = gym.wrappers.Monitor(env, path, force=True)
    rewards = 0
    done = False
    steps = 0
    state = env.reset()
    
    ##Taking actions for Supervised learning purposes    
    # states_array = []
    # action_array = []
    while not done:
        action = TrainNet.act(state)
        state, reward, done, _ = env.step(action)
        steps +=1
        # action_array.append(action)
        # states_array.append(np.array(state))
        rewards += reward
        print("Testing steps: {} rewards {}: ".format(steps, rewards))
        
        if done:
            print("Episode finish")
            state = env.reset()
    # print("Saving state-action pairs")
    # states = np.array(states_array)
    # actions = np.array(action_array)
    # filepath = path_name(env)
    # np.savez_compressed(filepath, states = states, actions = actions)
    return rewards
scores = []

replay_buffer = ReplayBuffer(int(1e6))
env = gym.make('PongNoFrameskip-v4')
#env.seed(42)

for t in range(100):
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    env = EpisodicLifeEnv(env)
    env = FireResetEnv(env)
    env = WarpFrame(env)
    env = PyTorchFrame(env)
    env = ClipRewardEnv(env)
    env = FrameStack(env, 4)
    policy_net= DQNAgent(env.observation_space, env.action_space, replay_buffer)
    target_net = DQNAgent(env.observation_space, env.action_space, replay_buffer)
    target_net.policy_network.load_state_dict(torch.load('Models/Learning Rate/pongnoframeskip-v4/pongnoframeskip1.pth'))

    score = make_video(env, target_net)
    scores.append(score)
scores = np.array(scores)

print(np.mean(scores))