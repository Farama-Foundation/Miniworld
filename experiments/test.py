#!/usr/bin/env python3

import time
import random
import argparse
import math
import json
from functools import reduce
import operator

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import gym
import gym_miniworld

from utils import *

class Model(nn.Module):
    def __init__(self, num_actions):
        super().__init__()

        self.encoder = nn.Sequential(
            #Print(),
            nn.Conv2d(3, 32, kernel_size=4, stride=2),

            #nn.BatchNorm2d(32),
            nn.LeakyReLU(),

            nn.Conv2d(32, 32, kernel_size=4, stride=2),
            #nn.BatchNorm2d(32),
            nn.LeakyReLU(),

            #nn.Conv2d(32, 32, 4, stride=2),
            #nn.BatchNorm2d(32),
            #nn.LeakyReLU()
        )

        self.enc_to_actions = nn.Sequential(
            nn.Linear(32 * 6 * 6, num_actions),
            nn.LeakyReLU(),
        )

        self.apply(init_weights)

    def forward(self, img):
        img = img / 255

        x = self.encoder(img)
        x = x.view(x.size(0), -1)

        action_vals = self.enc_to_actions(x)

        max_val, action = action_vals.max(1)

        return action




def gen_data():
    idx = random.randint(0, len(positions) - 1)
    cur_pos = np.array(positions[idx][0])
    cur_angle = positions[idx][1]
    vels = np.array(actions[idx])

    env.unwrapped.cur_pos = cur_pos
    env.unwrapped.cur_angle = cur_angle

    obs = env.unwrapped.render_obs().copy()
    obs = obs.transpose(2, 0, 1)

    return obs, vels

def eval_episode(model, env, seed=0):
    total_reward = 0

    env.seed(seed)
    obs = env.reset()

    while True:

        obs = obs.transpose(2, 0, 1)
        obs = make_var(obs)
        obs = obs.unsqueeze(0)
        action = model(obs)

        obs, reward, done, info = env.step(action)

        total_reward += reward

        if done:
            break

    return total_reward

def eval_model(model, env, num_episodes=20):
    total_reward = 0

    for i in range(0, num_episodes):
        total_reward += eval_episode(model, env, seed=i)

    return total_reward / num_episodes

def visualize(model, env, num_episodes=5):
    env.seed(0)

    for i in range(0, num_episodes):
        obs = env.reset()

        while True:
            env.render('human')

            obs = obs.transpose(2, 0, 1)
            obs = make_var(obs)
            obs = obs.unsqueeze(0)
            action = model(obs)

            #print(action)

            obs, reward, done, info = env.step(action)
            #time.sleep(1 / env.frame_rate / 3)

            if done:
                break

    return

if __name__ == "__main__":
    #parser = argparse.ArgumentParser()
    #parser.add_argument('--map-name', required=True)
    #args = parser.parse_args()


    env = gym.make('MiniWorld-Hallway-v0')

    """
    model = Model(env.action_space.n)
    model.cuda()
    print_model_info(model)
    """

    best_model = None
    best_r = -math.inf

    for i in range(0, 70):
        #model = Model(env.action_space.n)
        model = Model(3)
        model.cuda()

        r = eval_model(model, env)

        print('itr #%d, r=%s' % (i, r))

        if r > best_r:
            best_model = model
            best_r = r

            print('new best model')



    visualize(best_model, env, 1000000)
