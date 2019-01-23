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
    def __init__(self):
        super().__init__()

        self.obs_to_enc = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),

            nn.Conv2d(32, 32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),

            nn.Conv2d(32, 32, kernel_size=4, stride=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),

            # TODO: add some FC layers here, reduce size to 512?

            #Print(),
            Flatten(),
        )

        self.enc_size = 2240

        self.enc_to_delta = nn.Sequential(
            nn.Linear(self.enc_size, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 4),
        )

        self.apply(init_weights)

    def forward(self, obs0, obs1):
        obs0 = obs0 / 255
        obs1 = obs1 / 255

        obs0_enc = self.obs_to_enc(obs0)
        obs1_enc = self.obs_to_enc(obs1)

        enc = torch.cat((obs0_enc, obs1_enc), dim=1)
        #print(enc.size())

        pos_delta = self.enc_to_delta(enc)

        return pos_delta

parser = argparse.ArgumentParser()
parser.add_argument("--batch-size", default=32, type=int)
parser.add_argument("--buffer-size", default=2048, type=int)
parser.add_argument("--env", default="MiniWorld-MazeS3-v0")
args = parser.parse_args()

env = gym.make(args.env)

num_actions = env.action_space.n
print('num actions:', num_actions)

max_steps = env.max_episode_steps
print('max episode steps:', max_steps)

map_shape = (3, 200, 200)
num_trans = 0
cur_idx = 0

# Done indicates that we become done after the current step
buf_obs0 = np.zeros(shape=(args.buffer_size, 3, 80, 60), dtype=np.uint8)
buf_obs1 = np.zeros(shape=(args.buffer_size, 3, 80, 60), dtype=np.uint8)
buf_posd = np.zeros(shape=(args.buffer_size, 4), dtype=np.float32)

model = Model()
model.cuda()
print_model_info(model)

optimizer = optim.Adam(model.parameters(), lr=1e-4)

def gen_data(num_episodes=1):
    global cur_idx, num_trans

    for i in range(num_episodes):
        #print(i)

        obs = env.reset()
        obs = obs.transpose(2, 1, 0)

        cur_action = None
        steps_left = 0

        # Initial overhead map
        map = make_var(np.zeros(shape=map_shape)).unsqueeze(0)

        for step_idx in range(max_steps):
            # Pick a random transition index. Prioritize expanding the set.
            if num_trans < args.buffer_size and np.random.uniform(0, 1) < 0.5:
                cur_idx = num_trans
            else:
                cur_idx = np.random.randint(0, num_trans + 1) % args.buffer_size
            num_trans = max(num_trans, cur_idx+1)

            # Repeat turn_left, turn_right or move_forward for N steps
            if steps_left == 0:
                cur_action = np.random.choice([
                    env.actions.turn_left,
                    env.actions.turn_right,
                    env.actions.move_forward]
                )
                steps_left = np.random.randint(1, 17)

            pos0 = np.array([*env.agent.pos] + [env.agent.dir])
            buf_obs0[cur_idx] = obs

            obs, reward, done, info = env.step(cur_action)
            obs = obs.transpose(2, 1, 0)
            steps_left -= 1

            pos1 = np.array([*env.agent.pos] + [env.agent.dir])
            buf_obs1[cur_idx] = obs
            buf_posd[cur_idx] = pos1 - pos0

            if done:
                break

while num_trans <= args.batch_size:
    gen_data()

running_loss = None

for i in range(1000000):
    print('batch #{} (num trans={})'.format(i+1, num_trans))

    batch_idx = np.random.randint(0, num_trans - args.batch_size)

    batch_obs0 = make_var(buf_obs0[batch_idx:(batch_idx+args.batch_size)])
    batch_obs1 = make_var(buf_obs1[batch_idx:(batch_idx+args.batch_size)])
    batch_posd = make_var(buf_posd[batch_idx:(batch_idx+args.batch_size)])

    pred_posd = model(batch_obs0, batch_obs1)

    # Generate data while the GPU is computing
    gen_data()

    optimizer.zero_grad()
    diff = pred_posd - batch_posd
    loss = (diff * diff).mean() # L2 loss
    loss.backward()
    optimizer.step()

    if i == 0:
        running_loss = loss.data.item()
    else:
        running_loss = 0.99 * running_loss + 0.01 * loss.data.item()

    print('running loss: {:.3f}'.format(running_loss))

    if i % 100 == 0:
        print('saving model')
        torch.save(model.state_dict(), 'pos_delta_model.torch')
