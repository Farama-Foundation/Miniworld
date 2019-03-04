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

from .utils import *

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.obs_to_enc = nn.Sequential(
            nn.Conv2d(6, 128, kernel_size=5, stride=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),

            nn.Conv2d(128, 128, kernel_size=5, stride=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),

            nn.Conv2d(128, 128, kernel_size=4, stride=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),

            nn.Conv2d(128, 128, kernel_size=3, stride=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),

            #Print(),
            Flatten(),

            nn.Linear(768, 512),
            nn.LeakyReLU(),
        )

        self.enc_to_delta = nn.Sequential(
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 3),
        )

        self.apply(init_weights)

    def forward(self, obs0, obs1):

        #print(obs0.size())
        obs = torch.cat((obs0, obs1), dim=1)
        #print(obs.size())

        obs = obs / 255

        enc = self.obs_to_enc(obs)

        pos_delta = self.enc_to_delta(enc)

        return pos_delta

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", default=2048, type=int)
    parser.add_argument("--buffer-size", default=1000000, type=int)
    parser.add_argument("--weight-decay", default=0, type=float)
    parser.add_argument("--env", default="MiniWorld-TableTopRobot-v0")
    parser.add_argument("--model-path", default="pos_delta.torch")
    args = parser.parse_args()

    env = gym.make(args.env)

    num_actions = env.action_space.n
    print('num actions:', num_actions)

    max_steps = env.max_episode_steps
    print('max episode steps:', max_steps)

    num_trans = 0
    cur_idx = 0

    # Done indicates that we become done after the current step
    buf_obs = np.zeros(shape=(args.buffer_size, 3, 80, 60), dtype=np.uint8)
    buf_ang = np.zeros(shape=(args.buffer_size, 6), dtype=np.float32)

    model = Model()
    model.cuda()
    print_model_info(model)

    optimizer = optim.Adam(
        model.parameters(),
        lr=1e-4,
        weight_decay=args.weight_decay
    )

    def gen_data():
        global cur_idx, num_trans

        obs = env.reset()
        obs = obs.transpose(2, 1, 0)




        # Pick a random entry index. Prioritize expanding the set.
        if num_trans < args.buffer_size and np.random.uniform(0, 1) < 0.5:
            cur_idx = num_trans
        else:
            cur_idx = np.random.randint(0, num_trans + 1) % args.buffer_size
        num_trans = max(num_trans, cur_idx+1)



        buf_obs0[cur_idx] = obs








    while num_trans <= args.batch_size:
        gen_data()

    running_loss = None

    for i in range(5000000):
        print('batch #{} (num trans={})'.format(i+1, num_trans))

        batch_idx = np.random.randint(0, num_trans - args.batch_size)
        batch_obs0 = make_var(buf_obs0[batch_idx:(batch_idx+args.batch_size)])
        batch_obs1 = make_var(buf_obs1[batch_idx:(batch_idx+args.batch_size)])
        batch_posd = make_var(buf_posd[batch_idx:(batch_idx+args.batch_size)])

        pred_posd = model(batch_obs0, batch_obs1)

        # Generate data while the GPU is computing
        gen_data()

        # Compute an L2 loss
        # Rescale the position loss so the magnitude is similar to the rotation loss
        d0 = pred_posd[:, 0] - batch_posd[:, 0]
        d1 = pred_posd[:, 1] - batch_posd[:, 1]
        d2 = pred_posd[:, 2] - batch_posd[:, 2]
        loss = 10 * (d0*d0).mean() + 30 * (d1*d1).mean() + (d2*d2).mean() # L2 loss
        #diff = pred_posd - batch_posd
        #loss = (diff * diff).mean() # L2 loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i == 0:
            running_loss = loss.data.item()
        else:
            running_loss = 0.99 * running_loss + 0.01 * loss.data.item()

        print('frames: {}'.format((i+1) * args.batch_size))
        print('running loss: {:.5f}'.format(running_loss))

        if i % 100 == 0:
            print('saving model')
            torch.save(model.state_dict(), args.model_path)
