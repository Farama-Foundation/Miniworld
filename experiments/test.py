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
from gym_miniworld.wrappers import *

from utils import *

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            #Print(),
            nn.Conv2d(3, 32, kernel_size=4, stride=2),
            #nn.BatchNorm2d(32),
            nn.LeakyReLU(),

            nn.Conv2d(32, 32, kernel_size=4, stride=2),
            #nn.BatchNorm2d(32),
            nn.LeakyReLU(),

            nn.Conv2d(32, 32, 4, stride=2),
            #nn.BatchNorm2d(32),
            nn.LeakyReLU()
        )

        self.enc_to_out = nn.Sequential(
            nn.Linear(32 * 5 * 8, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 1),
            nn.LeakyReLU(),
        )

        self.apply(init_weights)

    def forward(self, img):
        img = img / 255

        x = self.encoder(img)
        #print(x.size())
        x = x.view(x.size(0), -1)
        #print(x.size())

        y = self.enc_to_out(x)
        return y

def gen_data():
    obs = env.reset()

    box = env.unwrapped.box
    agent = env.unwrapped.agent
    dist = np.linalg.norm(box.pos - agent.pos)

    return obs, dist

if __name__ == "__main__":
    #parser = argparse.ArgumentParser()
    #parser.add_argument('--map-name', required=True)
    #args = parser.parse_args()

    env = gym.make('MiniWorld-OneRoom-v0')
    #env.domain_rand = True
    env = PyTorchObsWrapper(env)

    model = Model()
    model.cuda()
    print_model_info(model)

    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    for i in range(10000):
        print(i)

        obs, target = gen_batch(gen_data, batch_size=32)

        y = model(obs)

        optimizer.zero_grad()
        loss = (target - y).abs().mean()
        loss.backward()
        optimizer.step()

        print(loss.data.item())
