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
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 2)
        )

        self.apply(init_weights)

    def forward(self, img):
        img = img / 255

        x = self.encoder(img)
        #print(x.size())
        x = x.view(x.size(0), -1)
        #print(x.size())

        out = self.enc_to_out(x)
        return out

def gen_data():
    obs = env.reset()

    box = env.unwrapped.box
    agent = env.unwrapped.agent

    #dist = np.linalg.norm(box.pos - agent.pos)
    #return obs, dist

    vec = box.pos - agent.pos
    dot_f = np.dot(agent.dir_vec, vec)
    dot_r = np.dot(agent.right_vec, vec)
    return obs, (dot_f, dot_r)

def visualize():

    for _ in range(8):

        obs = env.reset()

        env.render('human')

        dot_r = 0

        for _ in range(80):

            last_dot_r = dot_r

            obs = make_var(obs).unsqueeze(0)
            out = model(obs).squeeze(0)
            dot_f = out.data[0].item()
            dot_r = out.data[1].item()

            print(dot_f)
            print(dot_r)

            if dot_f < 0:
                action = env.unwrapped.actions.turn_right
            else:
                if abs(dot_r) < 0.5 or last_dot_r < 0 and dot_r > 0 or last_dot_r > 0 and dot_r < 0:
                    action = env.unwrapped.actions.move_forward
                elif dot_r > 0:
                    action = env.unwrapped.actions.turn_right
                else:
                    action = env.unwrapped.actions.turn_left

            obs, _, _, _ = env.step(action)

            env.render('human')
            #time.sleep(0.025)

        time.sleep(0.4)


if __name__ == "__main__":
    #parser = argparse.ArgumentParser()
    #parser.add_argument('--map-name', required=True)
    #args = parser.parse_args()

    env = gym.make('MiniWorld-SimToReal1-v0')
    env.domain_rand = True
    env = PyTorchObsWrapper(env)

    model = Model()
    model.cuda()
    print_model_info(model)

    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    for i in range(10000000):
        print(i)

        obs, target = gen_batch(gen_data, batch_size=32)
        y = model(obs)

        optimizer.zero_grad()
        loss = (target - y).abs().mean()
        loss.backward()
        optimizer.step()

        #print(loss.data.item())

        if i % 200 == 0:
            visualize()
