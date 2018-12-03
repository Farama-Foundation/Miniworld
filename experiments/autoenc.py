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

            nn.Conv2d(3, 32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),

            nn.Conv2d(32, 32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),

            nn.Conv2d(32, 32, kernel_size=4, stride=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),

            nn.Dropout(0.2),
        )

        self.decoder = nn.Sequential(
            #Print(),

            nn.ConvTranspose2d(32, 32, kernel_size=6, stride=2),
            #nn.BatchNorm2d(32),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(32, 32, kernel_size=6, stride=2),
            #nn.BatchNorm2d(32),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2),
            #nn.BatchNorm2d(32),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(32, 3, kernel_size=2, stride=1),
            #nn.BatchNorm2d(32),
            nn.LeakyReLU(),
        )

        """
        self.enc_to_out = nn.Sequential(
            nn.Linear(32 * 5 * 8, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 2)
        )
        """

        self.apply(init_weights)

    def forward(self, img):
        img = img / 255

        x = self.encoder(img)

        #print(x.size())
        #x = x.view(x.size(0), -1)
        #print(x.size())
        #out = self.enc_to_out(x)
        #return out

        #print(x.size())
        y = self.decoder(x)

        #print(y.size())
        y = 255 * y[:, :, 2:82, 3:63]
        #print(y.size())

        return y


batch_size = 128

buffer = np.zeros(shape=(32768, 3, 80, 60))
cur_gen_idx = 0
idx_avail = 0

def gen_data():
    global cur_gen_idx
    global idx_avail

    for _ in range(32):
        buffer[cur_gen_idx] = env.reset()
        cur_gen_idx = (cur_gen_idx + 1) % buffer.shape[0]
        idx_avail = max(idx_avail, cur_gen_idx)

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

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    while idx_avail <= batch_size:
        gen_data()

    for i in range(10000000):
        print(i, cur_gen_idx)

        batch_idx = np.random.randint(0, idx_avail - batch_size)
        batch = buffer[batch_idx:(batch_idx+batch_size)]
        batch = make_var(batch)

        y = model(batch)

        # Generate data while the GPU is computing
        gen_data()

        optimizer.zero_grad()
        diff = y - batch
        loss = (diff * diff).mean() # L2 loss
        #loss = (y - batch).abs().mean()
        loss.backward()
        optimizer.step()

        print(loss.data.item())

        if i % 50 == 0:
            save_img('test_obs.png', batch[0])
            save_img('test_out.png', y[0])
