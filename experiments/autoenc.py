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

from .utils import *

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.obs_to_enc = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),

            nn.Conv2d(64, 64, kernel_size=5, stride=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),

            nn.Conv2d(64, 64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),

            #Print(),
            Flatten(),

            nn.Linear(2240, 512),
            nn.LeakyReLU(),

            nn.Linear(512, 2240),
            nn.LeakyReLU(),
        )

        self.decoder = nn.Sequential(
            #Print(),

            nn.ConvTranspose2d(64, 64, kernel_size=6, stride=2),
            #nn.BatchNorm2d(32),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(64, 64, kernel_size=6, stride=2),
            #nn.BatchNorm2d(32),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2),
            #nn.BatchNorm2d(32),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(64, 3, kernel_size=2, stride=1),
            #nn.BatchNorm2d(32),
            nn.LeakyReLU(),
        )

        self.apply(init_weights)

    def forward(self, img):
        img = img / 255

        x = self.obs_to_enc(img)

        #print(x.size())
        x = x.view(x.size(0), 64, 7, 5)
        #print(x.size())

        #print(x.size())
        y = self.decoder(x)

        #print(y.size())
        y = 255 * y[:, :, 2:82, 3:63]
        #print(y.size())

        return y

def gen_data():
    global cur_gen_idx
    global idx_avail

    for _ in range(32):
        buffer[cur_gen_idx] = env.reset().transpose(2, 1, 0)
        cur_gen_idx = (cur_gen_idx + 1) % buffer.shape[0]
        idx_avail = max(idx_avail, cur_gen_idx)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--buffer-size", default=65536, type=int)
    parser.add_argument("--env", default="MiniWorld-SimToRealOdo2-v0")
    parser.add_argument("--model-path", default="pos_delta.torch")
    args = parser.parse_args()

    buffer = np.zeros(shape=(args.buffer_size, 3, 80, 60), dtype=np.uint8)
    cur_gen_idx = 0
    idx_avail = 0

    env = gym.make(args.env)

    model = Model()
    model.cuda()
    print_model_info(model)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    while idx_avail <= args.batch_size:
        gen_data()

    running_loss = None

    for i in range(10000000):
        print('batch #{} (imgs avail {})'.format(i+1, idx_avail-1))

        #print(i, cur_gen_idx)

        batch_idx = np.random.randint(0, idx_avail - args.batch_size)
        batch = buffer[batch_idx:(batch_idx+args.batch_size)]
        batch = make_var(batch)

        y = model(batch)

        # Generate data while the GPU is computing
        gen_data()

        optimizer.zero_grad()
        diff = y - batch
        loss = (diff * diff).mean() # L2 loss
        #loss = (y - batch).abs().mean() # L1 loss
        loss.backward()
        optimizer.step()

        if i == 0:
            running_loss = loss.data.item()
        else:
            running_loss = 0.99 * running_loss + 0.01 * loss.data.item()

        print('running loss: {:.5f}'.format(running_loss))

        if i > 0 and i % 100 == 0:
            # TODO: save model

            save_img('test_obs.png', batch[0])
            save_img('test_out.png', y[0])

            try:
                for i in range(0, 100):
                    img = load_img('robot_imgs/img_{:03d}.png'.format(i))
                    y = model(img)
                    save_img('img_{:03d}_in.png'.format(i), img[0])
                    save_img('img_{:03d}_out.png'.format(i), y[0])
            except:
                pass
