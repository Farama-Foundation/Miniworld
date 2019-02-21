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
            #Flatten(),
            #nn.Linear(2240, 512),
            #nn.LeakyReLU(),
            #nn.Linear(512, 2240),
            #nn.LeakyReLU(),
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

            nn.ConvTranspose2d(64, 1, kernel_size=2, stride=1),
            #nn.BatchNorm2d(32),
            nn.LeakyReLU(),
        )

        self.apply(init_weights)

    def forward(self, img):
        img = img / 255

        x = self.obs_to_enc(img)

        #print(x.size())
        #x = x.view(x.size(0), 64, 7, 5)
        #print(x.size())

        #print(x.size())
        y = self.decoder(x)

        #print(y.size())
        y = 1 * y[:, :, 2:82, 3:63]
        #print(y.size())

        return y

def gen_data():
    global buf_idx
    global buf_avail

    for _ in range(32):
        # Pick a random transition index. Prioritize expanding the set.
        if buf_avail < args.buffer_size and np.random.uniform(0, 1) < 0.5:
            buf_idx = buf_avail
        else:
            buf_idx = np.random.randint(0, buf_avail + 1) % buf_obs.shape[0]
        buf_avail = max(buf_avail, buf_idx+1)

        buf_obs[buf_idx] = env.reset().transpose(2, 1, 0)
        buf_dpt[buf_idx] = env.render_depth().transpose(2, 1, 0)

def depth_to_img(depth):
    depth = 255 * depth / depth.max()
    img = torch.cat([depth, depth, depth], dim=0)
    #print(img.size())
    return img

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--buffer-size", default=65536, type=int)
    parser.add_argument("--env", default="MiniWorld-SimToRealOdo-v0")
    parser.add_argument("--model-path", default="pos_delta.torch")
    args = parser.parse_args()

    buf_obs = np.zeros(shape=(args.buffer_size, 3, 80, 60), dtype=np.uint8)
    buf_dpt = np.zeros(shape=(args.buffer_size, 1, 80, 60), dtype=np.float32)
    buf_idx = 0
    buf_avail = 0

    env = gym.make(args.env)

    model = Model()
    model.cuda()
    print_model_info(model)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    while buf_avail <= args.batch_size:
        gen_data()

    running_loss = None

    for i in range(10000000):
        print('batch #{} (imgs avail {})'.format(i+1, buf_avail-1))

        #print(i, buf_idx)

        batch_idx = np.random.randint(0, buf_avail - args.batch_size)
        batch_obs = make_var(buf_obs[batch_idx:(batch_idx+args.batch_size)])
        batch_dpt = make_var(buf_dpt[batch_idx:(batch_idx+args.batch_size)])

        y = model(batch_obs)

        # Generate data while the GPU is computing
        gen_data()

        optimizer.zero_grad()
        diff = y - batch_dpt
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

            save_img('test_obs.png', batch_obs[0])
            save_img('test_dpt.png', depth_to_img(batch_dpt[0]))
            save_img('test_out.png', depth_to_img(y[0]))

            """
            try:
                for i in range(0, 100):
                    img = load_img('robot_imgs/img_{:03d}.png'.format(i))
                    save_img('img_{:03d}_in.png'.format(i), img[0])
                    save_img('img_{:03d}_out.png'.format(i), depth_to_img(y[0]))
            except:
                pass
            """
