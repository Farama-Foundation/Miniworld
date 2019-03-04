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

        self.obs_to_ang = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=5, stride=2),
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

            nn.Linear(512, 512),
            nn.LeakyReLU(),

            nn.Linear(512, 6),
        )

        self.apply(init_weights)

    def forward(self, obs):
        obs = obs / 255

        ang = self.obs_to_ang(obs)

        return ang

def recon_test(env, model):

    for i in range(10):

        obs = env.reset()
        obs = obs.transpose(2, 1, 0)
        obs = make_var(obs).unsqueeze(0)

        pred_angles = model(obs)

        img_orig = env.render('rgb_array')

        pred_angles = pred_angles.reshape(-1)
        pred_angles = pred_angles.detach().cpu().numpy()
        env.ergojr.angles = pred_angles

        img_pred = env.render('rgb_array')

        save_img('test_{:03d}_orig.png'.format(i), img_orig)
        save_img('test_{:03d}_pred.png'.format(i), img_pred)

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

    # Done indicates that we become done after the current step
    buf_obs = np.zeros(shape=(args.buffer_size, 3, 80, 60), dtype=np.uint8)
    buf_ang = np.zeros(shape=(args.buffer_size, 6), dtype=np.float32)

    buf_num = 0
    cur_idx = 0

    model = Model()
    model.cuda()
    print_model_info(model)

    optimizer = optim.Adam(
        model.parameters(),
        lr=1e-4,
        weight_decay=args.weight_decay
    )

    def gen_data():
        global cur_idx, buf_num

        obs = env.reset()
        obs = obs.transpose(2, 1, 0)

        # Pick a random entry index. Prioritize expanding the set.
        #if buf_num < args.buffer_size and np.random.uniform(0, 1) < 0.5:
        #    cur_idx = buf_num
        #else:
        #    cur_idx = np.random.randint(0, buf_num + 1) % args.buffer_size
        cur_idx = buf_num
        buf_num = max(buf_num, cur_idx+1)

        buf_obs[cur_idx] = obs
        buf_ang[cur_idx] = env.ergojr.angles

    while buf_num <= args.batch_size:
        gen_data()

    running_loss = None

    start_time = time.time()

    for batch_no in range(1, 5000000):
        print('batch #{} (num entries={})'.format(batch_no, buf_num))

        batch_idx = np.random.randint(0, buf_num - args.batch_size)
        batch_obs = make_var(buf_obs[batch_idx:(batch_idx+args.batch_size)])
        batch_ang = make_var(buf_ang[batch_idx:(batch_idx+args.batch_size)])

        pred_ang = model(batch_obs)

        # Generate data while the GPU is computing
        for i in range(16):
            gen_data()

        # Compute an L2 loss
        diff = pred_ang - batch_ang
        loss = (diff * diff).mean() # L2 loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_no == 1:
            running_loss = loss.data.item()
        else:
            running_loss = 0.99 * running_loss + 0.01 * loss.data.item()

        frame_count = batch_no * args.batch_size
        total_time = time.time() - start_time
        fps = int(frame_count / total_time)

        print('fps: {}'.format(fps))
        print('frames: {}'.format(frame_count))
        print('running loss: {:.5f}'.format(running_loss))
        print('running rms: {:.5f}'.format(math.sqrt(running_loss)))

        if batch_no % 100 == 0:
            print('saving model')
            torch.save(model.state_dict(), args.model_path)

            recon_test(env, model)
