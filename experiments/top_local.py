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

            #Print(),
            Flatten(),
        )

        self.map_to_enc = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=6, stride=3),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),

            nn.Conv2d(32, 32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),

            nn.Conv2d(32, 32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),

            nn.Conv2d(32, 32, kernel_size=4, stride=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),

            #Print(),
            Flatten(),
        )

        self.enc_size = 2276
        self.dec_size = 1152

        self.enc_to_emb = nn.Sequential(
            nn.Linear(self.enc_size, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.Linear(512, self.dec_size),
            nn.LeakyReLU(),
        )

        self.decoder = nn.Sequential(
            #Print(),

            nn.ConvTranspose2d(32, 32, kernel_size=6, stride=3),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(32, 32, kernel_size=6, stride=3),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(32, 32, kernel_size=5, stride=3),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=1),
            nn.LeakyReLU(),
        )

        self.apply(init_weights)

    def forward(self, obs, input_map, input_pos):
        obs = obs / 255
        input_map = input_map / 255

        obs_enc = self.obs_to_enc(obs)
        map_enc = self.map_to_enc(input_map)

        enc = torch.cat((obs_enc, map_enc, input_pos), dim=1)
        #print(enc.size())

        emb = self.enc_to_emb(enc)
        emb = emb.reshape(-1, 32, 6, 6)

        output_map = self.decoder(emb)
        output_map = 255 * output_map[:, :, 1:201, 1:201]

        return output_map

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
input_obs = np.zeros(shape=(args.buffer_size, 3, 80, 60), dtype=np.uint8)
input_pos = np.zeros(shape=(args.buffer_size, 4), dtype=np.float32)
input_map = np.zeros(shape=(args.buffer_size,) + map_shape, dtype=np.float32)
output_map = np.zeros(shape=(args.buffer_size,) + map_shape, dtype=np.float32)

model = Model()
model.cuda()
print_model_info(model)

optimizer = optim.Adam(model.parameters(), lr=1e-4)

def gen_data(num_episodes=1):
    global cur_idx, num_trans

    for i in range(num_episodes):
        #print(i)

        obs = env.reset()

        cur_action = None
        steps_left = 0

        # Initial overhead map
        map = make_var(np.zeros(shape=map_shape)).unsqueeze(0)

        for step_idx in range(max_steps):
            obs = obs.transpose(2, 1, 0)

            pos = env.agent.pos
            angle = env.agent.dir
            pos = np.array([*pos] + [angle])

            # Pick a random transition index. Prioritize expanding the set.
            if num_trans < args.buffer_size and np.random.uniform(0, 1) < 0.5:
                cur_idx = num_trans
            else:
                cur_idx = np.random.randint(0, num_trans + 1) % args.buffer_size
            num_trans = max(num_trans, cur_idx+1)

            input_obs[cur_idx] = obs
            input_pos[cur_idx] = pos
            input_map[cur_idx] = map.cpu().numpy()

            # Generate output map, store it
            top_view = env.render_top_view(env.vis_fb).transpose(2, 1, 0)
            output_map[cur_idx] = top_view

            with torch.no_grad():
                obs = make_var(obs).unsqueeze(0)
                pos = make_var(pos).unsqueeze(0)
                map = model(obs, map, pos)

            # Repeat turn_left, turn_right or move_forward for N steps
            if steps_left == 0:
                cur_action = np.random.choice([
                    env.actions.turn_left,
                    env.actions.turn_right,
                    env.actions.move_forward]
                )
                steps_left = np.random.randint(1, 17)

            obs, reward, done, info = env.step(cur_action)
            steps_left -= 1

            if done:
                break

while num_trans <= args.batch_size:
    gen_data()

running_loss = None

for i in range(1000000):
    print('batch #{} (num trans={})'.format(i+1, num_trans))

    batch_idx = np.random.randint(0, num_trans - args.batch_size)

    batch_obs = make_var(input_obs[batch_idx:(batch_idx+args.batch_size)])
    batch_pos = make_var(input_pos[batch_idx:(batch_idx+args.batch_size)])
    batch_in_map = make_var(input_map[batch_idx:(batch_idx+args.batch_size)])
    batch_out_map = make_var(output_map[batch_idx:(batch_idx+args.batch_size)])

    pred_out_map = model(batch_obs, batch_in_map, batch_pos)

    # Generate data while the GPU is computing
    gen_data()

    optimizer.zero_grad()
    diff = pred_out_map - batch_out_map
    loss = (diff * diff).mean() # L2 loss
    loss.backward()
    optimizer.step()

    if i == 0:
        running_loss = loss.data.item()
    else:
        running_loss = 0.99 * running_loss + 0.01 * loss.data.item()

    print('running loss: {:.1f}'.format(running_loss))

    if i % 100 == 0:
        print('saving images and model')

        for img_idx in range(20):
            save_img('img_{:02d}_img.png'.format(img_idx), batch_obs[img_idx])
            save_img('img_{:02d}_map.png'.format(img_idx), batch_out_map[img_idx])
            save_img('img_{:02d}_pred.png'.format(img_idx), pred_out_map[img_idx])

        torch.save(model.state_dict(), 'map_gen_model.torch')
