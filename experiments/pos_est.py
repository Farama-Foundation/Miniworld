#!/usr/bin/env python3

import math
from functools import reduce
import operator
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions.categorical import Categorical

import gym
import gym_miniworld
from gym_miniworld.wrappers import *

from utils import *

class Model(nn.Module):
    def __init__(self, num_actions):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),

            nn.Conv2d(32, 32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),

            nn.Conv2d(32, 32, kernel_size=4, stride=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),

            Flatten(),
            #Print(),

            nn.Linear(1120, 128),
            nn.LeakyReLU(),
        )

        self.rnn = nn.GRUCell(input_size=128, hidden_size=128)

        # GRU embedding to action
        self.enc_to_pos = nn.Sequential(
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 3),
        )

        self.apply(init_weights)

    def forward(self, img, memory):
        batch_size = img.size(0)

        x = self.encoder(img)
        memory = self.rnn(x, memory)
        pos = self.enc_to_pos(memory)

        return pos, memory

##############################################################################

env = gym.make('MiniWorld-OneRoom-v0')

num_actions = env.action_space.n
print('num actions:', num_actions)

max_steps = env.max_episode_steps
print('max episode steps:', max_steps)

max_demos = 8192
num_demos = 0
cur_idx = 0

# Done indicates that we become done after the current step
obss = np.zeros(shape=(max_demos, max_steps, 3, 60, 80), dtype=np.uint8)
poss = np.zeros(shape=(max_demos, max_steps, 3), dtype=np.float32)
active = np.zeros(shape=(max_demos, max_steps, 1), dtype=np.float32)

def gen_trajs(num_episodes=128):
    global cur_idx, num_demos

    for i in range(num_episodes):
        #print(i)

        active[cur_idx, :] = 0

        obs = env.reset()

        start_pos = env.agent.pos

        for step_idx in range(max_steps):
            obs = obs.transpose(2, 0, 1)

            pos = env.agent.pos
            rel_pos = pos - start_pos

            obss[cur_idx, step_idx] = obs
            poss[cur_idx, step_idx] = rel_pos
            active[cur_idx, step_idx] = 1

            # TODO: bias towards forward movement
            # or repetition of actions for multiple time steps
            action = np.random.randint(0, env.actions.move_forward+1)

            obs, reward, done, info = env.step(action)

            if done:
                break

        cur_idx = (cur_idx + 1) % max_demos
        num_demos = max(num_demos, cur_idx)


model = Model(num_actions)
model.cuda()

optimizer = optim.Adam(model.parameters(), lr=1e-4)

batch_size = 32

running_loss = 0

gen_trajs(batch_size)

# For each batch
for batch_idx in range(50000):
    print('batch #{}'.format(batch_idx+1))

    # Select a valid demo index in function of the batch size
    demo_idx = np.random.randint(0, num_demos - batch_size + 1)

    print('num_demos={}'.format(num_demos))
    print('demo_idx={}'.format(demo_idx))

    # Get the observations, actions and done flags for this batch
    obs_batch = obss[demo_idx:(demo_idx+batch_size)]
    pos_batch = poss[demo_idx:(demo_idx+batch_size)]
    active_batch = active[demo_idx:(demo_idx+batch_size)]

    obs_batch = make_var(obs_batch)
    pos_batch = Variable(torch.from_numpy(pos_batch)).cuda()
    active_batch = make_var(active_batch)

    # Create initial memory for the model
    memory = Variable(torch.zeros([batch_size, 128])).cuda()

    total_loss = 0
    total_steps = 0

    # For each step
    # We will iterate until the max demo len (or until all demos are done)
    for step_idx in range(max_steps-1):
        active_step = active_batch[:, step_idx, :]

        if active_step.sum().item() == 0:
            print('break at step', step_idx)
            break

        obs_step = obs_batch[:, step_idx]
        pos_step = pos_batch[:, step_idx]

        pos_out, memory = model(obs_step, memory)

        pos_loss = (pos_out - pos_step)
        pos_loss = (pos_loss * active_step).abs().sum()
        total_loss += pos_loss
        total_steps += active_step.sum().item()

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    # Generate new data
    gen_trajs(4)

    mean_loss = total_loss.item() / total_steps

    if batch_idx == 0:
        running_loss = mean_loss
    else:
        running_loss = running_loss * 0.99 + mean_loss * 0.01

    print('{:.4f}'.format(running_loss))
