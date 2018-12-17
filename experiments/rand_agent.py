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
        self.action_probs = nn.Sequential(
            nn.Linear(128, num_actions),
            nn.LeakyReLU(),
            nn.LogSoftmax(dim=1)
        )

        self.apply(init_weights)

    def predict_action(self, img, memory):
        batch_size = img.size(0)

        #x = img.view(batch_size, -1)
        x = self.encoder(img)

        memory = self.rnn(x, memory)
        action_probs = self.action_probs(memory)
        dist = Categorical(logits=action_probs)

        return dist, memory

##############################################################################

env = gym.make('MiniWorld-Hallway-v0')

num_actions = env.action_space.n
print('num actions:', num_actions)

max_steps = env.max_episode_steps
print('max episode steps:', max_steps)

def evaluate(model, seed=0, num_episodes=100):
    env = gym.make('MiniWorld-Hallway-v0')

    num_success = 0

    env.seed(seed)

    for i in range(num_episodes):
        #print(i)

        obs = env.reset()

        memory = Variable(torch.zeros([1, 128])).cuda()

        while True:

            obs = obs.transpose(2, 0, 1)
            obs = make_var(obs).unsqueeze(0)

            dist, memory = model.predict_action(obs, memory)
            action = dist.sample()

            obs, reward, done, info = env.step(action)

            if done:
                if reward > 0:
                    #print('success')
                    num_success += 1
                break

    return num_success / num_episodes



best_score = 0

for i in range(500):
    model = Model(num_actions)
    model.cuda()

    s = evaluate(model)

    print('#{}: {:.2f}'.format(i+1, s))

    if s > best_score:
        best_score = s
        print('new best score: {:.2f}'.format(s))


    del model
    torch.cuda.empty_cache()



# TODO; start with 10 random models, evaluate them
# perform reinforce based on

# TODO: use SGD optimizer

# TODO: gather off-policy experience
