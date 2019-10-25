#!/usr/bin/env python3

import os
import shutil
import time
import random
import pickle
import argparse
from functools import reduce
import operator

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision
import numpy as np

from imitation_model import Model
import gym
import gym_miniworld

##############################################################################

def make_var(arr):
    arr = np.ascontiguousarray(arr)
    #arr = torch.from_numpy(arr).float()
    arr = torch.from_numpy(arr)
    arr = Variable(arr)
    if torch.cuda.is_available():
        arr = arr.cuda()
    return arr

##############################################################################

parser = argparse.ArgumentParser()
parser.add_argument("--model-path", default="imitation_model.torch")
args = parser.parse_args()

model = Model()
model.cuda()
model.eval()

model.load_state_dict(torch.load(args.model_path))

env = gym.make('MiniWorld-OneRoomS6-v0')
obs = env.reset()

print('starting')

for i in range(0, 100000):
    action = model.sample_action(obs)

    #print(pos)
    obs, reward, done, info = env.step(action)

    env.render('human')
    time.sleep(0.05)

    if done:
        env.reset()
        env.render('human')
        time.sleep(0.05)
