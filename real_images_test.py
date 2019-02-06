#!/usr/bin/env python3

"""
This script allows you to manually control the simulator
using the keyboard arrows.
"""

import sys
import argparse
import pyglet
import math
from pyglet.window import key
from pyglet import clock
import numpy as np
import gym
import gym_miniworld

import torch
from experiments.pos_delta import Model
from experiments.utils import make_var, load_img, save_img

"""
parser = argparse.ArgumentParser()
parser.add_argument('--env-name', default='MiniWorld-SimToRealOdo-v0')
parser.add_argument('--domain-rand', action='store_true', help='enable domain randomization')
parser.add_argument('--no-time-limit', action='store_true', help='ignore time step limits')
parser.add_argument('--save-imgs', action='store_true', help='save images')
args = parser.parse_args()
"""

model = Model()
model.load_state_dict(torch.load('pos_delta.torch'))
model.eval()
model.cuda()

for img_idx in range(29):
    img0 = load_img('robot_imgs/img_{:03d}.png'.format(img_idx))
    img1 = load_img('robot_imgs/img_{:03d}.png'.format(img_idx+1))

    posd = model(img0, img0)
    posd = posd.squeeze().cpu().detach().numpy()
    print('{:+.3f} {:+.3f} {:+.3f}'.format(*posd))
