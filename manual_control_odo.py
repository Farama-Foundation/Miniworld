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
from experiments.utils import make_var, save_img

parser = argparse.ArgumentParser()
parser.add_argument('--env-name', default='MiniWorld-SimToRealOdo-v0')
parser.add_argument('--domain-rand', action='store_true', help='enable domain randomization')
parser.add_argument('--no-time-limit', action='store_true', help='ignore time step limits')
parser.add_argument('--save-imgs', action='store_true', help='save images')
args = parser.parse_args()

img_idx = 0

env = gym.make(args.env_name)

if args.no_time_limit:
    env.max_episode_steps = math.inf
if args.domain_rand:
    env.domain_rand = True

def reset_env():
    global prev_obs
    prev_obs = env.reset()

reset_env()

# Create the display window
env.render('pyglet')

model = Model()
model.load_state_dict(torch.load('pos_delta.torch'))
model.eval()
model.cuda()

def step(action):
    global prev_obs
    global img_idx

    print('step {}: {}'.format(env.step_count, env.actions(action).name))

    if hasattr(env, 'agent'):
        prev_pos = env.agent.pos
        prev_dir = env.agent.dir
        prev_dv = env.agent.dir_vec
        prev_rv = env.agent.right_vec

    obs, reward, done, info = env.step(action)

    obs0 = make_var(prev_obs.transpose(2, 1, 0)).unsqueeze(0)
    obs1 = make_var(obs.transpose(2, 1, 0)).unsqueeze(0)
    posd = model(obs0, obs1)
    posd = posd.squeeze().cpu().detach().numpy()

    print('{:+.3f} {:+.3f} {:+.3f}'.format(*posd))

    if hasattr(env, 'agent'):
        delta_dir = env.agent.dir - prev_dir
        delta_dv = np.dot(env.agent.pos - prev_pos, prev_dv)
        delta_rv = np.dot(env.agent.pos - prev_pos, prev_rv)
        print('{:+.3f} {:+.3f} {:+.3f}'.format(delta_dv, delta_rv, delta_dir))

    print()

    prev_obs = obs

    if args.save_imgs:
        save_img('img_{:03d}.png'.format(img_idx), obs)
        img_idx += 1

    if done:
        print('done! reward={:.2f}'.format(reward))
        reset_env()

    env.render('pyglet')

@env.unwrapped.window.event
def on_key_press(symbol, modifiers):
    """
    This handler processes keyboard commands that
    control the simulation
    """

    if symbol == key.BACKSPACE or symbol == key.SLASH:
        print('RESET')
        reset_env()
        env.render('pyglet')
        return

    if symbol == key.ESCAPE:
        env.close()
        sys.exit(0)

    if symbol == key.UP:
        step(env.actions.move_forward)
    elif symbol == key.DOWN:
        step(env.actions.move_back)

    elif symbol == key.LEFT:
        step(env.actions.turn_left)
    elif symbol == key.RIGHT:
        step(env.actions.turn_right)

    elif symbol == key.PAGEUP or symbol == key.P:
        step(env.actions.pickup)
    elif symbol == key.PAGEDOWN or symbol == key.D:
        step(env.actions.drop)

@env.unwrapped.window.event
def on_key_release(symbol, modifiers):
    pass

@env.unwrapped.window.event
def on_draw():
    env.render('pyglet')

@env.unwrapped.window.event
def on_close():
    pyglet.app.exit()

# Enter main event loop
pyglet.app.run()

env.close()
