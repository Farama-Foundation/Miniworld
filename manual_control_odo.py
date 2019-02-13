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
from experiments.pos_delta_6chan import Model
from experiments.utils import make_var, save_img
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--env-name', default='MiniWorld-SimToRealOdo2-v0')
parser.add_argument('--domain-rand', action='store_true', help='enable domain randomization')
parser.add_argument('--no-time-limit', action='store_true', default=True, help='ignore time step limits')
parser.add_argument('--save-imgs', action='store_true', help='save images')
args = parser.parse_args()

img_idx = 0

poss = []

env = gym.make(args.env_name)

if args.no_time_limit:
    env.max_episode_steps = math.inf
if args.domain_rand:
    env.domain_rand = True

def reset_env():
    global prev_obs
    global poss
    prev_obs = env.reset()
    dir = env.agent.dir if hasattr(env, 'agent') else 0
    poss = [(np.array([0, 0, 0]), dir)]

reset_env()

# Create the display window
env.render('pyglet')

model = Model()
model.load_state_dict(torch.load('pos_delta.torch'))
model.eval()
model.cuda()

def angle_to_dv(a):
    """
    Vector pointing in the direction of forward movement
    """

    x = math.cos(a)
    z = -math.sin(a)
    return np.array([x, 0, z])

def angle_to_rv(a):
    """
    Vector pointing to the right of the agent
    """

    x = math.sin(a)
    z = math.cos(a)
    return np.array([x, 0, z])

fig = plt.figure()
axe = fig.add_subplot(111)
axe.set_xlabel('dz')
axe.set_ylabel('dx')
sp, = axe.plot([],[], label='toto', markersize=4, color='k', marker='o', ls='')
fig.show()
fig.canvas.draw()

def step(action):
    global prev_obs
    global img_idx
    global poss

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

    # Plot the motion in 2D
    p, a = poss[-1]
    d = angle_to_dv(a)
    r = angle_to_rv(a)

    dd, dr, da = posd
    p = p + d*dd + r*dr
    a += da
    poss += [(p, a)]

    xs = [ p[2] for p,a in poss ]
    ys = [ p[0] for p,a in poss ]
    sp.set_data(xs, ys)
    axe.set_xlim(min(min(xs), -1), max(max(xs), 1))
    axe.set_ylim(min(min(ys), -1), max(max(ys), 1))
    fig.canvas.draw()

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
