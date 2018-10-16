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
import numpy as np
import gym
import gym_miniworld

parser = argparse.ArgumentParser()
parser.add_argument('--env-name', default='MiniWorld-Hallway-v0')
parser.add_argument('--domain-rand', action='store_true', help='enable domain randomization')
parser.add_argument('--no-time-limit', action='store_true', help='ignore time step limits')
parser.add_argument('--high-fps', action='store_true', help='run at a higher frame rate for smooth motion')
args = parser.parse_args()

env = gym.make(args.env_name)
env.reset()

if args.no_time_limit:
    env.max_episode_steps = math.inf
if args.high_fps:
    env.max_episode_steps *= 30 / env.frame_rate
    env.frame_rate = 30

# Create the display window
env.render('pyglet')

@env.unwrapped.window.event
def on_key_press(symbol, modifiers):
    """
    This handler processes keyboard commands that
    control the simulation
    """

    if symbol == key.BACKSPACE or symbol == key.SLASH:
        print('RESET')
        env.reset()
        env.render('pyglet')
        return

    if symbol == key.ESCAPE:
        env.close()
        sys.exit(0)

    action = None
    if symbol == key.LEFT:
        action = env.actions.turn_left
    elif symbol == key.RIGHT:
        action = env.actions.turn_right
    elif symbol == key.UP:
        action = env.actions.move_forward
    elif symbol == key.DOWN:
        action = env.actions.move_back

    if action != None:
        obs, reward, done, info = env.step(action)
        #print('step_count = %s, reward=%.2f' % (env.unwrapped.step_count, reward))

        if done:
            print('done! reward={:.2f}'.format(reward))
            env.reset()
            env.render('pyglet')

        env.render('pyglet')

# Enter main event loop
pyglet.app.run()

env.close()
