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
    elif symbol == key.ESCAPE:
        env.close()
        sys.exit(0)

# Register a keyboard handler
key_handler = key.KeyStateHandler()
env.unwrapped.window.push_handlers(key_handler)

def update(dt):
    """
    This function is called at every frame to handle
    movement/stepping and redrawing
    """

    action = env.actions.do_nothing

    if key_handler[key.UP]:
        action = env.actions.move_forward
    if key_handler[key.DOWN]:
        action = env.actions.move_back
    if key_handler[key.LEFT]:
        action = env.actions.turn_left
    if key_handler[key.RIGHT]:
        action = env.actions.turn_right

    obs, reward, done, info = env.step(action)
    #print('step_count = %s, reward=%.2f' % (env.unwrapped.step_count, reward))

    if done:
        print('done! reward={:.2f}'.format(reward))
        env.reset()
        env.render('pyglet')

    env.render('pyglet')

pyglet.clock.schedule_interval(update, 1 / env.unwrapped.frame_rate)

# Enter main event loop
pyglet.app.run()

env.close()
