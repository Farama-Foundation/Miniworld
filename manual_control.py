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

parser = argparse.ArgumentParser()
parser.add_argument('--env-name', default='MiniWorld-Hallway-v0')
parser.add_argument('--domain-rand', action='store_true', help='enable domain randomization')
parser.add_argument('--no-time-limit', action='store_true', help='ignore time step limits')
args = parser.parse_args()

env = gym.make(args.env_name)
env.reset()

if args.no_time_limit:
    env.max_episode_steps = math.inf
if args.domain_rand:
    env.domain_rand = True

# Create the display window
env.render('pyglet')

def step(dt, action, n=0, repeat=True):
    print('step {}: {}'.format(env.step_count, env.actions(action).name))

    obs, reward, done, info = env.step(action)
    #print('step_count = %s, reward=%.2f' % (env.unwrapped.step_count, reward))

    if done:
        print('done! reward={:.2f}'.format(reward))
        clock.unschedule(step)
        env.reset()

    env.render('pyglet')

    if repeat and not done:
        if n == 0:
            clock.schedule_once(step, 0.5, action=action, n=n+1)
        else:
            clock.schedule_once(step, 0.08, action=action, n=n+1)

@env.unwrapped.window.event
def on_key_press(symbol, modifiers):
    """
    This handler processes keyboard commands that
    control the simulation
    """

    clock.unschedule(step)

    if symbol == key.BACKSPACE or symbol == key.SLASH:
        print('RESET')
        env.reset()
        env.render('pyglet')
        return

    if symbol == key.ESCAPE:
        env.close()
        sys.exit(0)

    if symbol == key.UP:
        step(0, env.actions.move_forward)
    elif symbol == key.DOWN:
        step(0, env.actions.move_back)

    elif symbol == key.LEFT:
        step(0, env.actions.turn_left)
    elif symbol == key.RIGHT:
        step(0, env.actions.turn_right)

    elif symbol == key.PAGEUP or symbol == key.P:
        step(0, env.actions.pickup, repeat=False)
    elif symbol == key.PAGEDOWN or symbol == key.D:
        step(0, env.actions.drop, repeat=False)

@env.unwrapped.window.event
def on_key_release(symbol, modifiers):
    clock.unschedule(step)

@env.unwrapped.window.event
def on_draw():
    env.render('pyglet')

@env.unwrapped.window.event
def on_close():
    pyglet.app.exit()

# Enter main event loop
pyglet.app.run()

env.close()
