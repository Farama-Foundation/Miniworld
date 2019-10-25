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
parser.add_argument('--env-name', default='MiniWorld-OneRoomS6-v0')
args = parser.parse_args()

env = gym.make(args.env_name)

env.reset()

# Create the display window
env.render('pyglet')

def step(action):
    global demo_frames
    global demo_actions
    global last_img

    print('step {}/{}: {}'.format(env.step_count+1, env.max_episode_steps, env.actions(action).name))

    demo_frames += [np.copy(last_img)]
    demo_actions += [action]

    obs, reward, done, info = env.step(action)

    print(obs.shape)

    last_img = np.copy(obs)

    if reward > 0:
        print('reward={:.2f}'.format(reward))

    if done:
        print('done!')

        save_demo()

        env.reset()

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
        #save_demo()
    elif symbol == key.PAGEDOWN or symbol == key.D:
        step(env.actions.drop)
        #drop_demo()

    elif symbol == key.ENTER:
        step(env.actions.done)

@env.unwrapped.window.event
def on_key_release(symbol, modifiers):
    pass

@env.unwrapped.window.event
def on_draw():
    env.render('pyglet')

@env.unwrapped.window.event
def on_close():
    pyglet.app.exit()

demo_frames = []
demo_actions = []
total_frames = []
total_actions = []
num_demos = 0

def save_demo():
    import pickle
    global demo_frames
    global demo_actions
    global total_frames
    global total_actions
    global num_demos

    total_frames += demo_frames
    total_actions += demo_actions
    num_demos += 1

    print('Saving demo, len={}, total frames={}, num demos={}'.format(len(demo_frames), len(total_frames), num_demos))

    obj = {
        'frames': total_frames,
        'actions': total_actions
    }
    pickle.dump(obj, open("demos.pkl", "wb"))

    demo_frames = []
    demo_actions = []
    env.reset()

def drop_demo():
    global demo_frames
    global demo_actions
    global cur_arm_pos
    print('Dropping demo')
    demo_frames = []
    demo_actions = []
    env.reset()

last_img = np.zeros(shape=(60, 80, 3), dtype=np.uint8)
last_img[:, :, 0] = 255

# Enter main event loop
pyglet.app.run()

env.close()
