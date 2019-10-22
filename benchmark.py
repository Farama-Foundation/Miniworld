#!/usr/bin/env python3

import time
import numpy as np
import gym
import gym_miniworld

# Benchmark loading time
st = time.time()
env = gym.make('MiniWorld-Maze-v0')
env.seed(0)
env.reset()
load_time = 1000 * (time.time() - st)

# Benchmark the reset time
st = time.time()
for i in range(100):
    env.reset()
reset_time = 1000 * (time.time() - st) / 100

# Benchmark the rendering/update speed
num_frames = 0
st = time.time()

while True:
    dt = time.time() - st

    if dt > 5:
        break

    # Slow movement speed to minimize resets
    action = 0
    obs, reward, done, info = env.step(action)

    if done:
        env.reset()

    num_frames += 1

fps = num_frames / dt
frame_time = 1000 * dt / num_frames

print()
print('load time: {} ms'.format(int(load_time)))
print('reset time: {:,.1f} ms'.format(reset_time))
print('frame time: {:,.1f} ms'.format(frame_time))
print('frame rate: {:,.1f} FPS'.format(fps))

env.close()
