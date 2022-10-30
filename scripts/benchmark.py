#!/usr/bin/env python3

import time

import gym_miniworld
import gymnasium as gym

# Benchmark loading time
st = time.time()
env = gym.make("MiniWorld-Maze-v0")
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
    obs, reward, termination, truncation, info = env.step(action)

    if termination or truncation:
        env.reset()

    num_frames += 1

fps = num_frames / dt
frame_time = 1000 * dt / num_frames

print()
print(f"load time: {int(load_time)} ms")
print(f"reset time: {reset_time:,.1f} ms")
print(f"frame time: {frame_time:,.1f} ms")
print(f"frame rate: {fps:,.1f} FPS")

env.close()
