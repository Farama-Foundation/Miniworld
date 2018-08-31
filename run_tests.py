#!/usr/bin/env python3

import os
import numpy as np
import gym
import gym_miniworld
#from gym_miniworld.envs import HallwayEnv
#from gym_miniworld.wrappers import PyTorchObsWrapper

env = gym.make('MiniWorld-Hallway-v0')

# Try stepping a few times
for i in range(0, 10):
    obs, _, _, _ = env.step(0)

"""
# Check that the human rendering resembles the agent's view
first_obs = env.reset()
first_render = env.render('rgb_array')
m0 = first_obs.mean()
m1 = first_render.mean()
assert m0 > 0 and m0 < 255
assert abs(m0 - m1) < 5

# Check that the observation shapes match
second_obs, _, _, _ = env.step([0.0, 0.0])
assert first_obs.shape == env.observation_space.shape
assert first_obs.shape == second_obs.shape

# Test the PyTorch observation wrapper
env = PyTorchObsWrapper(env)
first_obs = env.reset()
second_obs, _, _, _ = env.step([0, 0])
assert first_obs.shape == env.observation_space.shape
assert first_obs.shape == second_obs.shape
"""
