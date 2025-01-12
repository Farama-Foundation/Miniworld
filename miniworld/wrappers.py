from typing import Optional

import gymnasium as gym
import numpy as np


class PyTorchObsWrapper(gym.ObservationWrapper):
    """
    Transpose the observation image tensors for PyTorch
    """

    def __init__(self, env):
        super().__init__(env)

        obs_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[2], obs_shape[1], obs_shape[0]],
            dtype=self.observation_space.dtype,
        )

    def observation(self, observation):
        return observation.transpose(2, 1, 0)


class GreyscaleWrapper(gym.ObservationWrapper):
    """
    Convert image observations from RGB to greyscale
    """

    def __init__(self, env):
        super().__init__(env)

        obs_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            (obs_shape[0], obs_shape[1], 1),
            dtype=self.observation_space.dtype,
        )

    def observation(self, obs):
        obs = 0.30 * obs[:, :, 0] + 0.59 * obs[:, :, 1] + 0.11 * obs[:, :, 2]

        return np.expand_dims(obs, axis=2)


class StochasticActionWrapper(gym.ActionWrapper):
    """
    Add stochasticity to the actions

    If a random action is provided, it is returned with probability `1 - prob`.
    Else, a random action is sampled from the action space.
    """

    def __init__(self, env, prob: float = 0.9, random_action: Optional[int] = None):
        super().__init__(env)

        self.prob = prob
        self.random_action = random_action

    def action(self, action):
        """ """
        if self.np_random.uniform() < self.prob:
            return action
        else:
            if self.random_action is None:
                return self.np_random.integers(0, 6)
            else:
                return self.random_action
