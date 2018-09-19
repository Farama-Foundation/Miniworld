import math
import numpy as np
import gym

class PyTorchObsWrapper(gym.ObservationWrapper):
    """
    Transpose the observation image tensors for PyTorch
    """

    def __init__(self, env=None):
        super().__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            self.observation_space.low[0,0,0],
            self.observation_space.high[0,0,0],
            [obs_shape[2], obs_shape[1], obs_shape[0]],
            dtype=self.observation_space.dtype
        )

    def observation(self, observation):
        return observation.transpose(2, 1, 0)

class GreyscaleWrapper(gym.ObservationWrapper):
    """
    Convert image obserations from RGB to greyscale
    """

    def __init__(self, env=None):
        super().__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            self.observation_space.low[0,0,0],
            self.observation_space.high[0,0,0],
            [obs_shape[0], obs_shape[1], 1],
            dtype=self.observation_space.dtype
        )

    def observation(self, obs):
        obs = (
            0.30 * obs[:,:,0] +
            0.59 * obs[:,:,1] +
            0.11 * obs[:,:,2]
        )

        return np.expand_dims(obs, axis=2)
