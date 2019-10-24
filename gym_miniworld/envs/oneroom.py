import numpy as np
import math
from ..miniworld import MiniWorldEnv, Room
from ..entity import Box
from ..params import DEFAULT_PARAMS
from gym import spaces

class OneRoom(MiniWorldEnv):
    """
    Environment in which the goal is to go to a red box
    placed randomly in one big room.
    """

    def __init__(self, size=10, max_episode_steps=180, **kwargs):
        assert size >= 2
        self.size = size

        super().__init__(
            max_episode_steps=max_episode_steps,
            **kwargs
        )

        # Allow only movement actions (left/right/forward)
        self.action_space = spaces.Discrete(self.actions.move_forward+1)

    def _gen_world(self):
        room = self.add_rect_room(
            min_x=0,
            max_x=self.size,
            min_z=0,
            max_z=self.size
        )

        self.box = self.place_entity(Box(color='red'))
        self.place_agent()

    def step(self, action):
        obs, reward, done, info = super().step(action)

        if self.near(self.box):
            reward += self._reward()
            done = True

        return obs, reward, done, info

class OneRoomS6(OneRoom):
    def __init__(self, max_episode_steps=100, **kwargs):
        super().__init__(size=6, max_episode_steps=max_episode_steps, **kwargs)

class OneRoomS6Fast(OneRoomS6):
    def __init__(self, forward_step=0.7, turn_step=45):
        # Parameters for larger movement steps, fast stepping
        params = DEFAULT_PARAMS.no_random()
        params.set('forward_step', forward_step)
        params.set('turn_step', turn_step)

        super().__init__(
            max_episode_steps=50,
            params=params,
            domain_rand=False
        )
