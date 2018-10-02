import numpy as np
import math
from ..miniworld import MiniWorldEnv, Room
from ..entity import Box

class JunctionEnv(MiniWorldEnv):
    def __init__(self, **kwargs):
        super().__init__(
            max_episode_steps=600,
            **kwargs
        )

    def _gen_world(self):
        room1 = self.create_rect_room(
            -1, -2,
            10,  4
        )
        room2 = self.create_rect_room(
            9, -10,
            4, 20
        )

        room1.make_portal(0, 0, 4)
        room2.make_portal(2, 8, 4)

        self.agent.dir = self.rand.float(-math.pi/4, math.pi/4)

    def step(self, action):
        obs, reward, done, info = super().step(action)

        x, _, z = self.agent.pos

        # TODO: reward for reaching red box?
        # position intersection function for entities?
        # Entity.pos_inside(p)?

        return obs, reward, done, info
