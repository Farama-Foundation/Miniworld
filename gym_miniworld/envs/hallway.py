import numpy as np
import math
from ..miniworld import MiniWorldEnv, Room

class HallwayEnv(MiniWorldEnv):
    def __init__(self, length=12):
        assert length >= 1
        self.length = length
        super().__init__()

    def _gen_world(self):

        room = self.create_rect_room(
            -1, -2,
            1 + self.length, 4
        )

        self.agent.position = np.array([
            self.rand.float(-0.5, 0.5),
            0,
            self.rand.float(-0.5, 0.5)
        ])

        self.agent.direction = self.rand.float(-math.pi/4, math.pi/4)

    def step(self, action):
        obs, reward, done, info = super().step(action)

        # TODO: reward computation



        return obs, reward, done, info
