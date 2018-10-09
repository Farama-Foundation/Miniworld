import numpy as np
import math
from ..miniworld import MiniWorldEnv, Room
from ..entity import Box

class HallwayEnv(MiniWorldEnv):
    def __init__(self, length=12, **kwargs):
        assert length >= 2
        self.length = length

        super().__init__(
            max_episode_steps=250,
            **kwargs
        )

    def _gen_world(self):
        # Create a long rectangular room
        room = self.add_rect_room(
            min_x=-1, max_x=-1 + self.length,
            min_z=-2, max_z=2
        )

        room.entities.append(Box([room.max_x - 0.5, 0, 0], 0, color='red'))

        # Place the agent a random distance away from the goal
        self.agent.pos = np.array([
            self.rand.float(room.min_x + 0.5, room.max_x - 0.51),
            0,
            self.rand.float(-0.5, 0.5)
        ])

        self.agent.dir = self.rand.float(-math.pi/4, math.pi/4)

    def step(self, action):
        obs, reward, done, info = super().step(action)

        room = self.rooms[0]
        x, _, z = self.agent.pos

        if x > room.max_x - 0.5 and x < room.max_x:
            if z > room.min_z and z < room.max_z:
                reward += self._reward()
                done = True

        return obs, reward, done, info
