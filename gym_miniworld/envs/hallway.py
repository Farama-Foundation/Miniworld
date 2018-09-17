import numpy as np
import math
from ..miniworld import MiniWorldEnv, Room
from ..entity import Box

class HallwayEnv(MiniWorldEnv):
    def __init__(self, length=12, **kwargs):
        assert length >= 1
        self.length = length

        super().__init__(
            max_episode_steps=300,
            **kwargs
        )

    def _gen_world(self):

        room = self.create_rect_room(
            -1, -2,
            1 + self.length, 4
        )

        room.entities.append(Box(room.max_x - 0.5, 0, 0, color='red'))

        # Place the agent a random distance away from the goal
        self.agent.position = np.array([
            self.rand.float(room.min_x + 0.5, room.max_x - 0.51),
            0,
            self.rand.float(-0.5, 0.5)
        ])

        self.agent.direction = self.rand.float(-math.pi/4, math.pi/4)

    def step(self, action):
        obs, reward, done, info = super().step(action)

        room = self.rooms[0]
        x, _, z = self.agent.position
        dx, _, _ = self.agent.dir_vec

        reward = max(dx, 0)

        if action == self.actions.move_forward:
            reward += 0.5

        if x > room.max_x - 0.5 and x < room.max_x:
            if z > room.min_z and z < room.max_z:
                reward += 1000
                done = True

        return obs, reward, done, info
