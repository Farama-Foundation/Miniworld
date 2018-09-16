import numpy as np
import math
from ..miniworld import MiniWorldEnv, Room
from ..entity import CeilingLight

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

        """
        room.entities.append(CeilingLight(
            1 + self.length - 2,
            room.wall_height,
            0
        ))
        """

        self.goal_pos = np.array([room.max_x - 0.5, 0, 0])

        # Place the agent a random distance away from the goal
        self.agent.position = np.array([
            self.rand.float(room.min_x + 0.5, self.goal_pos[0] - 0.25),
            0,
            self.rand.float(-0.5, 0.5)
        ])

        self.agent.direction = self.rand.float(-math.pi/4, math.pi/4)

    def step(self, action):
        obs, reward, done, info = super().step(action)

        dist = np.linalg.norm(self.agent.position - self.goal_pos)

        if dist < 0.25:
            reward = 1000 - self.step_count
            done = True

        return obs, reward, done, info
