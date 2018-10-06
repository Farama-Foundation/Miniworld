import numpy as np
import math
from ..miniworld import MiniWorldEnv, Room
from ..entity import Box

class TMazeEnv(MiniWorldEnv):
    """
    Two hallways connected in a T-junction
    """

    def __init__(self, **kwargs):
        super().__init__(
            max_episode_steps=600,
            **kwargs
        )

    def _gen_world(self):
        room1 = self.add_rect_room(
            -1, 10,
            -2, 2
        )
        room2 = self.add_rect_room(
            10, 14,
            -10, 10
        )
        room1.add_portal(0, 0, 4)
        room2.add_portal(2, 8, 4)

        self.agent.dir = self.rand.float(-math.pi/4, math.pi/4)

        # Add a box at a random end of the hallway
        z_pos = self.rand.elem([room2.min_z + 0.5, room2.max_z - 0.5])
        self.box = Box([room2.mid_x, 0, z_pos], 0, color='red')
        room2.entities.append(self.box)

    def step(self, action):
        obs, reward, done, info = super().step(action)

        # TODO: proper intersection test method for entities
        # Entity.pos_inside(p)?
        dist = np.linalg.norm(self.agent.pos - self.box.pos)
        if dist < self.box.size:
            reward += 1.0 - 0.2 * (self.step_count / self.max_episode_steps)
            done = True

        return obs, reward, done, info
