import numpy as np
import math
from ..miniworld import MiniWorldEnv, Room
from ..entity import Box

class FourRoomsEnv(MiniWorldEnv):
    """
    Two hallways connected in a T-junction
    """

    def __init__(self, **kwargs):
        super().__init__(
            max_episode_steps=500,
            frame_rate=6,
            **kwargs
        )

    def _gen_world(self):

        room1 = self.add_rect_room(
            min_x=-7, max_x=-1,
            min_z=-7, max_z=-1
        )
        room1.add_portal(3, min_x=-5, max_x=-3, max_y=2.2)
        room1.add_portal(0, min_z=-5, max_z=-3, max_y=2.2)

        room2 = self.add_rect_room(
            min_x=-7, max_x=-1,
            min_z=1, max_z=7
        )
        room2.add_portal(0, min_z=3, max_z=5, max_y=2.2)
        room2.add_portal(1, min_x=-5, max_x=-3, max_y=2.2)

        room3 = self.add_rect_room(
            min_x=1, max_x=7,
            min_z=-7, max_z=-1
        )
        room3.add_portal(2, min_z=-5, max_z=-3, max_y=2.2)
        room3.add_portal(3, min_x=3, max_x=5, max_y=2.2)

        room4 = self.add_rect_room(
            min_x=1, max_x=7,
            min_z=1, max_z=7
        )
        room4.add_portal(1, min_x=3, max_x=5, max_y=2.2)
        room4.add_portal(2, min_z=3, max_z=5, max_y=2.2)

        c0 = self.add_rect_room(
            min_x=-5, max_x=-3,
            min_z=-1, max_z=1,
            wall_height=2.2
        )
        c0.add_portal(1, min_x=-5, max_x=-3)
        c0.add_portal(3, min_x=-5, max_x=-3)

        c1 = self.add_rect_room(
            min_x=3, max_x=5,
            min_z=-1, max_z=1,
            wall_height=2.2
        )
        c1.add_portal(1, min_x=3, max_x=5)
        c1.add_portal(3, min_x=3, max_x=5)

        c2 = self.add_rect_room(
            min_x=-1, max_x=1,
            min_z=3, max_z=5,
            wall_height=2.2
        )
        c2.add_portal(0, min_z=3, max_z=5)
        c2.add_portal(2, min_z=3, max_z=5)

        c3 = self.add_rect_room(
            min_x=-1, max_x=1,
            min_z=-5, max_z=-3,
            wall_height=2.2
        )
        c3.add_portal(0, min_z=-5, max_z=-3)
        c3.add_portal(2, min_z=-5, max_z=-3)






        self.agent.pos[0] = -4
        self.agent.pos[2] = -4



    def step(self, action):
        obs, reward, done, info = super().step(action)

        return obs, reward, done, info
