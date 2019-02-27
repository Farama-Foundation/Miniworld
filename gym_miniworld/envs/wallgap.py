import numpy as np
import math
from gym import spaces
from ..miniworld import MiniWorldEnv, Room
from ..entity import Box, ImageFrame, MeshEnt

class WallGap(MiniWorldEnv):
    """
    Outside environment with two rooms connected by a gap in a wall
    """

    def __init__(self, **kwargs):
        super().__init__(
            max_episode_steps=300,
            **kwargs
        )

        # Allow only the movement actions
        self.action_space = spaces.Discrete(self.actions.move_forward+1)

    def _gen_world(self):
        # Top
        room0 = self.add_rect_room(
            min_x=-7, max_x=7,
            min_z=0.5 , max_z=8,
            wall_tex='brick_wall',
            floor_tex='asphalt',
            no_ceiling=True
        )
        # Bottom
        room1 = self.add_rect_room(
            min_x=-7, max_x=7,
            min_z=-8, max_z=-0.5,
            wall_tex='brick_wall',
            floor_tex='asphalt',
            no_ceiling=True
        )
        self.connect_rooms(room0, room1, min_x=-1.5, max_x=1.5)

        self.box = self.place_entity(Box(color='red'), room=room1)

        # Decorative building in the background
        self.place_entity(
            MeshEnt(
                mesh_name='building',
                height=30
            ),
            pos = np.array([30, 0, 30]),
            dir = -math.pi
        )

        self.place_agent(room=room0)

    def step(self, action):
        obs, reward, done, info = super().step(action)

        if self.near(self.box):
            reward += self._reward()
            done = True

        return obs, reward, done, info
