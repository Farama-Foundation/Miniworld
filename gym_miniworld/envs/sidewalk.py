import numpy as np
import math
from gym import spaces
from ..miniworld import MiniWorldEnv, Room
from ..entity import Box, ImageFrame, MeshEnt

class Sidewalk(MiniWorldEnv):
    """
    Walk on a sidewalk up to an object to be collected.
    Don't walk into the street.
    """

    def __init__(self, **kwargs):
        super().__init__(
            max_episode_steps=150,
            **kwargs
        )

        # Allow only the movement actions
        self.action_space = spaces.Discrete(self.actions.move_forward+1)

    def _gen_world(self):
        sidewalk = self.add_rect_room(
            min_x=-3, max_x=0,
            min_z=0 , max_z=12,
            wall_tex='brick_wall',
            floor_tex='concrete_tiles',
            no_ceiling=True
        )

        self.street = self.add_rect_room(
            min_x=0, max_x=6,
            min_z=-80, max_z=80,
            #wall_tex='brick_wall',
            #wall_height=0,
            floor_tex='asphalt',
            no_ceiling=True
        )

        self.connect_rooms(sidewalk, self.street, min_z=0, max_z=12)

        # Decorative building in the background
        self.place_entity(
            MeshEnt(
                mesh_name='building',
                height=30
            ),
            pos = np.array([30, 0, 30]),
            dir = -math.pi
        )

        for i in range(1, sidewalk.max_z//2):
            self.place_entity(
                MeshEnt(
                    mesh_name='cone',
                    height=0.75
                ),
                pos = np.array([1, 0, 2*i])
            )

        self.box = self.place_entity(
            Box(color='red'),
            room=sidewalk,
            min_z=sidewalk.max_z-2,
            max_z=sidewalk.max_z
        )

        self.place_agent(
            room=sidewalk,
            min_z=0,
            max_z=1.5
        )

    def step(self, action):
        obs, reward, done, info = super().step(action)

        # Walking into the street ends the episode
        if self.street.point_inside(self.agent.pos):
            reward = 0
            done = True

        if self.near(self.box):
            reward += self._reward()
            done = True

        return obs, reward, done, info
