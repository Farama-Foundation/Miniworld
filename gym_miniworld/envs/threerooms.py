import numpy as np
import math
from gym import spaces
from ..miniworld import MiniWorldEnv, Room
from ..entity import ImageFrame, MeshEnt, Box, Key, Ball, COLOR_NAMES

class ThreeRooms(MiniWorldEnv):
    """
    Two small rooms connected to one large room
    """

    def __init__(self, **kwargs):
        super().__init__(
            max_episode_steps=400,
            **kwargs
        )

        # Allow only the movement actions
        self.action_space = spaces.Discrete(self.actions.move_forward+1)

    def _gen_world(self):
        # Top room
        room0 = self.add_rect_room(
            min_x=-7, max_x=7,
            min_z=0.5 , max_z=7
        )
        # Bottom-left room
        room1 = self.add_rect_room(
            min_x=-7, max_x=-1,
            min_z=-7, max_z=-0.5
        )
        # Bottom-right room
        room2 = self.add_rect_room(
            min_x=1 , max_x=7,
            min_z=-7, max_z=-0.5
        )

        # Connect the rooms with portals/openings
        self.connect_rooms(room0, room1, min_x=-5.25, max_x=-2.75)
        self.connect_rooms(room0, room2, min_x=2.75, max_x=5.25)

        self.box = self.place_entity(Box(color='red'))
        #self.yellow_box = self.place_entity(Box(color='yellow', size=[0.8, 1.2, 0.5]))
        self.place_entity(Box(color='green', size=0.6))

        # Mila logo image on the wall
        self.entities.append(ImageFrame(
            pos=[0, 1.35, 7],
            dir=math.pi/2,
            width=1.8,
            tex_name='logo_mila'
        ))

        self.place_entity(MeshEnt(
            mesh_name='duckie',
            height=0.25,
            static=False
        ))

        self.place_entity(Key(color='blue'))
        self.place_entity(Ball(color='green'))

        self.place_agent()

    def step(self, action):
        obs, reward, done, info = super().step(action)

        return obs, reward, done, info
