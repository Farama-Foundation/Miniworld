import numpy as np
import math
from ..miniworld import MiniWorldEnv, Room
from ..entity import Box
from gym import spaces

class TMaze(MiniWorldEnv):
    """
    Two hallways connected in a T-junction
    """

    def __init__(
        self,
        goal_pos=None,
        **kwargs
    ):
        self.goal_pos = goal_pos

        super().__init__(
            max_episode_steps=280,
            **kwargs
        )

        # Allow only movement actions (left/right/forward)
        self.action_space = spaces.Discrete(self.actions.move_forward+1)

    def _gen_world(self):
        room1 = self.add_rect_room(
            min_x=-1, max_x=8,
            min_z=-2, max_z=2
        )
        room2 = self.add_rect_room(
            min_x=8, max_x=12,
            min_z=-8, max_z=8
        )
        self.connect_rooms(room1, room2, min_z=-2, max_z=2)

        # Add a box at a random end of the hallway
        self.box = Box(color='red')

        # Place the goal in the left or the right arm
        if self.goal_pos != None:
            self.place_entity(
                self.box,
                min_x=self.goal_pos[0],
                max_x=self.goal_pos[0],
                min_z=self.goal_pos[2],
                max_z=self.goal_pos[2],
            )
        else:
            if self.rand.bool():
                self.place_entity(self.box, room=room2, max_z=room2.min_z + 2)
            else:
                self.place_entity(self.box, room=room2, min_z=room2.max_z - 2)

        # Choose a random room and position to spawn at
        self.place_agent(
            dir=self.rand.float(-math.pi/4, math.pi/4),
            room=room1
        )

    def step(self, action):
        obs, reward, done, info = super().step(action)

        if self.near(self.box):
            reward += self._reward()
            done = True

        info['goal_pos'] = self.box.pos

        return obs, reward, done, info

class TMazeLeft(TMaze):
    def __init__(self):
        super().__init__(goal_pos=[10, 0, -6])

class TMazeRight(TMaze):
    def __init__(self):
        super().__init__(goal_pos=[10, 0, 6])
