import numpy as np
import math
from ..miniworld import MiniWorldEnv, Room
from ..entity import Box
from gym import spaces

class TMaze(MiniWorldEnv):
    """
    Two hallways connected in a T-junction
    """

    def __init__(self, goal_pos=None, **kwargs):
        # Position of the goal, left, right, or random
        assert goal_pos in [0, 1, None]
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

        # If no goal position is specified, pick a random one
        goal_pos = self.goal_pos
        if goal_pos is None:
            goal_pos = self.rand.bool()

        # Place the goal in the left or the right arm
        if goal_pos == 0:
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

        return obs, reward, done, info

class TMazeLeft(TMaze):
    def __init__(self):
        super().__init__(goal_pos=0)

class TMazeRight(TMaze):
    def __init__(self):
        super().__init__(goal_pos=1)
