import numpy as np
import math
from ..miniworld import MiniWorldEnv, Room
from ..entity import Box
from ..math import gen_rot_matrix
from gym import spaces

class YMaze(MiniWorldEnv):
    """
    Two hallways connected in a Y-junction
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
        # Outline of the main (starting) arm
        main_outline = np.array([
            [-9.15, 0, -2],
            [-9.15, 0, +2],
            [-1.15, 0, +2],
            [-1.15, 0, -2],
        ])

        main_arm = self.add_room(outline=np.delete(main_outline, 1, 1))

        # Triangular hub room, outline of XZ points
        hub_room = self.add_room(outline=np.array([
            [-1.15, -2],
            [-1.15, +2],
            [ 2.31,  0],
        ]))

        # Left arm of the maze
        m = gen_rot_matrix(np.array([0, 1, 0]), -120 * (math.pi/180))
        left_outline = np.dot(main_outline, m)
        left_arm = self.add_room(outline=np.delete(left_outline, 1, 1))

        # Right arm of the maze
        m = gen_rot_matrix(np.array([0, 1, 0]), +120 * (math.pi/180))
        right_outline = np.dot(main_outline, m)
        right_arm = self.add_room(outline=np.delete(right_outline, 1, 1))

        # Connect the maze arms with the hub
        self.connect_rooms(main_arm, hub_room, min_z=-2, max_z=2)
        self.connect_rooms(left_arm, hub_room, min_z=-1.995, max_z=0)
        self.connect_rooms(right_arm, hub_room, min_z=0, max_z=1.995)

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
                self.place_entity(self.box, room=left_arm, max_z=left_arm.min_z + 2.5)
            else:
                self.place_entity(self.box, room=right_arm, min_z=right_arm.max_z - 2.5)

        # Choose a random room and position to spawn at
        self.place_agent(
            dir=self.rand.float(-math.pi/4, math.pi/4),
            room=main_arm
        )

    def step(self, action):
        obs, reward, done, info = super().step(action)

        if self.near(self.box):
            reward += self._reward()
            done = True

        info['goal_pos'] = self.box.pos

        return obs, reward, done, info

class YMazeLeft(YMaze):
    def __init__(self):
        super().__init__(goal_pos=[3.9, 0, -7.0])

class YMazeRight(YMaze):
    def __init__(self):
        super().__init__(goal_pos=[3.9, 0, 7.0])
