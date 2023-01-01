import math

import numpy as np
from gymnasium import spaces, utils

from miniworld.entity import Box
from miniworld.math import gen_rot_matrix
from miniworld.miniworld import MiniWorldEnv


class YMaze(MiniWorldEnv, utils.EzPickle):
    """
    ## Description

    Two hallways connected in a Y-junction. the goal is to move the agent
    towards a red box within as little steps as possible. In
    `MiniWorld-YMazeLeft-v0`, the red box is located on the left wing of
    the Y-shaped junction. In `MiniWorld-YMazeRight-v0`,  the red box is
    located on the right wing of the Y-shaped junction.

    ## Action Space

    | Num | Action                      |
    |-----|-----------------------------|
    | 0   | turn left                   |
    | 1   | turn right                  |
    | 2   | move forward                |

    ## Observation Space

    The observation space is an `ndarray` with shape `(obs_height, obs_width, 3)`
    representing a RGB image of what the agents sees.

    ## Rewards:

    +(1 - 0.2 * (step_count / max_episode_steps)) when box reached

    ## Arguments

    ```python
    env = gym.make("MiniWorld-YMazeLeft-v0")
    # or
    env = gym.make("MiniWorld-YMazeRight-v0")
    ```

    """

    def __init__(self, goal_pos=None, **kwargs):
        self.goal_pos = goal_pos

        MiniWorldEnv.__init__(self, max_episode_steps=280, **kwargs)
        utils.EzPickle.__init__(self, goal_pos, **kwargs)

        # Allow only movement actions (left/right/forward)
        self.action_space = spaces.Discrete(self.actions.move_forward + 1)

    def _gen_world(self):
        # Outline of the main (starting) arm
        main_outline = np.array(
            [
                [-9.15, 0, -2],
                [-9.15, 0, +2],
                [-1.15, 0, +2],
                [-1.15, 0, -2],
            ]
        )

        main_arm = self.add_room(outline=np.delete(main_outline, 1, 1))

        # Triangular hub room, outline of XZ points
        hub_room = self.add_room(
            outline=np.array(
                [
                    [-1.15, -2],
                    [-1.15, +2],
                    [2.31, 0],
                ]
            )
        )

        # Left arm of the maze
        m = gen_rot_matrix(np.array([0, 1, 0]), -120 * (math.pi / 180))
        left_outline = np.dot(main_outline, m)
        left_arm = self.add_room(outline=np.delete(left_outline, 1, 1))

        # Right arm of the maze
        m = gen_rot_matrix(np.array([0, 1, 0]), +120 * (math.pi / 180))
        right_outline = np.dot(main_outline, m)
        right_arm = self.add_room(outline=np.delete(right_outline, 1, 1))

        # Connect the maze arms with the hub
        self.connect_rooms(main_arm, hub_room, min_z=-2, max_z=2)
        self.connect_rooms(left_arm, hub_room, min_z=-1.995, max_z=0)
        self.connect_rooms(right_arm, hub_room, min_z=0, max_z=1.995)

        # Add a box at a random end of the hallway
        self.box = Box(color="red")

        # Place the goal in the left or the right arm
        if self.goal_pos is not None:
            self.place_entity(
                self.box,
                min_x=self.goal_pos[0],
                max_x=self.goal_pos[0],
                min_z=self.goal_pos[2],
                max_z=self.goal_pos[2],
            )
        else:
            if self.np_random.integers(0, 2) == 0:
                self.place_entity(self.box, room=left_arm, max_z=left_arm.min_z + 2.5)
            else:
                self.place_entity(self.box, room=right_arm, min_z=right_arm.max_z - 2.5)

        # Choose a random room and position to spawn at
        self.place_agent(
            dir=self.np_random.uniform(-math.pi / 4, math.pi / 4), room=main_arm
        )

    def step(self, action):
        obs, reward, termination, truncation, info = super().step(action)

        if self.near(self.box):
            reward += self._reward()
            termination = True

        info["goal_pos"] = self.box.pos

        return obs, reward, termination, truncation, info


class YMazeLeft(YMaze):
    def __init__(self, goal_pos=[3.9, 0, -7.0], **kwargs):
        super().__init__(goal_pos=goal_pos, **kwargs)


class YMazeRight(YMaze):
    def __init__(self, goal_pos=[3.9, 0, 7.0], **kwargs):
        super().__init__(goal_pos=goal_pos, **kwargs)
