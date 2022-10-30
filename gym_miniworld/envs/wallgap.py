import math

import numpy as np
from gym_miniworld.entity import Box, MeshEnt
from gym_miniworld.miniworld import MiniWorldEnv
from gymnasium import spaces


class WallGap(MiniWorldEnv):
    """
    ## Description

    Outside environment with two rooms connected by a gap in a wall

    ## Action Space

    | Num | Action                      |
    |-----|-----------------------------|
    | 0   | turn left                   |
    | 1   | turn right                  |
    | 2   | move forward                |

    ## Observation Space

    The observation space is an `ndarray` with shape `(obs_height, obs_width, 3)`
    representing the view the agents sees.

    ## Rewards:

    +(1 - 0.2 * (step_count / max_episode_steps)) when box reached

    ## Arguments

    ```python
    WallGap()
    ```
    """

    def __init__(self, **kwargs):
        super().__init__(max_episode_steps=300, **kwargs)

        # Allow only the movement actions
        self.action_space = spaces.Discrete(self.actions.move_forward + 1)

    def _gen_world(self):
        # Top
        room0 = self.add_rect_room(
            min_x=-7,
            max_x=7,
            min_z=0.5,
            max_z=8,
            wall_tex="brick_wall",
            floor_tex="asphalt",
            no_ceiling=True,
        )
        # Bottom
        room1 = self.add_rect_room(
            min_x=-7,
            max_x=7,
            min_z=-8,
            max_z=-0.5,
            wall_tex="brick_wall",
            floor_tex="asphalt",
            no_ceiling=True,
        )
        self.connect_rooms(room0, room1, min_x=-1.5, max_x=1.5)

        self.box = self.place_entity(Box(color="red"), room=room1)

        # Decorative building in the background
        self.place_entity(
            MeshEnt(mesh_name="building", height=30),
            pos=np.array([30, 0, 30]),
            dir=-math.pi,
        )

        self.place_agent(room=room0)

    def step(self, action):
        obs, reward, termination, truncation, info = super().step(action)

        if self.near(self.box):
            reward += self._reward()
            termination = True

        return obs, reward, termination, truncation, info
