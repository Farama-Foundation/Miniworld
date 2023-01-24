import math

from gymnasium import spaces, utils

from miniworld.entity import Box
from miniworld.miniworld import MiniWorldEnv


class TMaze(MiniWorldEnv, utils.EzPickle):
    """
    ## Description

    Two hallways connected in a T-junction, the goal is to move the agent
    towards a red box within as few steps as possible. In
    `MiniWorld-TMazeLeft-v0`, the red box is located on the left wing of
    the T-shaped junction. In `MiniWorld-TMazeRight-v0`,  the red box is
    located on the right wing of the T-shaped junction.

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

    +(1 - 0.2 * (step_count / max_episode_steps)) when box reached and zero otherwise.

    ## Arguments

    ```python
    env = gym.make("MiniWorld-TMazeLeft-v0")
    # or
    env = gym.make("MiniWorld-TMazeRight-v0")
    ```
    """

    def __init__(self, goal_pos=None, **kwargs):
        self.goal_pos = goal_pos

        MiniWorldEnv.__init__(self, max_episode_steps=280, **kwargs)
        utils.EzPickle.__init__(self, goal_pos, **kwargs)

        # Allow only movement actions (left/right/forward)
        self.action_space = spaces.Discrete(self.actions.move_forward + 1)

    def _gen_world(self):
        room1 = self.add_rect_room(min_x=-1, max_x=8, min_z=-2, max_z=2)
        room2 = self.add_rect_room(min_x=8, max_x=12, min_z=-8, max_z=8)
        self.connect_rooms(room1, room2, min_z=-2, max_z=2)

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
                self.place_entity(self.box, room=room2, max_z=room2.min_z + 2)
            else:
                self.place_entity(self.box, room=room2, min_z=room2.max_z - 2)

        # Choose a random room and position to spawn at
        self.place_agent(
            dir=self.np_random.uniform(-math.pi / 4, math.pi / 4), room=room1
        )

    def step(self, action):
        obs, reward, termination, truncation, info = super().step(action)

        if self.near(self.box):
            reward += self._reward()
            termination = True

        info["goal_pos"] = self.box.pos

        return obs, reward, termination, truncation, info


class TMazeLeft(TMaze):
    def __init__(self, goal_pos=[10, 0, -6], **kwargs):
        super().__init__(goal_pos=goal_pos, **kwargs)


class TMazeRight(TMaze):
    def __init__(self, goal_pos=[10, 0, 6], **kwargs):
        super().__init__(goal_pos=goal_pos, **kwargs)
