from gymnasium import utils

from miniworld.entity import COLOR_NAMES, Box
from miniworld.miniworld import MiniWorldEnv


class PutNext(MiniWorldEnv, utils.EzPickle):
    """
    ## Description

    Single-room environment where a red box must be placed next
    to a yellow box. The goal is to perform this task in as few
    steps as possible.

    ## Action Space

    | Num | Action                      |
    |-----|-----------------------------|
    | 0   | turn left                   |
    | 1   | turn right                  |
    | 2   | move forward                |
    | 3   | move back                   |
    | 4   | pick up                     |
    | 5   | drop                        |
    | 6   | toggle / activate an object |
    | 7   | complete task               |

    ## Observation Space

    The observation space is an `ndarray` with shape `(obs_height, obs_width, 3)`
    representing a RGB image of what the agents sees.

    ## Rewards:

    +(1 - 0.2 * (step_count / max_episode_steps)) when red box is next to yellow box

    ## Arguments

    ```python
    PutNext(size=12)
    ```

    `size`: size of world

    """

    def __init__(self, size=12, **kwargs):
        assert size >= 2
        self.size = size

        MiniWorldEnv.__init__(self, max_episode_steps=250, **kwargs)
        utils.EzPickle.__init__(self, size, **kwargs)

    def _gen_world(self):
        # Create a long rectangular room
        self.add_rect_room(min_x=0, max_x=self.size, min_z=0, max_z=self.size)

        for color in COLOR_NAMES:
            box = Box(color=color, size=self.np_random.uniform(0.6, 0.85))
            self.place_entity(box)

            if box.color == "red":
                self.red_box = box
            elif box.color == "yellow":
                self.yellow_box = box

        # Place the agent a random distance away from the goal
        self.place_agent()

    def step(self, action):
        obs, reward, termination, truncation, info = super().step(action)

        if not self.agent.carrying:
            if self.near(self.red_box, self.yellow_box):
                reward += self._reward()
                termination = True

        return obs, reward, termination, truncation, info
