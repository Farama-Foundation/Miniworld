from gymnasium import spaces, utils

from miniworld.entity import Box
from miniworld.miniworld import MiniWorldEnv


class FourRooms(MiniWorldEnv, utils.EzPickle):
    """
    ## Description

    Classic four rooms environment. The goal is to reach the red box to get a
    reward in as few steps as possible.

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

    +(1 - 0.2 * (step_count / max_episode_steps)) when red box reached and zero otherwise.

    ## Arguments

    ```python
    env = gym.make("MiniWorld-FourRooms-v0")
    ```

    """

    def __init__(self, **kwargs):
        MiniWorldEnv.__init__(self, max_episode_steps=250, **kwargs)
        utils.EzPickle.__init__(self, **kwargs)

        # Allow only the movement actions
        self.action_space = spaces.Discrete(self.actions.move_forward + 1)

    def _gen_world(self):
        # Top-left room
        room0 = self.add_rect_room(min_x=-7, max_x=-1, min_z=1, max_z=7)
        # Top-right room
        room1 = self.add_rect_room(min_x=1, max_x=7, min_z=1, max_z=7)
        # Bottom-right room
        room2 = self.add_rect_room(min_x=1, max_x=7, min_z=-7, max_z=-1)
        # Bottom-left room
        room3 = self.add_rect_room(min_x=-7, max_x=-1, min_z=-7, max_z=-1)

        # Add openings to connect the rooms together
        self.connect_rooms(room0, room1, min_z=3, max_z=5, max_y=2.2)
        self.connect_rooms(room1, room2, min_x=3, max_x=5, max_y=2.2)
        self.connect_rooms(room2, room3, min_z=-5, max_z=-3, max_y=2.2)
        self.connect_rooms(room3, room0, min_x=-5, max_x=-3, max_y=2.2)

        self.box = self.place_entity(Box(color="red"))

        self.place_agent()

    def step(self, action):
        obs, reward, termination, truncation, info = super().step(action)

        if self.near(self.box):
            reward += self._reward()
            termination = True

        return obs, reward, termination, truncation, info
