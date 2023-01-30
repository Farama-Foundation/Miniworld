from gymnasium import spaces, utils

from miniworld.entity import Box
from miniworld.miniworld import MiniWorldEnv
from miniworld.params import DEFAULT_PARAMS


class OneRoom(MiniWorldEnv, utils.EzPickle):
    """
    ## Description

    Environment in which the goal is to go to a red box placed randomly in one big room.
    The `OneRoom` environment has two variants. The `OneRoomS6` environment gives you
    a room with size 6 (the `OneRoom` environment has size 10). The `OneRoomS6Fast`
    environment also is using a room with size 6, but the turning and moving motion
    is larger.

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
    env = gym.make("MiniWorld-OneRoom-v0")
    # or
    env = gym.make("MiniWorld-OneRoomS6-v0")
    # or
    env = gym.make("MiniWorld-OneRoomS6Fast-v0")
    ```

    """

    def __init__(self, size=10, max_episode_steps=180, **kwargs):
        assert size >= 2
        self.size = size

        MiniWorldEnv.__init__(self, max_episode_steps=max_episode_steps, **kwargs)
        utils.EzPickle.__init__(
            self, size=size, max_episode_steps=max_episode_steps, **kwargs
        )

        # Allow only movement actions (left/right/forward)
        self.action_space = spaces.Discrete(self.actions.move_forward + 1)

    def _gen_world(self):
        self.add_rect_room(min_x=0, max_x=self.size, min_z=0, max_z=self.size)

        self.box = self.place_entity(Box(color="red"))
        self.place_agent()

    def step(self, action):
        obs, reward, termination, truncation, info = super().step(action)

        if self.near(self.box):
            reward += self._reward()
            termination = True

        return obs, reward, termination, truncation, info


class OneRoomS6(OneRoom):
    def __init__(self, size=6, max_episode_steps=100, **kwargs):
        super().__init__(size=size, max_episode_steps=max_episode_steps, **kwargs)


# Parameters for larger movement steps, fast stepping
default_params = DEFAULT_PARAMS.no_random()
default_params.set("forward_step", 0.7)
default_params.set("turn_step", 45)


class OneRoomS6Fast(OneRoomS6):
    def __init__(
        self, max_episode_steps=50, params=default_params, domain_rand=False, **kwargs
    ):

        super().__init__(
            max_episode_steps=max_episode_steps,
            params=params,
            domain_rand=domain_rand,
            **kwargs
        )
