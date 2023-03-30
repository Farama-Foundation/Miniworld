import copy
from typing import Optional, Tuple

from gymnasium import spaces, utils
from gymnasium.core import ObsType

from miniworld.entity import COLOR_NAMES, Ball, Box, Key
from miniworld.miniworld import MiniWorldEnv

charset = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ")


class ConditionalPickUpObject(MiniWorldEnv, utils.EzPickle):
    """
    ## Description

    Room with multiple objects. The agent collects +1 reward for picking up
    the object mentioned in the mission, and -1 if it picks up anything else.

    ## Action Space

    | Num | Action                      |
    |-----|-----------------------------|
    | 0   | turn left                   |
    | 1   | turn right                  |
    | 2   | move forward                |
    | 3   | move_back                   |
    | 4   | pickup                      |
    | 5   | drop                        |

    ## Observation Space

    The observation space is a dictionary with the following entries:
        - image:
            an `ndarray` with shape `(obs_height, obs_width, 3)`
            representing a RGB image of what the agents sees.
        - mission:
            a string of the form 'pick up the COLOR TYPE' where
            COLOR and TYPE are a visible entity's color and type.

    ## Rewards:

    +1 when agent picks up correct object, -1 if pickup up wrong object.

    ## Arguments

    ```python
    ConditionalPickUpObject(size=12, num_objs=5, cam_pitch=-30)
    ```

    `size`: size of world

    `num_objs`: number of objects

    `cam_pitch`: pitch offset of the camera in degrees.

    """

    def __init__(self, size=12, num_objs=5, cam_pitch=-30, **kwargs):
        assert size >= 2
        self.size = size
        self.num_objs = num_objs
        self.cam_pitch = cam_pitch

        MiniWorldEnv.__init__(self, max_episode_steps=400, **kwargs)
        utils.EzPickle.__init__(self, size, num_objs, **kwargs)

        image_observation_space = self.observation_space
        self.observation_space = spaces.Dict(
            {
                "image": image_observation_space,
                "mission": spaces.Text(max_length=100, charset=charset),
            }
        )

        # Reduce the action space
        self.action_space = spaces.Discrete(self.actions.drop + 1)

    def _gen_world(self):
        self.add_rect_room(
            min_x=0,
            max_x=self.size,
            min_z=0,
            max_z=self.size,
            wall_tex="brick_wall",
            floor_tex="asphalt",
            no_ceiling=True,
        )

        obj_types = [Ball, Box, Key]
        colorlist = list(COLOR_NAMES)

        obj_list = []
        while len(obj_list) < self.num_objs:
            obj_type = obj_types[self.np_random.choice(len(obj_types))]
            color = colorlist[self.np_random.choice(len(colorlist))]

            obj_descr = (obj_type, color)
            if obj_descr in obj_list:
                continue

            obj_list.append(obj_descr)

            if obj_type == Box:
                self.place_entity(Box(color=color, size=0.9))
            if obj_type == Ball:
                self.place_entity(Ball(color=color, size=0.9))
            if obj_type == Key:
                self.place_entity(Key(color=color))

        self.place_agent()
        self.agent.cam_pitch = self.cam_pitch

        # Create mission:
        obj = self.np_random.choice(obj_list)
        obj_color = obj[1]
        obj_type = obj[0].__name__.lower()
        self.mission = f"pick up the {obj_color} {obj_type}"

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[ObsType, dict]:
        """
        Reset the simulation at the start of a new episode
        This also randomizes many environment parameters (domain randomization)
        """
        obs, info = super().reset(seed=seed)

        output_d = {
            "image": obs,
            "mission": copy.deepcopy(self.mission),
        }

        return output_d, info

    def step(self, action):
        self.agent.cam_pitch = self.cam_pitch
        obs, reward, termination, truncation, info = super().step(action)

        if self.agent.carrying:
            obj_type = type(self.agent.carrying).__name__.lower()
            obj_color = getattr(self.agent.carrying, "color", None)
            if obj_color in self.mission and obj_type in self.mission:
                reward = 1
            else:
                reward = -1

            if reward > 0:
                termination = True

        output_d = {
            "image": obs,
            "mission": copy.deepcopy(self.mission),
        }

        return output_d, reward, termination, truncation, info
