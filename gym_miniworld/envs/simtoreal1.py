import numpy as np
import math
from ..miniworld import MiniWorldEnv, Room
from ..entity import Box
from ..params import DEFAULT_PARAMS

# Simulation parameters
# These assume a robot about 15cm tall with a pi camera module v2
sim_params = DEFAULT_PARAMS.copy()
sim_params.set('forward_step', 0.04, 0.03, 0.05)
sim_params.set('turn_step', 15, 10, 20)
sim_params.set('bot_radius', 0.4, 0.38, 0.42)
sim_params.set('cam_pitch', -5, -10, 0)
sim_params.set('cam_fov_y', 49, 45, 55)
sim_params.set('cam_height', 0.18, 0.17, 0.19)
sim_params.set('cam_fwd_disp', 0, -0.02, 0.02)

class SimToReal1Env(MiniWorldEnv):
    """
    Environment designed for sim-to-real transfer

    This environment assumes a robot about 15cm tall
    using a Raspberri Pi camera module V2
    """

    def __init__(self, **kwargs):
        super().__init__(
            max_episode_steps=100,
            params=sim_params,
            domain_rand=True,
            **kwargs
        )

    def _gen_world(self):
        # 1-2 meter wide rink
        size = self.rand.float(1, 2)

        wall_height = self.rand.float(0.20, 0.45)

        box_size = self.rand.float(0.07, 0.12)

        self.agent.radius = 0.19

        floor_tex = self.rand.choice([
            'cardboard',
            'wood',
            'wood_planks'
        ])

        # Create a long rectangular room
        room = self.add_rect_room(
            min_x=0,
            max_x=size,
            min_z=0,
            max_z=size,
            no_ceiling=True,
            wall_height=wall_height,
            wall_tex='cardboard',
            floor_tex=floor_tex
        )

        # Place the box at the end of the hallway
        self.box = self.place_entity(Box(color='red', size=box_size))

        # Place the agent a random distance away from the goal
        self.place_agent()

    def step(self, action):
        obs, reward, done, info = super().step(action)

        if self.near(self.box):
            reward += self._reward()
            done = True

        return obs, reward, done, info
