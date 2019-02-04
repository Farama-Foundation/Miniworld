import numpy as np
import math
from ..miniworld import MiniWorldEnv, Room
from ..entity import Box
from ..params import DEFAULT_PARAMS
from gym import spaces

# Simulation parameters
# These assume a robot about 15cm tall with a pi camera module v2
sim_params = DEFAULT_PARAMS.copy()
sim_params.set('forward_step', 0.035, 0.025, 0.045)
sim_params.set('forward_drift', 0, -0.01, 0.01)
sim_params.set('turn_step', 17, 10, 22)
sim_params.set('bot_radius', 0.11, 0.11, 0.11)
sim_params.set('cam_pitch', -5, -8, -2)
sim_params.set('cam_fov_y', 49, 46, 53)
sim_params.set('cam_height', 0.18, 0.17, 0.19)
sim_params.set('cam_fwd_disp', 0, -0.02, 0.02)

class SimToRealOdoEnv(MiniWorldEnv):
    """
    Environment designed for sim-to-real transfer.
    In this environment, the robot has to push the
    red box towards the yellow box.
    """

    def __init__(self, **kwargs):
        super().__init__(
            max_episode_steps=150,
            params=sim_params,
            domain_rand=True,
            **kwargs
        )

        # Allow only the movement actions
        self.action_space = spaces.Discrete(self.actions.move_forward+1)

    def _gen_world(self):
        # Size of the rink the robot is placed in
        # Real rink expected to be 48" = 1.21m wide
        size = self.rand.float(1.1, 1.3)
        wall_height = self.rand.float(0.42, 0.50)

        self.agent.radius = 0.11

        floor_tex = self.rand.choice([
            'concrete',
            'concrete_tiles',
            #'wood',
            #'wood_planks',
        ])

        wall_tex = self.rand.choice([
            'drywall',
            'stucco',
            # Materials chosen because they have visible lines/seams
            'concrete_tiles',
            'ceiling_tiles',
            # Chosen because of random/slanted edges
            'marble',
            'rock',
        ])

        room = self.add_rect_room(
            min_x=0,
            max_x=size,
            min_z=0,
            max_z=size,
            no_ceiling=True,
            wall_height=wall_height,
            wall_tex=wall_tex,
            floor_tex=floor_tex
        )

        # Place the agent a random distance away from the goal
        self.place_agent()

    def step(self, action):
        obs, reward, done, info = super().step(action)

        return obs, reward, done, info
