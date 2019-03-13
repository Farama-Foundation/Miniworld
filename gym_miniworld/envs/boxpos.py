import numpy as np
import math
from ..opengl import *
from ..miniworld import MiniWorldEnv, Room
from ..params import DEFAULT_PARAMS
from ..entity import Entity, Box
from ..math import *

# Simulation parameters
sim_params = DEFAULT_PARAMS.copy()
sim_params.set('forward_step', 0.035)
sim_params.set('forward_drift', 0)
sim_params.set('turn_step', 17)
sim_params.set('bot_radius', 0.11)
sim_params.set('cam_pitch', 0)
sim_params.set('cam_fov_y', 49)
sim_params.set('cam_height', 0.02)
sim_params.set('cam_fwd_disp', 0)

class BoxPos(MiniWorldEnv):
    """
    Environment to train for box prediction
    """

    def __init__(self, **kwargs):
        super().__init__(
            max_episode_steps=math.inf,
            params=sim_params,
            domain_rand=True,
            **kwargs
        )

    def _gen_world(self):
        room = self.add_rect_room(
            min_x=-0.40,
            max_x=self.rand.float(0.25, 0.35),
            min_z=-0.40,
            max_z=+0.40,
            wall_height=0.45,
            no_ceiling=True,
            wall_tex=self.rand.choice(['concrete', 'white', 'drywall', 'ceiling_tiles']),
            floor_tex=self.rand.choice(['concrete', 'white', 'drywall', 'ceiling_tiles'])
        )

        # The box looks the same from all sides, so restrict angles to [0, 90]
        self.box = self.place_entity(
            Box(color='green', size=0.03),
            min_x=0.055,
            max_x=0.30,
            min_z=-0.15,
            max_z=+0.15,
            dir=self.rand.float(0, math.pi/2)
        )
        self.box.pos[1] = self.rand.float(0, 0.15)

        # Camera is 3.5cm forward
        self.entities.append(self.agent)
        self.agent.radius = 0
        self.agent.dir = 0
        self.agent.pos = np.array([
            0.035,
            0,
            0
        ])

    def step(self, action):
        obs, reward, done, info = super().step(action)
        return obs, reward, done, info
