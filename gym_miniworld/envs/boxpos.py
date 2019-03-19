import numpy as np
import math
from ..opengl import *
from ..miniworld import MiniWorldEnv, Room
from ..params import DEFAULT_PARAMS
from ..entity import Entity, Box
from ..math import *
from .ergojr import ErgoJr, sample_angles

class BoxPos(MiniWorldEnv):
    """
    Environment to train for box prediction
    """

    def __init__(self, domain_rand=True, img_noise=True, **kwargs):
        # Simulation parameters
        sim_params = DEFAULT_PARAMS.copy()
        sim_params.set('light_pos', [0, 2.5, 0], [-10, 0.5, -10], [10, 5, 10])
        sim_params.set('light_color', [0.7, 0.7, 0.7], [0.45, 0.45, 0.45], [0.9, 0.9, 0.9])
        sim_params.set('light_ambient', [0.45, 0.45, 0.45], [0.35, 0.35, 0.35], [0.65, 0.65, 0.65])
        sim_params.set('forward_step', 0.035)
        sim_params.set('forward_drift', 0)
        sim_params.set('turn_step', 17)
        sim_params.set('bot_radius', 0.11)
        sim_params.set('cam_pitch', 0)
        sim_params.set('cam_height', 0.019)
        sim_params.set('cam_fov_y', 49)
        sim_params.set('cam_fwd_disp', 0)

        self.img_noise = img_noise

        super().__init__(
            max_episode_steps=math.inf,
            params=sim_params,
            domain_rand=domain_rand,
            **kwargs
        )

    def _gen_world(self):
        if self.domain_rand:
            wall_tex = self.rand.choice(['concrete', 'white', 'drywall', 'ceiling_tiles'])
            floor_tex = self.rand.choice(['concrete', 'white', 'drywall', 'ceiling_tiles'])
        else:
            wall_tex = 'white'
            floor_tex = 'white'

        room = self.add_rect_room(
            min_x=-0.40,
            max_x=+0.30,
            min_z=-0.40,
            max_z=+0.40,
            wall_height=0.45,
            no_ceiling=True,
            wall_tex=wall_tex,
            floor_tex=floor_tex
        )

        self.ergojr = self.place_entity(ErgoJr(), pos=[0, 0, 0], dir=0)
        self.ergojr.angles = sample_angles(y_max=0.10)

        # The box looks the same from all sides, so restrict angles to [0, 90]
        self.box = self.place_entity(
            Box(color='green', size=0.025),
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

    def render_obs(self, frame_buffer=None):
        obs = super().render_obs(frame_buffer)

        # Add gaussian noise to observations, and exposure noise
        if self.img_noise:
            noise = np.random.normal(loc=0, scale=4, size=obs.shape)
            fact = np.random.normal(loc=1, scale=0.02, size=(1,1,3)).clip(0.95, 1.05)
            obs = (fact * obs + noise).clip(0, 255).astype(np.uint8)

        return obs

    def step(self, action):
        obs, reward, done, info = super().step(action)
        return obs, reward, done, info
