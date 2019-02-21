import numpy as np
import math
from gym import spaces
from ..miniworld import MiniWorldEnv, Room
from ..entity import Box, ImageFrame, MeshEnt
from ..params import DEFAULT_PARAMS

# Simulation parameters
# These assume a robot about 15cm tall with a pi camera module v2
sim_params = DEFAULT_PARAMS.copy()
sim_params.set('light_pos', [0, 2.5, 0], [-40, 1.0, -40], [40, 7, 40])
sim_params.set('light_color', [0.7, 0.7, 0.7], [0.3, 0.3, 0.3], [1.2, 1.2, 1.2])
sim_params.set('light_ambient', [0.45, 0.45, 0.45], [0.2, 0.2, 0.2], [1.0, 1.0, 1.0])
sim_params.set('forward_step', 0.035, 0.020, 0.050)
sim_params.set('forward_drift', 0, -0.011, 0.011)
sim_params.set('turn_step', 15, 8, 22)
sim_params.set('bot_radius', 0.11, 0.11, 0.11)
sim_params.set('cam_pitch', -10, -13, -7)
sim_params.set('cam_fov_y', 49, 46, 53)
sim_params.set('cam_height', 0.18, 0.17, 0.19)
sim_params.set('cam_fwd_disp', 0, -0.02, 0.02)

class SimToRealOdo2Env(MiniWorldEnv):
    """
    Two small rooms connected to one large room
    """

    def __init__(self, **kwargs):
        super().__init__(
            max_episode_steps=2000,
            params=sim_params,
            domain_rand=True,
            **kwargs
        )

        # Allow only the movement actions
        self.action_space = spaces.Discrete(self.actions.move_forward+1)

    def _gen_world(self):
        # FIXME: not currently possible to set this through domain params
        self.agent.radius = 0.11

        floor_tex = self.rand.choice([
            'concrete',
            'asphalt',
            #'white',
            #'concrete_tiles',
        ])

        wall_tex = self.rand.choice([
            'drywall',
            'stucco',
            #'white',
            # Materials chosen because they have visible lines/seams
            'concrete_tiles',
            'ceiling_tiles',
            # Chosen because of random/slanted edges
            #'marble',
            #'rock',
        ])

        wall_height = self.rand.float(2, 5)

        # Top room
        room0 = self.add_rect_room(
            min_x=self.rand.float(-10, -7),
            max_x=self.rand.float(7, 10),
            min_z=0.5,
            max_z=self.rand.float(4, 10),
            wall_tex=wall_tex,
            floor_tex=floor_tex,
            wall_height=wall_height
        )
        # Bottom-left room
        room1 = self.add_rect_room(
            min_x=self.rand.float(-10, -7),
            max_x=-1,
            min_z=-7,
            max_z=-0.5,
            wall_tex=wall_tex,
            floor_tex=floor_tex,
            wall_height=wall_height
        )
        # Bottom-right room
        room2 = self.add_rect_room(
            min_x=1,
            max_x=self.rand.float(7, 10),
            min_z=-7,
            max_z=-0.5,
            wall_tex=wall_tex,
            floor_tex=floor_tex,
            wall_height=wall_height
        )

        # Connect the rooms with portals/openings
        self.connect_rooms(room0, room1, min_x=-5.25, max_x=-2.75)
        self.connect_rooms(room0, room2, min_x=2.75, max_x=5.25)

        # Mila logo image on the wall
        self.entities.append(ImageFrame(
            pos=[0, 1.35, room0.max_z],
            dir=math.pi/2,
            width=1.8,
            tex_name='logo_mila'
        ))

        self.place_entity(MeshEnt(
            mesh_name='duckie',
            height=0.25,
            static=False
        ))

        for i in range(self.rand.int(0, 10)):
            self.place_entity(Box(
                color='grey',
                size=[
                    self.rand.float(0.05, 0.4),
                    self.rand.float(0.5, 1.4),
                    self.rand.float(0.05, 0.4)
                ]
            ))

        for i in range(self.rand.int(0, 10)):
            self.place_entity(MeshEnt(
                mesh_name='office_chair',
                height=self.rand.float(1.1, 1.3),
                static=False
            ))

        for i in range(self.rand.int(0, 4)):
            self.place_entity(MeshEnt(
                mesh_name='office_desk',
                height=self.rand.float(1.7, 2.0),
                static=False
            ))

        self.place_agent()

    def render_obs(self, frame_buffer=None):
        obs = super().render_obs(frame_buffer)

        # Add gaussian noise to observations, and exposure noise
        noise = np.random.normal(loc=0, scale=4, size=obs.shape)
        fact = np.random.normal(loc=1, scale=0.02, size=(1,1,3)).clip(0.95, 1.05)
        obs = (fact * obs + noise).clip(0, 255).astype(np.uint8)

        return obs

    def step(self, action):
        pos0 = self.agent.pos

        obs, reward, done, info = super().step(action)

        pos1 = self.agent.pos

        if action == self.actions.move_forward:
            if np.array_equal(pos0, pos1):
                reward = -1
            else:
                reward = 1

        return obs, reward, done, info
