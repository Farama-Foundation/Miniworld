import numpy as np
import math
from ..miniworld import MiniWorldEnv, Room
from ..entity import Box
from ..params import DEFAULT_PARAMS
from gym import spaces

# Simulation parameters
# These assume a robot about 15cm tall with a pi camera module v2
sim_params = DEFAULT_PARAMS.copy()
sim_params.set('forward_step', 0.035, 0.028, 0.042)
sim_params.set('turn_step', 17, 13, 21)
sim_params.set('bot_radius', 0.4, 0.38, 0.42) # FIXME: not used
sim_params.set('cam_pitch', -10, -15, -3)
sim_params.set('cam_fov_y', 49, 45, 55)
sim_params.set('cam_height', 0.18, 0.17, 0.19)
sim_params.set('cam_fwd_disp', 0, -0.02, 0.02)
# TODO: modify lighting parameters

class SimToRealPushEnv(MiniWorldEnv):
    """
    Environment designed for sim-to-real transfer.
    In this environment, the robot has to push the
    red box towards the yellow box.
    """

    def __init__(self, **kwargs):
        super().__init__(
            max_episode_steps=100,
            params=sim_params,
            domain_rand=True,
            **kwargs
        )

        # Allow only the movement actions
        self.action_space = spaces.Discrete(self.actions.move_forward+1)

    def _gen_world(self):
        # ~1.2 meter wide rink
        size = self.rand.float(1.1, 1.3)

        wall_height = self.rand.float(0.42, 0.50)

        box1_size = self.rand.float(0.075, 0.090)
        box2_size = self.rand.float(0.075, 0.090)

        self.agent.radius = 0.11

        floor_tex = self.rand.choice([
            'cardboard',
            'wood',
            'wood_planks',
        ])

        wall_tex = self.rand.choice([
            'drywall',
            'stucco',
            # Chosen because they have visible lines/seams
            'concrete_tiles',
            'ceiling_tiles',
        ])

        # Create a long rectangular room
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

        # FIXME: the box to be pushed can't be too close to the walls,
        # limit its spawn area

        min_dist = box1_size + box2_size
        self.goal_dist = 1.5 * min_dist

        while True:
            self.box1 = self.place_entity(Box(color='red', size=box1_size))
            self.box2 = self.place_entity(Box(color='yellow', size=box2_size))

            self.start_dist = np.linalg.norm(self.box1.pos - self.box2.pos)
            if self.start_dist > self.goal_dist:
                break

            self.entities.remove(self.box1)
            self.entities.remove(self.box2)

        # Place the agent a random distance away from the goal
        self.place_agent()

    def step(self, action):
        obs, reward, done, info = super().step(action)

        # TODO: sparse rewards?

        dist = np.linalg.norm(self.box1.pos - self.box2.pos)

        if dist < self.goal_dist:
            reward = 1
            done = True

        return obs, reward, done, info
