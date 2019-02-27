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
sim_params.set('forward_drift', 0, -0.005, 0.005)
sim_params.set('turn_step', 17, 13, 21)
sim_params.set('bot_radius', 0.11, 0.11, 0.11)
sim_params.set('cam_pitch', -10, -15, -3)
sim_params.set('cam_fov_y', 49, 45, 55)
sim_params.set('cam_height', 0.18, 0.17, 0.19)
sim_params.set('cam_fwd_disp', 0, -0.02, 0.02)
# TODO: modify lighting parameters

class SimToRealPush(MiniWorldEnv):
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

        # Allow only the movement actions (left, right, forward, back)
        self.action_space = spaces.Discrete(self.actions.move_back+1)

    def _gen_world(self):
        # Size of the rink the robot is placed in
        size = self.rand.float(1.6, 1.7)
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
            # Materials chosen because they have visible lines/seams
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

        # Target distance for the boxes
        min_dist = box1_size + box2_size
        self.goal_dist = 1.5 * min_dist

        # Avoid spawning boxes in the corners (where they can't be pushed)
        min_pos = 2 * self.params.get_max('bot_radius')
        max_pos = size - 2 * self.params.get_max('bot_radius')

        while True:
            self.box1 = self.place_entity(
                Box(color='red', size=box1_size),
                min_x = min_pos,
                min_z = min_pos,
                max_x = max_pos,
                max_z = max_pos
            )
            self.box2 = self.place_entity(
                Box(color='yellow', size=box2_size),
                min_x = min_pos,
                min_z = min_pos,
                max_x = max_pos,
                max_z = max_pos
            )

            # Boxes can't start too close to each other
            self.start_dist = np.linalg.norm(self.box1.pos - self.box2.pos)
            if self.start_dist > self.goal_dist:
                break

            self.entities.remove(self.box1)
            self.entities.remove(self.box2)

        # Place the agent a random distance away from the goal
        self.place_agent()

    def step(self, action):
        # Very crude approximation the physics of box pushing
        if action == self.actions.move_forward:
            fwd_dist = self.params.get_max('forward_step')
            delta_pos = self.agent.dir_vec * fwd_dist
            next_pos = self.agent.pos + delta_pos

            for box in [self.box1, self.box2]:
                vec = box.pos - next_pos
                dist = np.linalg.norm(vec)

                if dist < self.agent.radius + box.radius:
                    #print('collision')
                    next_box_pos = box.pos + vec
                    if not self.intersect(box, next_box_pos, box.radius):
                        box.pos = next_box_pos
                        box.dir += self.rand.float(-math.pi/5, math.pi/5)

        obs, reward, done, info = super().step(action)

        # TODO: give sparse rewards?
        dist = np.linalg.norm(self.box1.pos - self.box2.pos)
        if dist < self.goal_dist:
            reward = 1
            done = True

        return obs, reward, done, info
