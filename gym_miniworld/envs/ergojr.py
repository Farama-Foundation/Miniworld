import numpy as np
import math
from ..opengl import *
from ..miniworld import MiniWorldEnv, Room
from ..params import DEFAULT_PARAMS
from ..entity import Entity

# Simulation parameters
sim_params = DEFAULT_PARAMS.copy()
sim_params.set('forward_step', 0.035, 0.028, 0.042)
sim_params.set('forward_drift', 0, -0.005, 0.005)
sim_params.set('turn_step', 17, 13, 21)
sim_params.set('bot_radius', 0.11, 0.11, 0.11)
sim_params.set('cam_pitch', -10, -15, -3)
sim_params.set('cam_fov_y', 49, 45, 55)
sim_params.set('cam_height', 0.25)
sim_params.set('cam_fwd_disp', 0, -0.02, 0.02)

def drawBox():
    """
    Draw a box centered around the origin
    """

    glBegin(GL_QUADS)




    glEnd(GL_QUADS)

class ErgoJr(Entity):
    """
    Visual simulation of the Poppy ergo-jr robot arm
    """

    def __init__(self):
        super().__init__()

        self.radius = 0.2
        self.height = 0.4

    def step(self, delta_time):
        pass

    def render(self):
        glPushMatrix()
        glTranslatef(*self.pos)
        glRotatef(self.dir * (180/math.pi), 0, 1, 0)




        glPopMatrix()

class TableTopRobot(MiniWorldEnv):
    """
    Tabletop environment with a robot arm
    """

    def __init__(self, **kwargs):
        super().__init__(
            max_episode_steps=math.inf,
            params=sim_params,
            domain_rand=False,
            **kwargs
        )

    def _gen_world(self):
        room = self.add_rect_room(
            min_x=-0.7,
            max_x=0.7,
            min_z=-0.7,
            max_z=0.7,
            wall_height=0,
            no_ceiling=True,
            floor_tex='concrete'
        )

        self.ergojr = self.place_entity(ErgoJr(), pos=[0, 0, 0], dir=0)

        self.agent.radius = 0.15
        self.place_agent(
            dir = -math.pi / 4,
            min_x = -0.8,
            max_x = -0.8,
            min_z = -0.8,
            max_z = -0.8
        )

    def step(self, action):
        obs, reward, done, info = super().step(action)
        return obs, reward, done, info
