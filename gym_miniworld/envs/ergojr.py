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


def drawAxes(len=0.1):

    glBegin(GL_LINES)

    glColor3f(1, 0, 0)
    glVertex3f(0, 0, 0)
    glVertex3f(len, 0, 0)

    glColor3f(0, 1, 0)
    glVertex3f(0, 0, 0)
    glVertex3f(0, len, 0)

    glColor3f(0, 0, 1)
    glVertex3f(0, 0, 0)
    glVertex3f(0, 0, len)

    glEnd()

def drawBox(
    x_min,
    x_max,
    y_min,
    y_max,
    z_min,
    z_max
):
    glBegin(GL_QUADS)

    glNormal3f(0, 0, 1)
    glVertex3f(x_max, y_max, z_max)
    glVertex3f(x_min, y_max, z_max)
    glVertex3f(x_min, y_min, z_max)
    glVertex3f(x_max, y_min, z_max)

    glNormal3f(0, 0, -1)
    glVertex3f(x_min, y_max, z_min)
    glVertex3f(x_max, y_max, z_min)
    glVertex3f(x_max, y_min, z_min)
    glVertex3f(x_min, y_min, z_min)

    glNormal3f(-1, 0, 0)
    glVertex3f(x_min, y_max, z_max)
    glVertex3f(x_min, y_max, z_min)
    glVertex3f(x_min, y_min, z_min)
    glVertex3f(x_min, y_min, z_max)

    glNormal3f(1, 0, 0)
    glVertex3f(x_max, y_max, z_min)
    glVertex3f(x_max, y_max, z_max)
    glVertex3f(x_max, y_min, z_max)
    glVertex3f(x_max, y_min, z_min)

    glNormal3f(0, 1, 0)
    glVertex3f(x_max, y_max, z_max)
    glVertex3f(x_max, y_max, z_min)
    glVertex3f(x_min, y_max, z_min)
    glVertex3f(x_min, y_max, z_max)

    glEnd()

class ErgoJr(Entity):
    """
    Visual simulation of the Poppy ergo-jr robot arm
    """

    def __init__(self):
        super().__init__()

        self.radius = 0.1
        self.height = 0.4

        self.angles = [0] * 6

    def drawGripper(self):
        # TODO: last motor, rotate right plate around the Y axis
        # TODO: just hack the visuals for now, can refine later

        pass






    def drawSeg4(self):
        """
        Fourth segment, horizontal, rotates around +Y axis
        """

        glRotatef(self.angles[3], 0, 1, 0)
        drawAxes()

        glPushMatrix()
        glColor3f(0.7, 0.7, 0.7)
        glTranslatef(0.015, 0.01, 0)
        drawBoxOld(0.05, 0.02, 0.02)
        glPopMatrix()

        #glPushMatrix()
        #glTranslatef(0, 0.07, 0)
        #self.drawSeg4()
        #glPopMatrix()








    def drawSeg3(self):
        """
        Third segment, vertical, rotates around +Z axis
        """

        glRotatef(self.angles[2], 0, 0, 1)
        drawAxes()

        glPushMatrix()
        glColor3f(1, 1, 1)
        glTranslatef(0, 0.015, 0)
        drawBoxOld(0.02, 0.04, 0.02)
        glPopMatrix()

        glPushMatrix()
        glTranslatef(0, 0.04, 0)
        self.drawSeg4()
        glPopMatrix()

    def drawSeg2(self):
        """
        Second segment, vertical, rotates around +Z axis
        """

        glRotatef(self.angles[1], 0, 0, 1)
        drawAxes()

        glPushMatrix()
        glColor3f(0.6, 0.6, 0.6)
        glTranslatef(0, 0.03, 0)
        drawBoxOld(0.02, 0.07, 0.02)
        glPopMatrix()

        glPushMatrix()
        glTranslatef(0, 0.07, 0)
        self.drawSeg3()
        glPopMatrix()





    def drawSeg1(self):
        """
        First segment, vertical, rotates around +Y atop the base
        """

        glRotatef(self.angles[0], 0, 1, 0)
        drawAxes()

        glColor3f(1, 1, 1)
        drawBox(
            x_min=-0.01,
            x_max=+0.01,
            y_min=+0.00,
            y_max=+0.03,
            z_min=-0.01,
            z_max=+0.01
        )

        glPushMatrix()
        glTranslatef(0, 0.03, 0)
        #self.drawSeg2()
        glPopMatrix()

    def drawBase(self):
        """
        Robot base, sits on the ground plane, doesn't move
        """

        #drawAxes()

        # Base sits above Y=0
        glColor3f(0.5, 0.5, 0.5)
        drawBox(
            x_min=-0.02,
            x_max=+0.01,
            y_min=-0.00,
            y_max=+0.02,
            z_min=-0.01,
            z_max=+0.01
        )

        # First segment rotates above base
        glPushMatrix()
        glTranslatef(0, 0.02, 0)
        self.drawSeg1()
        glPopMatrix()

    def render(self):
        glDisable(GL_TEXTURE_2D)

        glPushMatrix()

        # Rotate and translate to the robot's position
        glTranslatef(*self.pos)
        glRotatef(self.dir * (180/math.pi), 0, 1, 0)
        self.drawBase()

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
        self.ergojr.angles = [ self.rand.float(-10, 10) for i in range(6) ]

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
