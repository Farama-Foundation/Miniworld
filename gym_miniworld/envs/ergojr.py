import numpy as np
import math
from ..opengl import *
from ..miniworld import MiniWorldEnv, Room
from ..params import DEFAULT_PARAMS
from ..entity import Entity, Box

# Simulation parameters
sim_params = DEFAULT_PARAMS.copy()
sim_params.set('forward_step', 0.035)
sim_params.set('forward_drift', 0)
sim_params.set('turn_step', 17)
sim_params.set('bot_radius', 0.11)
sim_params.set('cam_pitch', -10)
sim_params.set('cam_fov_y', 49)
sim_params.set('cam_height', 0.18)
sim_params.set('cam_fwd_disp', 0)

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
        """
        Gripper. Right plate rotates around the -Y axis
        Positive angles open the gripper
        """

        #glColor3f(0.9, 0.9, 0.9)
        glColor3f(1, 0, 0)

        drawBox(
            x_min=+0,
            x_max=+0.05,
            y_min=+0.00,
            y_max=+0.02,
            z_min=-0.015,
            z_max=-0.010
        )

        glColor3f(1, 1, 0)
        angle = np.clip(self.angles[5], -25, 90)
        glRotatef(angle, 0, -1, 0)
        drawBox(
            x_min=+0,
            x_max=+0.05,
            y_min=+0.00,
            y_max=+0.02,
            z_min=+0.01,
            z_max=+0.015
        )

    def drawSeg5(self):
        """
        Fifth segment, horizontal, rotates around -Z axis
        Positive angles rotate segment downwards
        """

        glRotatef(self.angles[4], 0, 0, -1)
        #drawAxes()

        glColor3f(1, 1, 1)
        drawBox(
            x_min=-0.01,
            x_max=+0.04,
            y_min=+0.00,
            y_max=+0.02,
            z_min=-0.01,
            z_max=+0.01
        )

        glPushMatrix()
        glTranslatef(0.03, 0, 0)
        self.drawGripper()
        glPopMatrix()

    def drawSeg4(self):
        """
        Fourth segment, horizontal, rotates around +Y axis
        """

        glRotatef(self.angles[3], 0, 1, 0)
        #drawAxes()

        glColor3f(0.4, 0.4, 0.4)
        drawBox(
            x_min=-0.01,
            x_max=+0.04,
            y_min=+0.00,
            y_max=+0.02,
            z_min=-0.01,
            z_max=+0.01
        )

        glPushMatrix()
        glTranslatef(0.03, 0, 0)
        self.drawSeg5()
        glPopMatrix()

    def drawSeg3(self):
        """
        Third segment, vertical, rotates around -Z axis
        Positive angles rotate segment downwards
        """

        glRotatef(self.angles[2], 0, 0, -1)
        #drawAxes()

        glColor3f(1, 1, 1)
        drawBox(
            x_min=-0.01,
            x_max=+0.01,
            y_min=+0.00,
            y_max=+0.04,
            z_min=-0.01,
            z_max=+0.01
        )

        glPushMatrix()
        glTranslatef(0, 0.04, 0)
        self.drawSeg4()
        glPopMatrix()

    def drawSeg2(self):
        """
        Second segment, vertical, rotates around -Z axis
        Positive angles rotate segment downwards
        """

        glRotatef(self.angles[1], 0, 0, -1)
        #drawAxes()

        glColor3f(0.4, 0.4, 0.4)
        drawBox(
            x_min=-0.01,
            x_max=+0.01,
            y_min=+0.00,
            y_max=+0.06,
            z_min=-0.01,
            z_max=+0.01
        )

        glPushMatrix()
        glTranslatef(0, 0.05, 0)
        self.drawSeg3()
        glPopMatrix()

    def drawSeg1(self):
        """
        First segment, vertical, rotates around +Y atop the base
        """

        glRotatef(self.angles[0], 0, 1, 0)
        #drawAxes()

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
        glTranslatef(0, 0.02, 0)
        self.drawSeg2()
        glPopMatrix()

    def drawBase(self):
        """
        Robot base, sits on the ground plane, doesn't move
        """

        # Base sits above Y=0
        glColor3f(0.4, 0.4, 0.4)
        drawBox(
            x_min=-0.02,
            x_max=+0.01,
            y_min=-0.00,
            y_max=+0.02,
            z_min=-0.01,
            z_max=+0.01
        )

        # Base plate
        glColor3f(1, 1, 1)
        drawBox(
            x_min=-0.15,
            x_max=+0.01,
            y_min=-0.00,
            y_max=+0.003,
            z_min=-0.08,
            z_max=+0.08
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
            domain_rand=True,
            **kwargs
        )

    def _gen_world(self):
        room = self.add_rect_room(
            min_x=-0.15,
            max_x=0.7,
            min_z=-0.7,
            max_z=0.7,
            wall_height=0,
            no_ceiling=True,
            floor_tex=self.rand.choice(['concrete', 'white', 'drywall'])
        )

        # The box looks the same from all sides, so restrict angles to [0, 90]
        self.box = self.place_entity(
            Box(color='green', size=0.03),
            min_x=0.05,
            max_x=0.2,
            min_z=-0.1,
            max_z=+0.1,
            dir=self.rand.float(0, math.pi/2)
        )
        self.box.pos[1] = self.rand.float(0, 0.15)

        self.ergojr = self.place_entity(ErgoJr(), pos=[0, 0, 0], dir=0)

        self.ergojr.angles = [
            self.rand.float(-90, 90),
            self.rand.float(-40, 40),
            self.rand.float(-40, 40),
            0,
            self.rand.float(-40, 40),
            self.rand.float(-30, 30),
        ]

        self.entities.append(self.agent)
        self.agent.radius = 0.15
        self.agent.dir = self.rand.float(-2.0, -2.4)
        self.agent.pos = np.array([
            self.rand.float(0.28, 0.32),
            0,
            self.rand.float(-0.28, -0.32)
        ])

    def step(self, action):
        obs, reward, done, info = super().step(action)
        return obs, reward, done, info
