import numpy as np
from .objmesh import *

class Entity:
    def __init__(self):
        pass

    def render(self):
        """
        Draw the object
        """
        raise NotImplementedError

    def step(self, delta_time):
        """
        Update the state of the object
        """
        pass

    @property
    def is_static(self):
        """
        True for objects that cannot move or animate
        (can be rendered statically)
        """
        return False

class CeilingLight(Entity):
    """
    Ceiling light object
    """

    def __init__(self, x, y, z):
        super().__init__()
        self.pos = np.array([x, y, z])

    def is_static(self):
        return True

    def render(self):
        """
        Draw the object
        """

        x, y, z = self.pos

        glDisable(GL_LIGHTING)
        glDisable(GL_TEXTURE_2D)

        glBegin(GL_QUADS)
        glVertex3f(x - 0.5, y - 0.05, z + 0.5)
        glVertex3f(x + 0.5, y - 0.05, z + 0.5)
        glVertex3f(x + 0.5, y - 0.05, z - 0.5)
        glVertex3f(x - 0.5, y - 0.05, z - 0.5)
        glEnd(GL_QUADS)

class Box(Entity):
    """
    Colored box object
    """

    def __init__(self, x, y, z, color, angle=0, size=0.5):
        super().__init__()
        self.color = color
        self.size = size
        self.pos = np.array([x, y, z])

    def is_static(self):
        return True

    def render(self):
        """
        Draw the object
        """

        from .miniworld import COLORS

        sz = self.size
        hs = sz / 2
        x, y, z = self.pos

        glDisable(GL_TEXTURE_2D)
        glColor3f(*COLORS[self.color])

        glBegin(GL_QUADS)
        glVertex3f(x - hs, y + sz, z + hs)
        glVertex3f(x + hs, y + sz, z + hs)
        glVertex3f(x + hs, y     , z + hs)
        glVertex3f(x - hs, y     , z + hs)

        glVertex3f(x - hs, y + sz, z - hs)
        glVertex3f(x + hs, y + sz, z - hs)
        glVertex3f(x + hs, y     , z - hs)
        glVertex3f(x - hs, y     , z - hs)

        glVertex3f(x - hs, y + sz, z - hs)
        glVertex3f(x - hs, y + sz, z + hs)
        glVertex3f(x - hs, y     , z + hs)
        glVertex3f(x - hs, y     , z - hs)

        glVertex3f(x + hs, y + sz, z + hs)
        glVertex3f(x + hs, y + sz, z - hs)
        glVertex3f(x + hs, y     , z - hs)
        glVertex3f(x + hs, y     , z + hs)

        glVertex3f(x + hs, y + sz, z + hs)
        glVertex3f(x + hs, y + sz, z - hs)
        glVertex3f(x - hs, y + sz, z - hs)
        glVertex3f(x - hs, y + sz, z + hs)

        glEnd(GL_QUADS)

class Agent(Entity):
    def __init__(self):
        super().__init__()

        # Position of the agent (at floor level)
        self.position = np.array([0, 0, 0])

        # Direction angle in radians
        # Angle zero points towards the positive X axis
        self.direction = 0

        # Distance between the camera and the floor
        self.cam_height = 1.5

        # Camera up/down angle
        self.cam_angle = 0

        # Vertical field of view
        self.cam_fov_y = 60

    @property
    def dir_vec(self):
        """
        Vector pointing in the direction of forward movement
        """

        x = math.cos(self.direction)
        z = -math.sin(self.direction)
        return np.array([x, 0, z])

    @property
    def right_vec(self):
        """
        Vector pointing to the right of the agent
        """

        x = math.sin(self.direction)
        z = math.cos(self.direction)
        return np.array([x, 0, z])

    @property
    def cam_pos(self):
        return self.position + np.array([0, self.cam_height, 0])

    @property
    def cam_dir(self):
        # FIXME: take cam_angle into account
        return self.dir_vec

    def step(self, delta_time):
        pass
