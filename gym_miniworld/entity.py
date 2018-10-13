import numpy as np
from .objmesh import *

class Entity:
    def __init__(self, radius=None):
        # World position
        # Note: for most entities, the position is at floor level
        self.pos = None

        # Direction/orientation angle in radians
        self.dir = None

        # Radius for bounding circle/cylinder
        self.radius = radius

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
    def dir_vec(self):
        """
        Vector pointing in the direction of forward movement
        """

        x = math.cos(self.dir)
        z = -math.sin(self.dir)
        return np.array([x, 0, z])

    @property
    def right_vec(self):
        """
        Vector pointing to the right of the agent
        """

        x = math.sin(self.dir)
        z = math.cos(self.dir)
        return np.array([x, 0, z])

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
    Note: the position is at ceiling level
    """

    def __init__(self, pos, dir):
        super().__init__(pos, dir)

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

        glEnable(GL_LIGHTING)

class Box(Entity):
    """
    Colored box object
    """

    def __init__(self, color, size=0.8):
        super().__init__(radius=1.41 * (size/2))
        self.color = color
        self.size = size

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
        glNormal3f(0, 0, 1)
        glVertex3f(x + hs, y + sz, z + hs)
        glVertex3f(x - hs, y + sz, z + hs)
        glVertex3f(x - hs, y     , z + hs)
        glVertex3f(x + hs, y     , z + hs)

        glNormal3f(0, 0, -1)
        glVertex3f(x - hs, y + sz, z - hs)
        glVertex3f(x + hs, y + sz, z - hs)
        glVertex3f(x + hs, y     , z - hs)
        glVertex3f(x - hs, y     , z - hs)

        glNormal3f(-1, 0, 0)
        glVertex3f(x - hs, y + sz, z + hs)
        glVertex3f(x - hs, y + sz, z - hs)
        glVertex3f(x - hs, y     , z - hs)
        glVertex3f(x - hs, y     , z + hs)

        glNormal3f(1, 0, 0)
        glVertex3f(x + hs, y + sz, z - hs)
        glVertex3f(x + hs, y + sz, z + hs)
        glVertex3f(x + hs, y     , z + hs)
        glVertex3f(x + hs, y     , z - hs)

        glNormal3f(0, 1, 0)
        glVertex3f(x + hs, y + sz, z + hs)
        glVertex3f(x + hs, y + sz, z - hs)
        glVertex3f(x - hs, y + sz, z - hs)
        glVertex3f(x - hs, y + sz, z + hs)

        glEnd(GL_QUADS)

class Agent(Entity):
    def __init__(self):
        super().__init__(radius=0.4)

        # Distance between the camera and the floor
        self.cam_height = 1.5

        # Camera up/down angle
        self.cam_angle = 0

        # Vertical field of view in degrees
        self.cam_fov_y = 60

    @property
    def cam_pos(self):
        return self.pos + np.array([0, self.cam_height, 0])

    @property
    def cam_dir(self):
        # FIXME: take cam_angle into account
        # NOTE: this is useful even if just for slight domain
        # randomization of camera angle
        return self.dir_vec

    def step(self, delta_time):
        pass
