import math
import numpy as np
from .opengl import *
from .objmesh import ObjMesh

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

    def draw_bound(self):
        """
        Draw the bounding circle
        Used for debugging purposes
        """

        x, _, z = self.pos

        glColor3f(1, 0, 0)
        glBegin(GL_LINES)

        for i in range(60):
            a = i * 2 * math.pi / 60
            cx = x + self.radius * math.cos(a)
            cz = z + self.radius * math.sin(a)
            glVertex3f(cx, 0.01, cz)

        glEnd(GL_LINES)

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

class MeshEnt(Entity):
    """
    Entity whose appearance is defined by a mesh file

    height -- scale the model to this height
    static -- flag indicating this object cannot move
    radius -- collision radius, None means no collision detection
    """

    def __init__(
        self,
        mesh_name,
        height,
        static=True,
        radius=None
    ):
        super().__init__(radius=radius)

        self.static = static

        self.mesh = ObjMesh.get(mesh_name)

        # Compute the mesh scaling factor
        self.scale = height / mesh.max_coords[1]

    def render(self):
        """
        Draw the object
        """

        glPushMatrix()
        glTranslatef(*self.cur_pos)
        glScalef(self.scale, self.scale, self.scale)
        glRotatef(self.dir * 180 / math.pi, 0, 1, 0)
        glColor3f(1, 1, 1)
        self.mesh.render()
        glPopMatrix()

    @property
    def is_static(self):
        return self.static

class ImageFrame(Entity):
    """
    Frame to display an image on a wall
    Note: the position is in the middle of the frame, on the wall
    """

    def __init__(self, pos, dir, tex_name, width):
        super().__init__(radius=0)

        self.pos = pos
        self.dir = dir

        # Load the image to be displayed
        self.tex = Texture.get(tex_name)

        self.width = width
        self.height = (float(self.tex.height) / self.tex.width) * self.width

    def is_static(self):
        return True

    def render(self):
        """
        Draw the object
        """

        x, y, z = self.pos

        # sx is depth
        # Frame points towards +sx
        sx = 0.05
        hz = self.width / 2
        hy = self.height / 2

        glPushMatrix()
        glTranslatef(*self.pos)
        glRotatef(self.dir * (180/math.pi), 0, 1, 0)

        # Bind texture for front
        glColor3f(1, 1, 1)
        glEnable(GL_TEXTURE_2D)
        self.tex.bind()

        # Front face, showing image
        glBegin(GL_QUADS)
        glNormal3f(1, 0, 0)
        glTexCoord2f(1, 1)
        glVertex3f(sx, +hy, -hz)
        glTexCoord2f(0, 1)
        glVertex3f(sx, +hy, +hz)
        glTexCoord2f(0, 0)
        glVertex3f(sx, -hy, +hz)
        glTexCoord2f(1, 0)
        glVertex3f(sx, -hy, -hz)
        glEnd(GL_QUADS)

        # Black frame/border
        glDisable(GL_TEXTURE_2D)
        glColor3f(0, 0, 0)

        glBegin(GL_QUADS)

        # Left
        glNormal3f(0, 0, -1)
        glVertex3f(0  , +hy, -hz)
        glVertex3f(+sx, +hy, -hz)
        glVertex3f(+sx, -hy, -hz)
        glVertex3f(0  , -hy, -hz)

        # Right
        glNormal3f(0, 0, 1)
        glVertex3f(+sx, +hy, +hz)
        glVertex3f(0  , +hy, +hz)
        glVertex3f(0  , -hy, +hz)
        glVertex3f(+sx, -hy, +hz)

        # Top
        glNormal3f(0, 1, 0)
        glVertex3f(+sx, +hy, +hz)
        glVertex3f(+sx, +hy, -hz)
        glVertex3f(0  , +hy, -hz)
        glVertex3f(0  , +hy, +hz)

        # Bottom
        glNormal3f(0, -1, 0)
        glVertex3f(+sx, -hy, -hz)
        glVertex3f(+sx, -hy, +hz)
        glVertex3f(0  , -hy, +hz)
        glVertex3f(0  , -hy, -hz)

        glEnd(GL_QUADS)

        glPopMatrix()

class Box(Entity):
    """
    Colored box object
    """

    def __init__(self, color, size=0.8):
        if type(size) is int or type(size) is float:
            size = np.array([size, size, size])
        size = np.array(size)
        sx, _, sz = size
        super().__init__(radius=math.sqrt(sx*sx + sz*sz)/2)
        self.color = color
        self.size = size

    def render(self):
        """
        Draw the object
        """

        from .miniworld import COLORS

        sx, sy, sz = self.size
        hx = sx / 2
        hz = sz / 2

        glDisable(GL_TEXTURE_2D)
        glColor3f(*COLORS[self.color])

        glPushMatrix()
        glTranslatef(*self.pos)
        glRotatef(self.dir * (180/math.pi), 0, 1, 0)

        glBegin(GL_QUADS)
        glNormal3f(0, 0, 1)
        glVertex3f(+hx, +sy, +hz)
        glVertex3f(-hx, +sy, +hz)
        glVertex3f(-hx, 0  , +hz)
        glVertex3f(+hx, 0  , +hz)

        glNormal3f(0, 0, -1)
        glVertex3f(-hx, +sy, -hz)
        glVertex3f(+hx, +sy, -hz)
        glVertex3f(+hx, 0  , -hz)
        glVertex3f(-hx, 0  , -hz)

        glNormal3f(-1, 0, 0)
        glVertex3f(-hx, +sy, +hz)
        glVertex3f(-hx, +sy, -hz)
        glVertex3f(-hx, 0  , -hz)
        glVertex3f(-hx, 0  , +hz)

        glNormal3f(1, 0, 0)
        glVertex3f(+hx, +sy, -hz)
        glVertex3f(+hx, +sy, +hz)
        glVertex3f(+hx, 0  , +hz)
        glVertex3f(+hx, 0  , -hz)

        glNormal3f(0, 1, 0)
        glVertex3f(+hx, +sy, +hz)
        glVertex3f(+hx, +sy, -hz)
        glVertex3f(-hx, +sy, -hz)
        glVertex3f(-hx, +sy, +hz)
        glEnd(GL_QUADS)

        glPopMatrix()

class Agent(Entity):
    def __init__(self):
        super().__init__(radius=0.4)

        # Distance between the camera and the floor
        self.cam_height = 1.5

        # Camera up/down angle
        self.cam_angle = 0

        # Vertical field of view in degrees
        self.cam_fov_y = 60

        # Object currently being carried by the agent
        self.carrying = None

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
