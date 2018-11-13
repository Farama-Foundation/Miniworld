import math
import numpy as np
from .math import *
from .opengl import *
from .objmesh import ObjMesh

class Entity:
    def __init__(self):
        # World position
        # Note: for most entities, the position is at floor level
        self.pos = None

        # Direction/orientation angle in radians
        self.dir = None

        # Radius for bounding circle/cylinder
        self.radius = 0

        # Height of bounding cylinder
        self.height = 0

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
    """

    def __init__(
        self,
        mesh_name,
        height,
        static=True
    ):
        super().__init__()

        self.static = static

        # Load the mesh
        self.mesh = ObjMesh.get(mesh_name)

        # Get the mesh extents
        sx, sy, sz = self.mesh.max_coords

        # Compute the mesh scaling factor
        self.scale = height / sy

        # Compute the radius and height
        self.radius = math.sqrt(sx*sx + sz*sz) * self.scale
        self.height = height

    def render(self):
        """
        Draw the object
        """

        glPushMatrix()
        glTranslatef(*self.pos)
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
        super().__init__()

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
        super().__init__()

        if type(size) is int or type(size) is float:
            size = np.array([size, size, size])
        size = np.array(size)
        sx, sy, sz = size

        self.color = color
        self.size = size

        self.radius = math.sqrt(sx*sx + sz*sz)/2
        self.height = sy

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
        super().__init__()

        # Distance between the camera and the floor
        self.cam_height = 1.5

        # Camera up/down angles in degrees
        # Positive angles tilt the camera upwards
        self.cam_angle = 0

        # Vertical field of view in degrees
        self.cam_fov_y = 60

        # Bounding cylinder size for the agent
        self.radius = 0.4
        self.height = 1.6

        # Object currently being carried by the agent
        self.carrying = None

    @property
    def cam_pos(self):
        """
        Camera position in 3D space
        """

        return self.pos + np.array([0, self.cam_height, 0])

    @property
    def cam_dir(self):
        """
        Camera direction (lookat) vector

        Note: this is useful even if just for slight domain
        randomization of camera angle
        """

        rot_z = gen_rot_matrix(Z_VEC, self.cam_angle * math.pi/180)
        rot_y = gen_rot_matrix(Y_VEC, self.dir)

        dir = np.dot(X_VEC, rot_z)
        dir = np.dot(dir, rot_y)

        return dir

    def render(self):
        """
        Draw the agent
        """

        # Note: this is currently only used in the top view
        # Eventually, we will want a proper 3D model

        p = self.pos + Y_VEC * self.height
        dv = self.dir_vec * self.radius
        rv = self.right_vec * self.radius

        p0 = p + dv
        p1 = p + 0.75 * (rv - dv)
        p2 = p + 0.75 * (-rv - dv)

        glColor3f(1, 0, 0)
        glBegin(GL_TRIANGLES)
        glVertex3f(*p0)
        glVertex3f(*p2)
        glVertex3f(*p1)
        glEnd()

        """
        glBegin(GL_LINE_STRIP)
        for i in range(20):
            a = (2 * math.pi * i) / 20
            pc = p + dv * math.cos(a) + rv * math.sin(a)
            glVertex3f(*pc)
        glEnd()
        """

    def step(self, delta_time):
        pass
