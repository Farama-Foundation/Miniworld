import math
from enum import IntEnum
import numpy as np
import gym
#import pybullet
from .random import *
from .opengl import *
#from .objmesh import *
from .entity import *

# Texture size/density in texels/meter
TEX_DENSITY = 512

# Blue sky horizon color
BLUE_SKY_COLOR = np.array([0.45, 0.82, 1])

# Map of color names to RGB values
COLORS = {
    'red'   : np.array([255, 0, 0]),
    'green' : np.array([0, 255, 0]),
    'blue'  : np.array([0, 0, 255]),
    'purple': np.array([112, 39, 195]),
    'yellow': np.array([255, 255, 0]),
    'grey'  : np.array([100, 100, 100])
}

# List of color names, sorted alphabetically
COLOR_NAMES = sorted(list(COLORS.keys()))

def gen_tex_coords(tex, width, height):
    w = width * (TEX_DENSITY / tex.width)
    h = height * (TEX_DENSITY / tex.height)

    return np.array(
        [
            [0, 0],
            [0, h],
            [w, h],
            [w, 0],
        ],
        dtype=np.float32
    )

class Room:
    """
    Represent an individual room and its contents
    """

    def __init__(
        self,
        outline,
        wall_height=2.74
    ):
        # The outlien should have shape Nx2
        assert len(outline.shape) == 2
        assert outline.shape[1] == 2

        # Add a Y coordinate to the outline points
        outline = np.insert(outline, 1, 0,axis=1)

        # Number of outline vertices / walls
        self.num_walls = outline.shape[0]

        # List of 2D points forming the outline of the room
        # Shape is Nx3
        self.outline = outline

        # Compute the min and max x, z extents
        self.min_x = self.outline[:, 0].min()
        self.max_x = self.outline[:, 0].max()
        self.min_z = self.outline[:, 2].min()
        self.max_z = self.outline[:, 2].max()

        # Height of the room walls
        self.wall_height = wall_height

        self.wall_tex = Texture.get('concrete')
        self.floor_tex = Texture.get('floor_tiles_bw')
        self.ceil_tex = Texture.get('concrete_tiles')

        # List of portals
        self.portals = []

        # List of neighbor rooms
        # Same length as list of portals
        self.neighbors = []

        # List of entities contained
        self.entities = []

    def _gen_polys(self):
        """
        Generate polygons for this room
        Needed for rendering and collision detection
        Note: the wall polygons are quads, but the floor and
              ceiling can be arbitrary n-gons
        """

        up_vec = np.array([0, self.wall_height, 0])

        self.floor_verts = self.outline
        self.floor_texcs = gen_tex_coords(
            self.floor_tex,
            np.linalg.norm(self.outline[2,:] - self.outline[1,:]),
            np.linalg.norm(self.outline[1,:] - self.outline[0,:])
        )

        # Flip the ceiling vertex order because of backface culling
        self.ceil_verts = np.flip(self.outline, axis=0) + up_vec
        self.ceil_texcs = gen_tex_coords(
            self.ceil_tex,
            np.linalg.norm(self.outline[2,:] - self.outline[1,:]),
            np.linalg.norm(self.outline[1,:] - self.outline[0,:])
        )

        self.wall_verts = []
        self.wall_texcs = []

        for i in range(self.num_walls):
            p0 = self.outline[i, :]
            p1 = self.outline[(i+1) % self.num_walls, :]
            side_vec = p1 - p0
            wall_width = np.linalg.norm(side_vec)

            self.wall_verts.append(p0)
            self.wall_verts.append(p0+up_vec)
            self.wall_verts.append(p1+up_vec)
            self.wall_verts.append(p1)

            texcs = gen_tex_coords(
                self.wall_tex,
                wall_width,
                self.wall_height
            )
            self.wall_texcs.append(texcs)

        self.wall_verts = np.array(self.wall_verts)
        self.wall_texcs = np.concatenate(self.wall_texcs)

    def _render(self):
        """
        Render the static elements of the room
        """

        glEnable(GL_TEXTURE_2D)
        glColor3f(1, 1, 1)

        # Draw the floor
        self.floor_tex.bind()
        glBegin(GL_POLYGON)
        for i in range(self.floor_verts.shape[0]):
            glTexCoord2f(*self.floor_texcs[i, :])
            glVertex3f(*self.floor_verts[i, :])
        glEnd()

        # Draw the ceiling
        self.ceil_tex.bind()
        glBegin(GL_POLYGON)
        for i in range(self.ceil_verts.shape[0]):
            glTexCoord2f(*self.ceil_texcs[i, :])
            glVertex3f(*self.ceil_verts[i, :])
        glEnd()

        # Draw the walls
        self.wall_tex.bind()
        glBegin(GL_QUADS)
        for i in range(self.wall_verts.shape[0]):
            glTexCoord2f(*self.wall_texcs[i, :])
            glVertex3f(*self.wall_verts[i, :])
        glEnd()

        # Render the static entities
        for ent in self.entities:
            if ent.is_static():
                ent.render()

class MiniWorldEnv(gym.Env):
    """
    Base class for MiniWorld environments. Implements the procedural
    world generation and simulation logic.
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 30
    }

    # Enumeration of possible actions
    class Actions(IntEnum):
        # Turn left or right by a small amount
        turn_left = 0
        turn_right = 1

        # Move forward or back by a small amount
        move_forward = 2
        move_back = 3

        # Pitch the camera up or down
        look_up = 4
        look_down = 5

        # Pick up or drop an object being carried
        pickup = 6
        drop = 7

        # Toggle/activate an object
        toggle = 8

        # Done completing task
        done = 9

        do_nothing = 10

    def __init__(
        self,
        max_episode_steps=1500,
        frame_rate=30,
        obs_width=80,
        obs_height=60,
        window_width=800,
        window_height=600,
        domain_rand=True
    ):
        # Action enumeration for this environment
        self.actions = MiniWorldEnv.Actions

        # Actions are discrete integer values
        self.action_space = gym.spaces.Discrete(len(self.actions))

        # Observations are RGB images with pixels in [0, 255]
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(obs_height, obs_width, 3),
            dtype=np.uint8
        )

        self.reward_range = (-math.inf, math.inf)

        # Maximum number of steps per episode
        self.max_episode_steps = max_episode_steps

        # Frame rate to run at
        self.frame_rate = frame_rate

        # Flag to enable/disable domain randomization
        self.domain_rand = domain_rand

        # Window for displaying the environment to humans
        self.window = None

        # Invisible window to render into (shadow OpenGL context)
        self.shadow_window = pyglet.window.Window(width=1, height=1, visible=False)

        # Enable depth testing and backface culling
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_CULL_FACE)

        # Frame buffer used to render observations
        self.obs_fb = FrameBuffer(obs_width, obs_height, 8)

        # Frame buffer used for human visualization
        self.vis_fb = FrameBuffer(window_width, window_height, 16)

        # Compute the observation display size
        self.obs_disp_width = 256
        self.obs_disp_height = obs_height * (self.obs_disp_width / obs_width)

        # For displaying text
        self.text_label = pyglet.text.Label(
            font_name="Arial",
            font_size=14,
            multiline=True,
            width=400,
            x = window_width + 5,
            y = window_height - (self.obs_disp_height + 19)
        )

        # Initialize the state
        self.seed()
        self.reset()

    def close(self):
        pass

    def seed(self, seed=None):
        self.rand = RandGen(seed)
        return [seed]

    def reset(self):
        """
        Reset the simulation at the start of a new episode
        This also randomizes many environment parameters (domain randomization)
        """

        # Step count since episode start
        self.step_count = 0

        # Create the agent
        self.agent = Agent([0, 0, 0], 0)

        # List of rooms in the world
        self.rooms = []

        # TODO: randomize elements of the world
        # Perform domain-randomization

        # Generate the world
        self._gen_world()

        # Generate the polygons for each room
        for room in self.rooms:
            room._gen_polys()

        # Pre-compile static parts of the environment into a display list
        self._render_static()

        # Generate the first camera image
        obs = self.render_obs()

        # Return first observation
        return obs

    def step(self, action):
        """
        Perform one action and update the simulation
        """

        self.step_count += 1

        delta_time = 1 / self.frame_rate

        if action == self.actions.move_forward:
            self.agent.pos = self.agent.pos + self.agent.dir_vec * 0.18

        elif action == self.actions.move_back:
            self.agent.pos = self.agent.pos - self.agent.dir_vec * 0.18

        elif action == self.actions.turn_left:
            self.agent.dir += math.pi * 0.04

        elif action == self.actions.turn_right:
            self.agent.dir -= math.pi * 0.04

        # TODO: update the world state, objects, etc.

        # Generate the current camera image
        obs = self.render_obs()

        # If the maximum time step count is reached
        if self.step_count >= self.max_episode_steps:
            done = True
            reward = 0
            return obs, reward, done, {}

        reward = 0
        done = False

        return obs, reward, done, {}

    def create_rect_room(
        self,
        min_x, min_z,
        size_x, size_z
    ):
        """
        Create a rectangular room
        """

        # 2D outline coordinates of the room
        outline = np.array([
            [min_x          , min_z],
            [min_x          , min_z + size_z],
            [min_x + size_x , min_z + size_z],
            [min_x + size_x , min_z]
        ])

        room = Room(outline)
        self.rooms.append(room)
        return room

    def _gen_world(self):
        """
        Generate the world. Derived classes must implement this method.
        """

        raise NotImplementedError

    def _render_static(self):
        """
        Render the static elements of the scene into a display list.
        Called once at the beginning of each episode.
        """

        # TODO: manage this automatically
        # glIsList
        glDeleteLists(1, 1);
        glNewList(1, GL_COMPILE);

        for room in self.rooms:
            room._render()

        glEndList()

    def _render_world(
        self,
        frame_buffer,
        cam_pos,
        cam_dir,
        cam_fov_y
    ):
        """
        Render the world from a given camera position into a frame buffer,
        and produce a numpy image array as output.
        """

        # Switch to the default context
        # This is necessary on Linux Nvidia drivers
        self.shadow_window.switch_to()

        # Bind the frame buffer before rendering into it
        frame_buffer.bind()

        # Clear the color and depth buffers
        glClearColor(*BLUE_SKY_COLOR, 1.0)
        glClearDepth(1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        # Set the projection matrix
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(
            cam_fov_y,
            frame_buffer.width / float(frame_buffer.height),
            0.04,
            100.0
        )

        # Setup the camera
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(
            # Eye position
            *cam_pos,
            # Target
            *(cam_pos + cam_dir),
            # Up vector
            0, 1.0, 0.0
        )

        # Call the display list for the static parts of the environment
        glCallList(1)

        # Resolve the rendered imahe into a numpy array
        return frame_buffer.resolve()

    def render_obs(self, frame_buffer=None):
        """
        Render an observation from the point of view of the agent
        """

        if frame_buffer == None:
            frame_buffer = self.obs_fb

        return self._render_world(
            frame_buffer,
            self.agent.cam_pos,
            self.agent.cam_dir,
            self.agent.cam_fov_y
        )

    def render(self, mode='human', close=False):
        """
        Render the environment for human viewing
        """

        if close:
            if self.window:
                self.window.close()
            return

        # Render the human-view image
        img = self.render_obs(self.vis_fb)
        img_width = img.shape[1]
        img_height = img.shape[0]

        if mode == 'rgb_array':
            return img

        # Render the neural network view
        obs = self.render_obs()
        obs_width = obs.shape[1]
        obs_height = obs.shape[0]

        window_width = img_width + self.obs_disp_width
        window_height = img_height

        if self.window is None:
            config = pyglet.gl.Config(double_buffer=True)
            self.window = pyglet.window.Window(
                width=window_width,
                height=window_height,
                resizable=False,
                config=config
            )

        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()

        # Bind the default frame buffer
        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        # Clear the color and depth buffers
        glClearColor(0, 0, 0, 1.0)
        glClearDepth(1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        # Setup orghogonal projection
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glOrtho(0, window_width, 0, window_height, 0, 10)

        # Draw the human render to the rendering window
        img = np.ascontiguousarray(np.flip(img, axis=0))
        img_data = pyglet.image.ImageData(
            img_width,
            img_height,
            'RGB',
            img.ctypes.data_as(POINTER(GLubyte)),
            pitch=img_width * 3,
        )
        img_data.blit(
            0,
            0,
            0,
            width=img_width,
            height=img_height
        )

        # Draw the observation
        obs = np.ascontiguousarray(np.flip(obs, axis=0))
        obs_data = pyglet.image.ImageData(
            obs_width,
            obs_height,
            'RGB',
            obs.ctypes.data_as(POINTER(GLubyte)),
            pitch=obs_width * 3,
        )
        obs_data.blit(
            img_width,
            img_height - self.obs_disp_height,
            0,
            width=self.obs_disp_width,
            height=self.obs_disp_height
        )

        # Draw the text label in the window
        self.text_label.text = "pos: (%.2f, %.2f, %.2f)\nangle: %d\nsteps: %d" % (
            *self.agent.pos,
            int(self.agent.dir * 180 / math.pi),
            self.step_count
        )
        self.text_label.draw()

        # Force execution of queued commands
        glFlush()

        # If we are not running the Pyglet event loop,
        # we have to manually flip the buffers
        if mode == 'human':
            self.window.flip()

        return None
