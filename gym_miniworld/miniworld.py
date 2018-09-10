import math
from enum import IntEnum
import numpy as np
import gym
import pybullet
from .random import *
from .opengl import *
#from .objmesh import *
from .entity import *

# Texture size/density in texels/meter
TEX_DENSITY = 512

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

        # Height of the room walls
        self.wall_height = wall_height

        self.wall_tex = Texture.get('concrete')
        self.floor_tex = Texture.get('floor_tiles_bw')
        self.ceil_tex = Texture.get('concrete_tiles')

        # List of portals
        self.portals = []

        # List of neighbor rooms
        # Same length as portals
        self.neighbors = []

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

        self.ceil_verts = self.outline + up_vec
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

        # TODO: start with different colors for floor, walls, ceiling
        # random colors?

        glEnable(GL_TEXTURE_2D)

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
        do_nothing = 0

        # Turn left or right by a small amount
        turn_left = 1
        turn_right = 2

        # Move forward or back by a small amount
        move_forward = 3
        move_back = 4

        # Pitch the camera up or down
        look_up = 5
        look_down = 6

        # Pick up or drop an object being carried
        pickup = 7
        drop = 8

        # Toggle/activate an object
        toggle = 9

        # Done completing task
        done = 10

    def __init__(
        self,
        max_episode_steps=1500,
        frame_rate=30,
        obs_width=210,
        obs_height=160,
        window_width=800,
        window_height=600,
        domain_rand=True
    ):
        # Action enumeration for this environment
        self.actions = MiniWorldEnv.Actions

        # Actions are discrete integer values
        self.action_space = gym.spaces.Discrete(len(self.actions))

        # Observations are RGB images with pixels in [0, 255]
        # The default observation size matches that of Atari environment
        # for compatibility with existing RL frameworks.
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(obs_height, obs_width, 3),
            dtype=np.uint8
        )

        self.reward_range = (0, 1)

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

        # Frame buffer used to render observations
        self.obs_fb = FrameBuffer(obs_width, obs_height)

        # Frame buffer used for human visualization
        self.vis_fb = FrameBuffer(window_width, window_height, 8)

        """
        # For displaying text
        self.text_label = pyglet.text.Label(
            font_name="Arial",
            font_size=14,
            x = 5,
            y = WINDOW_HEIGHT - 19
        )
        """

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
        self.agent = Agent()

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
            self.agent.position = self.agent.position + self.agent.dir_vec * 0.05

        elif action == self.actions.move_back:
            self.agent.position = self.agent.position - self.agent.dir_vec * 0.05

        elif action == self.actions.turn_left:
            self.agent.direction += math.pi * 0.025

        elif action == self.actions.turn_right:
            self.agent.direction -= math.pi * 0.025

        # TODO: update the world state, objects, etc.

        # Generate the current camera image
        obs = self.render_obs()

        # If the maximum time step count is reached
        if self.step_count >= self.max_episode_steps:
            done = True
            reward = 0
            return obs, reward, done, {}

        # TODO: reward computation
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

        self.rooms.append(Room(outline))

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
        glClearColor(0, 0, 0, 1.0)
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

        # Render the image
        img = self.render_obs(self.vis_fb)

        if mode == 'rgb_array':
            return img

        width = img.shape[1]
        height = img.shape[0]

        if self.window is None:
            config = pyglet.gl.Config(double_buffer=False)
            self.window = pyglet.window.Window(
                width=width,
                height=height,
                resizable=False,
                config=config
            )

        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()

        # Bind the default frame buffer
        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        # Setup orghogonal projection
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glOrtho(0, width, 0, height, 0, 10)

        # Draw the image to the rendering window
        img = np.ascontiguousarray(np.flip(img, axis=0))
        img_data = pyglet.image.ImageData(
            width,
            height,
            'RGB',
            img.ctypes.data_as(POINTER(GLubyte)),
            pitch=width * 3,
        )
        img_data.blit(
            0,
            0,
            0,
            width=width,
            height=height
        )

        # Force execution of queued commands
        glFlush()
