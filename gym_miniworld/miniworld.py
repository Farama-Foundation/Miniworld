import math
from enum import IntEnum
import numpy as np
import gym
import pybullet
from .random import *
from .opengl import *
#from .objmesh import *
from .entity import *

class Room:
    """
    Represent an individual room and its contents
    """

    def __init__(
        self,
        outline,
        wall_height=2.74
    ):
        assert len(outline) >= 3
        self.wall_height = wall_height

        # List of 2D points forming the outline of the room
        self.outline = outline

        # TODO: list of wall textures
        # Could limit to just one to start
        #self.wall_tex
        #self.floor_tex
        #self.ceil_tex

        # List of portals
        self.portals = []

        # List of neighbor rooms
        # Same length as portals
        self.neighbors = []

    def gen_polys(self):
        """
        Generate polygons for this room
        Needed for rendering and collision detection
        """

        # TODO: to begin, start with no portals case

        # Do we actually need this? Could just do it in render()
        # Yes, we need it for collision detection!




        pass

    def render(self):
        """
        Render the static elements of the room
        """

        # TODO: start with different colors for floor, walls, ceiling
        # random colors?

        #glBegin(GL_POLYGON) for floor and ceiling
        #glEnd()




        #glBegin(GL_QUADS) for walls
        #glEnd()


        pass

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

        """
        self.agent.position = np.array([
            self.rand.float(-0.5, 0.5),
            0,
            self.rand.float(-0.5, 0.5)
        ])
        """
        #self.agent.direction = self.rand.float(-math.pi/4, math.pi/4)

        # List of rooms in the world
        self.rooms = []

        # TODO: randomize elements of the world
        # Perform domain-randomization

        # Generate the world
        self._gen_world()

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

        # TODO: update the world

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

        # TODO: compute the outline coordinates

        #outline = np.ndarray([
        #[],
        #[],
        #[],
        #])

        #self.rooms.append(Room(outline))


        pass


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

        glNewList(1, GL_COMPILE);

        for room in self.rooms:
            room.render()

        for i in range(0, 100):
            glColor3f(1, 0, 0)
            glBegin(GL_TRIANGLES)
            glVertex3f(5, 2.0,-0.5)
            glVertex3f(5, 2.0, 0.5)
            glVertex3f(5, 1.0, 0.5)
            glEnd()

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
