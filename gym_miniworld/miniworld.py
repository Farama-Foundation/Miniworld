import math
from enum import IntEnum
import numpy as np
import gym
import pybullet
from .random import *
from .opengl import *
#from .objmesh import *

class MiniWorld(gym.Env):
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
        self.actions = MiniWorld.Actions

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

        # TODO: randomize elements of the world




        # Generate the first camera image
        obs = self.render_obs()

        # Return first observation
        return obs

    def step(self, actions):
        """
        Perform one action and update the simulation
        """

        self.step_count += 1

        delta_time = 1 / self.frame_rate

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

    def _render_static():
        """
        Render the static elements of the scene into a display list.
        Called once at the beginning of each episode.
        """

        glNewList(0, GL_COMPILE);







        glEndList()

    def _render_world(self, frame_buffer):
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




        #glCallList(0)






        # Resolve the rendered imahe into a numpy array
        return frame_buffer.resolve()

    def render_obs(self):
        """
        Render an observation from the point of view of the agent
        """

        # TODO: pass appropriate camera parameter for the agent
        return self._render_world(self.obs_fb)

    def render(self, mode='human', close=False):
        """
        Render the environment for human viewing
        """

        if close:
            if self.window:
                self.window.close()
            return






        # TODO
        return None
