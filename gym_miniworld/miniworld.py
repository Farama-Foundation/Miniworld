import math
from enum import IntEnum
import numpy as np
import gym
import pyglet
from pyglet.gl import *
import pybullet

from .random import *
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
        frame_skip=1,
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
            shape=(210, 160, 3),
            dtype=np.uint8
        )

        self.reward_range = (0, 1)

        # Maximum number of steps per episode
        self.max_episode_steps = max_episode_steps

        # Frame rate to run at
        self.frame_rate = frame_rate

        # Number of frames to skip per action
        self.frame_skip = frame_skip

        # Flag to enable/disable domain randomization
        self.domain_rand = domain_rand

        # Window for displaying the environment to humans
        self.window = None

        # Invisible window to render into (shadow OpenGL context)
        self.shadow_window = pyglet.window.Window(width=1, height=1, visible=False)

        """
        # For displaying text
        self.text_label = pyglet.text.Label(
            font_name="Arial",
            font_size=14,
            x = 5,
            y = WINDOW_HEIGHT - 19
        )

        # Create a frame buffer object for the observation
        self.multi_fbo, self.final_fbo = create_frame_buffers(
            CAMERA_WIDTH,
            CAMERA_HEIGHT,
            16
        )

        # Array to render the image into (for observation rendering)
        self.img_array = np.zeros(shape=self.observation_space.shape, dtype=np.uint8)

        # Create a frame buffer object for human rendering
        self.multi_fbo_human, self.final_fbo_human = create_frame_buffers(
            WINDOW_WIDTH,
            WINDOW_HEIGHT,
            4
        )

        # Array to render the image into (for human rendering)
        self.img_array_human = np.zeros(shape=(WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8)
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

        # Actions could be a Python list
        actions = np.array(actions)

        delta_time = 1 / self.frame_rate

        for _ in range(self.frame_skip):
            self.step_count += 1

            # TODO: update the world





        # Generate the current camera image
        obs = self.render_obs()

        # If the maximum time step count is reached
        if self.step_count >= self.max_episode_steps:
            done = True
            reward = 0
            return obs, reward, done, {}

        # TODO
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

    def render_obs(self):
        """
        Render an observation from the point of view of the agent
        """

        #glCallList(0)

        # TODO
        pass

    def render(self, mode='human', close=False):
        """
        Render the environment for viewing
        """

        if close:
            if self.window:
                self.window.close()
            return






        # TODO
        return None
