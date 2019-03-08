#!/usr/bin/env python
# coding=utf-8

import gym
from gym import spaces
from gym.utils import seeding
import math
import numpy
import numpy as np
from ..miniworld import MiniWorldEnv
import pyglet
from pyglet.gl import *

# Try importing ZMQ
try:
    import zmq
except:
    zmq = None

# For Python 3 compatibility
import sys
if sys.version_info > (3,):
    buffer = memoryview

# Rendering window size
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600

# Port to connect to on the server
SERVER_PORT = 7777

def recv_array(socket):
    """
    Receive a numpy array over zmq
    """

    md = socket.recv_json()
    msg = socket.recv(copy=True, track=False)
    buf = buffer(msg)
    A = numpy.frombuffer(buf, dtype=md['dtype'])
    A = A.reshape(md['shape'])
    return A

class RemoteBot(gym.Env):
    """
    An environment that is an interface to remotely
    control an actual real robot
    """

    Actions = MiniWorldEnv.Actions

    metadata = {
        'render.modes': ['human', 'rgb_array', 'pyglet'],
        'video.frames_per_second' : 30
    }

    def __init__(
        self,
        serverAddr="minibot1.local",
        serverPort=SERVER_PORT,
        obs_width=80,
        obs_height=60
    ):
        assert zmq is not None, "Please install zmq (pip3 install zmq)"

        # Action enumeration for this environment
        self.actions = RemoteBot.Actions

        # Actions are discrete integer values
        self.action_space = spaces.Discrete(len(self.actions))

        # We observe an RGB image with pixels in [0, 255]
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(obs_height, obs_width, 3),
            dtype=np.uint8
        )

        self.obs_width = obs_width
        self.obs_height = obs_height

        self.reward_range = (0, 1)

        # Environment configuration
        self.max_episode_steps = math.inf

        # For rendering
        self.window = None

        # We continually stream in images and then just take the latest one.
        self.latest_img = None

        # For displaying text
        import pyglet
        self.textLabel = pyglet.text.Label(
            font_name="Arial",
            font_size=14,
            x = 5,
            y = WINDOW_HEIGHT - 19
        )

        # Connect to the Gym bridge ROS node
        addr_str = "tcp://%s:%s" % (serverAddr, serverPort)
        print("Connecting to %s ..." % addr_str)
        context = zmq.Context()
        self.socket = context.socket(zmq.PAIR)
        self.socket.connect(addr_str)

        # Initialize the state
        self.seed()
        self.reset()
        print('Connected')

    def close(self):
        # Stop the motors
        #self.step(numpy.array([0, 0]))
        pass

    def _recv_frame(self):
        # Receive a camera image from the server
        img = recv_array(self.socket)

        self.img = img

    def reset(self):
        # Step count since episode start
        self.step_count = 0

        self.socket.send_json({
            "command": "reset",
            "obs_width": self.obs_width,
            "obs_height": self.obs_height
        })

        # Receive a camera image from the server
        self._recv_frame()

        return self.img

    def seed(self, seed=None):
        self.np_random, _ = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        # Send the action to the server
        self.socket.send_json({
            "command": "action",
            "action": RemoteBot.Actions(action).name
        })

        # Receive a camera image from the server
        self._recv_frame()

        self.step_count += 1

        # We don't care about rewards or episodes since we're not training
        reward = 0
        done = False

        return self.img, reward, done, {}

    def render(self, mode='human', close=False):
        if close:
            if self.window:
                self.window.close()
            return

        if mode == 'rgb_array':
            return self.img

        if self.window is None:
            context = pyglet.gl.get_current_context()
            self.window = pyglet.window.Window(
                width=WINDOW_WIDTH,
                height=WINDOW_HEIGHT
            )

        self.window.switch_to()

        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT)

        self.window.clear()

        # Setup orghogonal projection
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glOrtho(0, WINDOW_WIDTH, 0, WINDOW_HEIGHT, 0, 10)

        # Draw the image to the rendering window
        img = np.ascontiguousarray(np.flip(self.img, axis=0))
        width = img.shape[1]
        height = img.shape[0]
        imgData = pyglet.image.ImageData(
            width,
            height,
            'RGB',
            img.tobytes(),
            #self.img.ctypes.data_as(POINTER(GLubyte)),
            pitch = width * 3,
        )
        imgData.blit(0, 0, 0, WINDOW_WIDTH, WINDOW_HEIGHT)

        # If we are not running the Pyglet event loop,
        # we have to manually flip the buffers and dispatch events
        if mode == 'human':
            self.window.flip()
            self.window.dispatch_events()
