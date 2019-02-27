import numpy as np
import math
from ..miniworld import MiniWorldEnv, Room
from ..entity import Box, Ball, Key

class RoomObjs(MiniWorldEnv):
    """
    Single room with multiple objects
    Inspired by the single room environment of
    the Generative Query Networks paper:
    https://deepmind.com/blog/neural-scene-representation-and-rendering/
    """

    def __init__(self, size=10, **kwargs):
        assert size >= 2
        self.size = size

        super().__init__(
            max_episode_steps=math.inf,
            **kwargs
        )

    def _gen_world(self):
        room = self.add_rect_room(
            min_x=0,
            max_x=self.size,
            min_z=0,
            max_z=self.size,
            wall_tex='brick_wall',
            floor_tex='asphalt',
            no_ceiling=True,
        )

        # Reduce chances that objects are too close to see
        self.agent.radius=1.5

        self.place_entity(Box(color=self.rand.color(), size=0.9))

        self.place_entity(Ball(color=self.rand.color(), size=0.9))

        self.place_entity(Key(color=self.rand.color()))

        self.place_agent()

    def step(self, action):
        obs, reward, done, info = super().step(action)
        return obs, reward, done, info
