import numpy as np
import math
from ..miniworld import MiniWorldEnv, Room
from ..entity import Box

class TMazeEnv(MiniWorldEnv):
    """
    Two hallways connected in a T-junction
    """

    def __init__(self, **kwargs):
        super().__init__(
            max_episode_steps=140,
            **kwargs
        )

    def _gen_world(self):
        room1 = self.add_rect_room(
            min_x=-1, max_x=8,
            min_z=-2, max_z=2
        )
        room2 = self.add_rect_room(
            min_x=8, max_x=12,
            min_z=-8, max_z=8
        )
        room1.add_portal(0, min_z=-2, max_z=2)
        room2.add_portal(2, min_z=-2, max_z=2)

        # Add a box at a random end of the hallway
        self.box = Box(color='red')
        if self.rand.bool():
            self.place_entity(self.box, room=room2, min_z=room2.max_z - 2)
        else:
            self.place_entity(self.box, room=room2, max_z=room2.min_z + 2)

        # Choose a random room and position to spawn at
        self.place_agent(
            dir=self.rand.float(-math.pi/4, math.pi/4),
            room=room1
        )

    def step(self, action):
        obs, reward, done, info = super().step(action)

        dist = np.linalg.norm(self.agent.pos - self.box.pos)

        # TODO: use forward movement step size once defined
        if dist < 1.5 * (self.agent.radius + self.box.radius):
            reward += self._reward()
            done = True

        return obs, reward, done, info
