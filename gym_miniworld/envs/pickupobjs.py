import numpy as np
import math
from gym import spaces
from ..miniworld import MiniWorldEnv, Room
from ..entity import Box, Ball, Key

class PickupObjs(MiniWorldEnv):
    """
    Room with multiple objects. The agent collects +1 reward for picking up
    each object. Objects disappear when picked up.
    """

    def __init__(self, size=12, num_objs=5, **kwargs):
        assert size >= 2
        self.size = size
        self.num_objs = num_objs

        super().__init__(
            max_episode_steps=400,
            **kwargs
        )

        # Reduce the action space
        self.action_space = spaces.Discrete(self.actions.pickup+1)

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

        obj_types = [Ball, Box, Key]

        for obj in range(self.num_objs):
            obj_type = self.rand.choice(obj_types)
            color = self.rand.color()

            if obj_type == Box:
                self.place_entity(Box(color=color, size=0.9))
            if obj_type == Ball:
                self.place_entity(Ball(color=color, size=0.9))
            if obj_type == Key:
                self.place_entity(Key(color=color))

        self.place_agent()

        self.num_picked_up = 0

    def step(self, action):
        obs, reward, done, info = super().step(action)

        if self.agent.carrying:
            self.entities.remove(self.agent.carrying)
            self.agent.carrying = None
            self.num_picked_up += 1
            reward = 1

            if self.num_picked_up == self.num_objs:
                done = True

        return obs, reward, done, info
