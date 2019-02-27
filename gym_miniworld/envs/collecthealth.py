import numpy as np
import math
from ..miniworld import MiniWorldEnv, Room
from ..entity import MeshEnt

class CollectHealth(MiniWorldEnv):
    """
    Environment where the agent has to collect healthkits and stay
    alive as long as possible. This is inspired from the VizDoom
    HealthGathering environment. Please note, however, that the rewards
    produced by this environment are not directly comparable to those
    of the VizDoom environment.

    reward:
    +2 for each time step
    -100 for dying
    """

    def __init__(self, size=16, **kwargs):
        assert size >= 2
        self.size = size

        super().__init__(
            max_episode_steps=1000,
            **kwargs
        )

    def _gen_world(self):
        # Create a long rectangular room
        room = self.add_rect_room(
            min_x=0,
            max_x=self.size,
            min_z=0,
            max_z=self.size,
            wall_tex='cinder_blocks',
            floor_tex='slime'
        )

        for _ in range(18):
            self.box = self.place_entity(MeshEnt(
                mesh_name='medkit',
                height=0.40,
                static=False
            ))

        # Place the agent a random distance away from the goal
        self.place_agent()

        self.health = 100

    def step(self, action):
        obs, reward, done, info = super().step(action)

        self.health -= 2

        # If the agent picked up a health kit
        if action == self.actions.pickup:
            if self.agent.carrying:
                # Respawn the health kit
                self.entities.remove(self.agent.carrying)
                self.place_entity(self.agent.carrying)
                self.agent.carrying = None

                # Reset the agent's health
                self.health = 100

        if self.health > 0:
            reward = 2
        else:
            reward = -100
            done = True

        # Pass current health value in info dict
        info['health'] = self.health

        return obs, reward, done, info
