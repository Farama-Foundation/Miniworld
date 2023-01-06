import inspect

import gymnasium as gym

from miniworld.envs.collecthealth import CollectHealth
from miniworld.envs.fourrooms import FourRooms
from miniworld.envs.hallway import Hallway
from miniworld.envs.maze import Maze, MazeS2, MazeS3, MazeS3Fast
from miniworld.envs.oneroom import OneRoom, OneRoomS6, OneRoomS6Fast
from miniworld.envs.pickupobjects import PickupObjects
from miniworld.envs.putnext import PutNext
from miniworld.envs.roomobjects import RoomObjects
from miniworld.envs.sidewalk import Sidewalk
from miniworld.envs.sign import Sign
from miniworld.envs.threerooms import ThreeRooms
from miniworld.envs.tmaze import TMaze, TMazeLeft, TMazeRight
from miniworld.envs.wallgap import WallGap
from miniworld.envs.ymaze import YMaze, YMazeLeft, YMazeRight

# Registered environment ids
from miniworld.miniworld import MiniWorldEnv

env_ids = []


def register_envs():
    module_name = __name__
    global_vars = globals()

    # Iterate through global names
    for global_name in sorted(list(global_vars.keys())):
        env_class = global_vars[global_name]

        if not inspect.isclass(env_class):
            continue

        if not issubclass(env_class, gym.core.Env):
            continue

        if env_class is MiniWorldEnv:
            continue

        # Register the environment with OpenAI Gym
        gym_id = f"MiniWorld-{global_name}-v0"
        entry_point = f"{module_name}:{global_name}"

        gym.envs.registration.register(
            id=gym_id,
            entry_point=entry_point,
        )

        env_ids.append(gym_id)

        # print('Registered env:', gym_id)


register_envs()
