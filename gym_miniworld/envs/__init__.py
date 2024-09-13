import inspect

import gym

from gym_miniworld.envs.collecthealth import CollectHealth
from gym_miniworld.envs.fourrooms import FourRooms
from gym_miniworld.envs.hallway import Hallway
from gym_miniworld.envs.maze import Maze, MazeS2, MazeS3, MazeS3Fast, MazeS4, MazeS5, MazeS6, MazeS7
from gym_miniworld.envs.oneroom import OneRoom, OneRoomS6, OneRoomS6Fast
from gym_miniworld.envs.pickupobjs import PickupObjs
from gym_miniworld.envs.putnext import PutNext
from gym_miniworld.envs.remotebot import RemoteBot
from gym_miniworld.envs.roomobjs import RoomObjs
from gym_miniworld.envs.sidewalk import Sidewalk
from gym_miniworld.envs.sign import Sign
from gym_miniworld.envs.simtorealgoto import SimToRealGoTo
from gym_miniworld.envs.simtorealpush import SimToRealPush
from gym_miniworld.envs.threerooms import ThreeRooms
from gym_miniworld.envs.tmaze import TMaze, TMazeLeft, TMazeRight
from gym_miniworld.envs.wallgap import WallGap
from gym_miniworld.envs.ymaze import YMaze, YMazeLeft, YMazeRight

# Registered environment ids
from gym_miniworld.miniworld import MiniWorldEnv

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
