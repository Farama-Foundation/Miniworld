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

__all__ = [
    "CollectHealth",
    "FourRooms",
    "Hallway",
    "Maze",
    "MazeS2",
    "MazeS3",
    "MazeS3Fast",
    "OneRoom",
    "OneRoomS6",
    "OneRoomS6Fast",
    "PickupObjects",
    "PutNext",
    "RoomObjects",
    "Sidewalk",
    "Sign",
    "ThreeRooms",
    "TMaze",
    "TMazeLeft",
    "TMazeRight",
    "WallGap",
    "YMaze",
    "YMazeLeft",
    "YMazeRight",
]

gym.register(
    id="MiniWorld-CollectHealth-v0",
    entry_point="miniworld.envs.collecthealth:CollectHealth",
)

gym.register(
    id="MiniWorld-FourRooms-v0",
    entry_point="miniworld.envs.fourrooms:FourRooms",
)

gym.register(
    id="MiniWorld-Hallway-v0",
    entry_point="miniworld.envs.hallway:Hallway",
)

gym.register(
    id="MiniWorld-Maze-v0",
    entry_point="miniworld.envs.maze:Maze",
)

gym.register(
    id="MiniWorld-MazeS2-v0",
    entry_point="miniworld.envs.maze:MazeS2",
)

gym.register(
    id="MiniWorld-MazeS3-v0",
    entry_point="miniworld.envs.maze:MazeS3",
)

gym.register(
    id="MiniWorld-MazeS3Fast-v0",
    entry_point="miniworld.envs.maze:MazeS3Fast",
)

gym.register(
    id="MiniWorld-OneRoom-v0",
    entry_point="miniworld.envs.oneroom:OneRoom",
)

gym.register(
    id="MiniWorld-OneRoomS6-v0",
    entry_point="miniworld.envs.oneroom:OneRoomS6",
)

gym.register(
    id="MiniWorld-OneRoomS6Fast-v0",
    entry_point="miniworld.envs.oneroom:OneRoomS6Fast",
)

gym.register(
    id="MiniWorld-PickupObjects-v0",
    entry_point="miniworld.envs.pickupobjects:PickupObjects",
)

gym.register(
    id="MiniWorld-PutNext-v0",
    entry_point="miniworld.envs.putnext:PutNext",
)

gym.register(
    id="MiniWorld-RoomObjects-v0",
    entry_point="miniworld.envs.roomobjects:RoomObjects",
)

gym.register(
    id="MiniWorld-Sidewalk-v0",
    entry_point="miniworld.envs.sidewalk:Sidewalk",
)

gym.register(
    id="MiniWorld-Sign-v0",
    entry_point="miniworld.envs.sign:Sign",
)

gym.register(
    id="MiniWorld-TMaze-v0",
    entry_point="miniworld.envs.tmaze:TMaze",
)

gym.register(
    id="MiniWorld-TMazeLeft-v0",
    entry_point="miniworld.envs.tmaze:TMazeLeft",
)

gym.register(
    id="MiniWorld-TMazeRight-v0",
    entry_point="miniworld.envs.tmaze:TMazeRight",
)

gym.register(
    id="MiniWorld-ThreeRooms-v0",
    entry_point="miniworld.envs.threerooms:ThreeRooms",
)

gym.register(
    id="MiniWorld-WallGap-v0",
    entry_point="miniworld.envs.wallgap:WallGap",
)

gym.register(
    id="MiniWorld-YMaze-v0",
    entry_point="miniworld.envs.ymaze:YMaze",
)

gym.register(
    id="MiniWorld-YMazeLeft-v0",
    entry_point="miniworld.envs.ymaze:YMazeLeft",
)

gym.register(
    id="MiniWorld-YMazeRight-v0",
    entry_point="miniworld.envs.ymaze:YMazeRight",
)
