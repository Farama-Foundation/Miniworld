import gymnasium as gym

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
