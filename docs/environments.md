# List of MiniWorld Environments

For details about the observations and action space, consult the
[MiniWorld Design](/docs/design.md) document.

# Hallway
In this environment, the agent has to go to a red box at the end of a
straight hallway. Requires the agent to walk forward and correct its course
to avoid bumping into walls. Think of this environment as a kind of "Hello World!".
It is easily solved with reinforcement learning and is useful for debugging purposes,
to verify that your training code is working correctly.

<p align="center">
    <img src="/images/hallway_0.jpg" width=300 alt="Figure of Hallway v0 environment">
</p>


Registered configurations:
- `MiniWorld-Hallway-v0`

# OneRoom

One single large room in which the agent has to navigate to a red box.
Requires the agent to turn around and scan the room to find the red
box.

<p align="center">
    <img src="/images/oneroom_0.jpg" width=300 alt="Figure of OneRoom v0 environment">
</p>

Registered configurations:
- `MiniWorld-OneRoom-v0`
- `MiniWorld-OneRoomS6-v0`
- `MiniWorld-OneRoomS6Fast-v0`

# T-Maze

In this environment, there is a T-junction. The agent has to get to the
junction point, and turn either left or right to reach a red box, which
is randomly placed at one end or the other. The environment comes in three
variants. In the default variant, the red box is randomly placed in the left
or right arm of the maze in each episode. In the `MiniWorld-TMazeLeft-v0`
variant, the box is always in the left arm, and in the `MiniWorld-TMazeRight-v0`
variant, the box is always in the right arm. This can be used to experiment
with transfer learning.

<p align="center">
    <img src="/images/tmaze_0.jpg" width=300 alt="Figure of T-Maze v0 environment">
</p>

Registered configurations:
- `MiniWorld-TMaze-v0`
- `MiniWorld-TMazeLeft-v0`
- `MiniWorld-TMazeRight-v0`

# Y-Maze

Similar to the T-Maze environment, but with a Y-shaped junction instead of
a T-shaped junction. This environment can be used to test skill transfer
between it and the T-Maze environment.

<p align="center">
    <img src="/images/ymaze_0.jpg" width=300 alt="Figure of Y-Maze v0 environment">
</p>

Registered configurations:
- `MiniWorld-YMaze-v0`
- `MiniWorld-YMazeLeft-v0`
- `MiniWorld-YMazeRight-v0`

# Maze

Navigate to a goal through a procedurally generated maze. 
The largest version of this environment (`MiniWorld-Maze-v0`) is extremely 
hard to solve as it has a sparse reward and a long time horizon. 
The `MazeS3Fast` environment has faster movement actions (bigger time steps) 
making it easier to solve than the regular `MazeS3`.

<p align="center">
<img src="/images/maze_0.jpg" width=300 alt="Figure of Maze v0 environment">
</p>

Registered configurations:
- `MiniWorld-Maze-v0`
- `MiniWorld-MazeS3-v0`
- `MiniWorld-MazeS3Fast-v0`
- `MiniWorld-MazeS2-v0`

# FourRooms

Inspired by the classic four-rooms gridworld environment. The agent appears
at a random position inside 4 rooms connected by 4 openings. In order to
get a reward, the agent must reach a red box.

<p align="center">
    <img src="/images/fourrooms_0.jpg" width=300 alt="Figure of FourRoom environment">
</p>

Registered configurations:
- `MiniWorld-FourRooms-v0`

# Sidewalk

The agent must walk to an object at the end of the sidewalk, while avoiding
walking into the street. Reaching the object provides a positive reward,
but walking into the street terminates the episode with zero reward.

<p align="center">
    <img src="/images/sidewalk_0.jpg" width=300 alt="Figure of Sidewalk environment">
</p>

Registered configurations:
- `MiniWorld-Sidewalk-v0`

# PickupObjs

One single large room in which the agent has to collect several objects. The
agent gets +1 reward for collecting each object. The episode terminates when
all objects are collected or when the time step limit is exceeded.

<p align="center">
    <img src="/images/pickupobjs_0.jpg" width=300 alt="Figure of Pickup Objects environment">
</p>

Registered configurations:
- `MiniWorld-PickupObjs-v0`

# CollectHealth

Inspired by VizDoom's HealthGathering environment. The agent is placed inside
a room filled with acid and has to collect medkits in order to survive. The
reward corresponds to the survival time of the agent. Please note that the
rewards produced in this environment are not directly comparable to VizDoom's
as the dynamics are not exactly the same.

<p align="center">
    <img src="/images/collecthealth_0.jpg" width=300 alt="Figure of Collect health environment">
</p>

Registered configurations:
- `MiniWorld-CollectHealth-v0`

# PutNext

There are multiple colored boxes of random sizes in one large room. In order
to get a reward, the agent must put the red box next to the yellow box.

Registered configurations:
- `MiniWorld-PutNext-v0`

# Sign

There are 6 objects of color (red, green blue) and shape (key, box) in a
U-shaped maze. The agent starts on one side of the barrier in the U-shaped
maze, and on the other side of the barrier is a sign that says "blue,"
"green," or "red." The sign is highlighted above in yellow. Additionally, the
state includes a goal that specifies either key or box. In order to get
reward, the agent must read the sign and go to the object with shape specified
by the goal and color specified by the sign. Going to any other object yields
-1 reward.

Note that the state structure differs from the standard MiniWorld state. In
particular, the state is a dict where `state["obs"]` is the standard
observation, and `state["goal"]` is an additional int specifying key or box.

This environment is from the paper [Decoupling Exploration and Exploitation
for Meta-Reinforcement Learning without Sacrifices](https://arxiv.org/abs/2008.02790).
If you use this environment, please cite this paper:

```
@inproceedings{liu2021decoupling,
  title={Decoupling Exploration and Exploitation for Meta-Reinforcement Learning without Sacrifices},
  author={Liu, Evan Z and Raghunathan, Aditi and Liang, Percy and Finn, Chelsea},
  booktitle={International Conference on Machine Learning},
  pages={6925--6935},
  year={2021},
  organization={PMLR}
}
```

<p align="center">
    <img src="/images/sign.jpg" width=300 alt="Figure of Sign environment">
</p>

Registered configurations:
- `MiniWorld-Sign-v0`

# RemoteBot

This is a fake environment that uses ZMQ to connect remotely to a small robot (MiniBot). 
The robot uses differential drive and discrete actions that match those of MiniWorld. 
This makes it possible to do sim-to-real transfer experiments. 
Note that domain randomization needs to be enabled to make this work. 
This [repository](https://github.com/maximecb/minibot-iface) contains the code that runs
on the robot and interfaces with the `RemoteBot` environment.

<p align="center">
    < src="/images/minibot.jpg" width=300>
</p>

Registered configurations:
- `MiniWorld-RemoteBot-v0`
