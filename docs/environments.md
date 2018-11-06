# List of MiniWorld Environments

For details about the observations and action space, consult the
[MiniWorld Design](/docs/design.md) document.

# Hallway

Registered configurations:
- `MiniWorld-Hallway-v0`

<p align="center">
<img src="/images/hallway_0.jpg" width=300></img>
</p>

In this environment, the agent has to go to a red box at the end of a
straight hallway. Requires the agent to walk forward and correct its course
to avoid bumping into walls. This environment is easily solved with
reinforcement learning, it is useful for debugging purposes, to verify that
your training code is working correctly.

# OneRoom

Registered configurations:
- `MiniWorld-OneRoom-v0`

One single large room in which the agent has to navitage to a red box.
Requires the agent to turn around and scan the room to find the red
box.

# T-Maze

Registered configurations:
- `MiniWorld-TMaze-v0`

In this environment, there is a T-junction. The agent has to get to the
junction point, and turn either left or right to reach a red box, which
is randomly placed at one end or the other.

# Maze

Registered configurations:
- `MiniWorld-Maze-v0`
- `MiniWorld-MazeS3-v0`

<p align="center">
<img src="/images/maze_0.jpg" width=300></img>
</p>

Randomly generated maze, in which the agent starts in a random location, and
has to navigate to a red box placed at another random location. This is a
very challenging environment to solve with reinforcement learning.

# CollectHealth

Registered configurations:
- `MiniWorld-CollectHealth-v0`

<p align="center">
<img src="/images/collecthealth_0.jpg" width=300></img>
</p>

Inspired by VizDoom's HealthGathering environment. The agent is placed inside
a room filled with acid and has to collect medkits in order to survive. The
reward corresponds to the survival time of the agent. Please note that the
rewards produced in this environment are not directly comparable to VizDoom's
as the dynamics are not exactly the same.

# PutNext

Registered configurations:
- `MiniWorld-PutNext-v0`

There are multiple colored boxes of random sizes in one large room. In order
to get a reward, the agent must put the red box next to the yellow box.
