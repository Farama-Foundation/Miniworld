# List of MiniWorld Environments

For details about the observations and action space, consult the
[MiniWorld Design](/docs/design.md) document.

# Hallway

Registered configurations:
- `MiniWorld-Hallway-v0`

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

Randomly generated maze, in which the agent starts in a random location, and
has to navigate to a red box placed at another random location. This is a
very challenging environment to solve with reinforcement learning.

# PutNext

Registered configurations:
- `MiniWorld-PutNext-v0`

There are multiple colored boxes of random sizes in one large room. In order
to get a reward, the agent must put the red box next to the yellow box.
