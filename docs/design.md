# Design of MiniWorld

## The World

In MiniWorld, the world is made of static elements (rooms and hallways), as well as objects which may be dynamic, which we call entities. Environments are essentially 2D floorplans made of connected rooms. Rooms can have any convex outline defined by at least 3 points. Portals (openings) can be created in walls to create doors or windows into other rooms. Hallways are themselves small rooms with some walls completely open.

The entities are defined in [gym_miniworld/entity.py](/gym_miniworld/entity.py) and include:

- The agent/robot
- Movable colored boxes
- Image frames (to display pictures on walls)

## Coordinate System

MiniWorld uses OpenGL's right-handed coordinate system. The ground plane lies along the X and Z axes, and the Y axis points up. When direction angles are specified, a positive angle corresponds to a counter-clockwise (leftward) rotation. Angles are in degrees for ease of hand-editing. By convention, angle zero points towards the positive X axis.

## Observations

The observations are single camera images, as numpy arrays of size (80, 60, 3). These arrays contain unsigned 8-bit integer values in the [0, 255] range. It is possible to change the observation image size by directly instantiating the environment class and setting the appropriate
parameters in the constructor.

## Actions

For simplicity, actions are discrete. The available actions are:
- `turn_left`
- `turn_right`
- `move_forward`
- `move_back`
- `pickup` (pick up an object in front of the agent)
- `drop` (drop an object being carried)
- `toggle` (toggle an item/entity to perform some function)

The turn and move actions will rotate or move the agent by a small fixed interval. The simulator assumes that the agent is a [differential drive](https://groups.csail.mit.edu/drl/courses/cs54-2001s/diffdrive.html) robot.

## Reward Function

Each environment has an associated `max_episode_steps` variable which specifies the maximum number of time steps allowed to complete an episode. By default, rewards are sparse and in the [0, 1] range, with a small penalty being given based on the number of time steps needed to complete the task. If the task is not completed within `max_episode_steps`, a reward of 0 is produced. See the `_reward()` method of `MiniWorldEnv`.

