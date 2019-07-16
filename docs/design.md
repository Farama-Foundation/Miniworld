# Design and Customization

MiniWorld is essentially a very simple 3D game engine. At a few thousand lines of code, it is [minimalistic by design](https://pointersgonewild.com/2018/02/18/minimalism-in-programming/). It has a limited feature set, but its simplicity means that it's easier to troubleshoot and relatively easy to customize based on your needs. If you need any help, please never hesitate to open an issue on this repository, even if it's just to ask a question.

## The World

In MiniWorld, the world is made of static elements (rooms and hallways), as well as objects which may be dynamic, which we call entities. Environments are essentially 2D floorplans made of connected rooms. Rooms can have any convex outline defined by at least 3 points. Portals (openings) can be created in walls to create doors or windows into other rooms. Hallways are themselves small rooms with some walls removed. To get an idea how to create and connect rooms, you should take a look at the implementation of the [ThreeRooms environment](/gym_miniworld/envs/threerooms.py).

The entities are defined in [gym_miniworld/entity.py](/gym_miniworld/entity.py) and include:

- The agent/robot
- Movable colored boxes
- Image frames (to display pictures on walls)

## Coordinate System

MiniWorld uses OpenGL's right-handed coordinate system. The ground plane lies along the X and Z axes, and the Y axis points up. The coordinate units are in meters by convention. When direction angles are specified, a positive angle corresponds to a counter-clockwise (leftward) rotation. Angles are in degrees for ease of hand-editing. By convention, angle zero points towards the positive X axis.

## Observations

The observations are single camera images, as numpy arrays of size (80, 60, 3). These arrays contain unsigned 8-bit integer values (`uint8`) in the [0, 255] range. It is possible to change the observation image size by directly instantiating the environment class (`MiniWorldEnv`) and setting the appropriate parameters in the constructor.

<p align="center">
<img src="/images/maze_top_view.jpg" width=260></img><br>
Top view of the Maze environment
</p>

A top-down fully observable view of the environment can be produced as well. To produce this view, you can call the `env.render_top_view()` method, which returns a NumPy RGB array as output.

## Actions

For simplicity, actions are discrete. The default available actions are:
- `turn_left`
- `turn_right`
- `move_forward`
- `move_back`
- `pickup` (pick up an object in front of the agent)
- `drop` (drop an object being carried)
- `toggle` (toggle an item/entity to perform some function)

The turn and move actions will rotate or move the agent by a small fixed interval. The simulator assumes that the agent behaves like a [differential drive](https://groups.csail.mit.edu/drl/courses/cs54-2001s/diffdrive.html) robot. However, it is possible to implement new actions and different motion dynamics in your own environments.

## Reward Function

Each environment has an associated `max_episode_steps` variable which specifies the maximum number of time steps allowed to complete an episode. By default, rewards are sparse and in the [0, 1] range, with a small penalty being given based on the number of time steps needed to successfully complete the task. If the task is not completed within `max_episode_steps`, the current episode is terminated and a reward of 0 is produced. See the `_reward()` method of `MiniWorldEnv`.

### Loading 3D Models

MiniWorld has built-in support for [OBJ mesh files](https://en.wikipedia.org/wiki/Wavefront_.obj_file). Some 3D models are included under the [gym_miniworld/meshes](https://github.com/maximecb/gym-miniworld/tree/master/gym_miniworld/meshes) directory. These models are all tested and working with MiniWorld. In order to load a 3D model in an environment, you should create a `MeshEnt` object and specify the name of the model to load. See the [sidewalk environment](https://github.com/maximecb/gym-miniworld/blob/master/gym_miniworld/envs/sidewalk.py) for an example of how to do this. The `height` parameter specifies how to scale the model, based on the height you want it to have, in meters.


<p align="center">
<img src="/images/sidewalk_0.jpg" width=260></img><br>
Street cone 3D models
</p>

You can find many more 3D models to load on the [OpenGameArt](https://opengameart.org/) website. Make sure to check the OBJ checkbox when searching, and that these models are compatible with the open source license of your project. Something to be aware of is that MiniWorld only supports triangle polygons. If you want to load a mesh that contains non-triangular polygons, you can convert it to triangles only by loading it in [Blender](https://www.blender.org/) (a free model-editing program), and re-exporting them with the triangulate mesh option. If you want to create your own custom 3D models, I recommend the [Wings 3D](http://www.wings3d.com/) editor, which is relatively simple and easy to use.
