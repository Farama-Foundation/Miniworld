# Tutorial on Creating Environments

In this tutorial, we will go through the process of creating a new environment.

## Boilerplate Code

```python
def __init__(self, size=10, **kwargs):
    # Size of environment
    self.size = size

    super().__init__(self, **kwargs)

    # Allow only the movement actions
    self.action_space = spaces.Discrete(self.actions.move_forward + 1)
```

First, we need to create a class the inherits from `MiniWorldEnv`, we call our class `SimpleEnv`. Then, we define the action space to be only consisting of turn left (0), turn right (1), move forward (2), and move backward (3).

## Generate the walls

To generate the walls, we override the function `_gen_world`.

```python
def _gen_world(self):
    self.add_rect_room(min_x=0, max_x=self.size, min_z=0, max_z=self.size)
    self.place_agent()
```

The function `_gen_world` takes the arguments: `min_x`, `max_x`, `min_z`, `max_z`. Note that instead of using the X-Y plane, we use the X-Z plane for movement. After doing this, the environment should look like this:

```{figure} ../../images/tutorial_imgs/first_step.png
:alt: env after first step
:width: 500px
```

### Place Goal

To place a goal in the environment, we use the function

```python
self.box = self.place_entity(Box(color=COLOR_NAMES[0]), pos=np.array([4.5, 0.5, 4.5]), dir=0.0)
```

which places the goal in the middle. Now the environment should look like this:

```{figure} ../../images/tutorial_imgs/second_step.png
:alt: env after second step
:width: 500px
```

### Add reward 

To add a reward when the agent gets close to the box, we can do the following:

```python
def step(self, action):
    obs, reward, termination, truncation, info = super().step(action)

    if self.near(self.box):
        reward += self._reward()
        termination = True

    return obs, reward, termination, truncation, info
```
