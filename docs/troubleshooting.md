# Troubleshooting Guide

If you run into problems of any kind, don't hesitate to [open an issue](https://github.com/maximecb/gym-miniworld/issues) on this repository. It's quite possible that you've run into some bug we aren't aware of. Please make sure to give some details about your system configuration (ie: PC or Mac, operating system, etc), and to paste the command you used to run the simulator, as well as the complete error message that was produced, if any.

## ImportError: Library "GLU" not found

You may need to manually install packaged needed by Pyglet or OpenAI Gym on your system. The command you need to use will vary depending which OS you are running. For example, to install the glut package on Ubuntu:

```
sudo apt-get install freeglut3-dev
```

And on Fedora:

```
sudo dnf install freeglut-devel
```

## NoSuchDisplayException: Cannot connect to "None"

If you are connected through SSH, or running the simulator in a Docker image, you will need to use `xvfb-run` to create a virtual frame buffer (virtual display) in order to run the simulator. The following command can be used to test that the simularor is working correctly:

```
xvfb-run -a -s "-screen 0 1024x768x24 -ac +extension GLX +render -noreset" ./run_tests.py
```

Depending on which system you are running this command, you may need to install xvfb and mesa packages.

## Running headless and training on AWS

We recommend using the Ubuntu-based [Deep Learning AMI](https://aws.amazon.com/marketplace/pp/B077GCH38C) to provision your server which comes with all the deep learning libraries. To begin with, you will want to install xvfb and mesa. You may also need to uninstall the Nvidia display drivers in order for OpenGL/GLX to work properly:

```
# Install xvfb
sudo apt-get install xvfb mesa-utils -y

# Remove the nvidia display drivers (this doesn't remove the CUDA drivers)
# This is necessary as nvidia display doesn't play well with xvfb
sudo nvidia-uninstall -y

# Sanity check to make sure you still have CUDA driver installed
nvcc --version
```

Once this is done, you should be able to run training code through `xvfb-run`, for example:

```
cd pytorch-a2c-ppo-acktr
xvfb-run -a -s "-screen 0 1024x768x24 -ac +extension GLX +render -noreset" python3 main.py --algo ppo --num-processes 16 --num-steps 80 --lr 0.00005 --env-name MiniWorld-Hallway-v0
```

## Poor performance, low frame rate

It's possible to improve the performance of the simulator by disabling Pyglet error-checking code. Export this environment variable before running the simulator:

```
export PYGLET_DEBUG_GL=True
```

## Training not converging or taking too long to converge

If training is taking too long to converge, you can make environments easier by making them smaller, or by increasing the step size, that is, the amount of distance the agent can go forward or turns at each step. An example of an environment that does this is `MazeS3Fast`, which [can be found here](https://github.com/maximecb/gym-miniworld/blob/master/gym_miniworld/envs/maze.py#L123).

## Unknown encoder 'libx264' when using gym.wrappers.Monitor

It is possible to use `gym.wrappers.Monitor` to record videos of the agent performing a task. See [examples here](https://www.programcreek.com/python/example/100947/gym.wrappers.Monitor).

The libx264 error is due to a problem with the way ffmpeg is installed on some linux distributions. One possible way to circumvent this is to reinstall ffmpeg using conda:

```
conda install -c conda-forge ffmpeg
```

Alternatively, screencasting programs such as [Kazam](https://launchpad.net/kazam) can be used to record the graphical output of a single window.
