# gym-miniworld

MiniWorld is a minimalistic 3D interior environment simulator for reinforcement
learning &amp; robotics research. It can be used to simulate environments with
rooms, doors, hallways and various objects (eg: office and home environments).
MiniWorld can be seen as an alternative to VizDoom or DMLab. It is written
100% in Python and designed to be easily modified or extended.

Please use this bibtex if you want to cite this repository in your publications:

```
@misc{gym_miniworld,
  author = {Maxime Chevalier-Boisvert},
  title = {gym-miniworld environment for OpenAI Gym},
  year = {2018},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/maximecb/gym-miniworld}},
}
```

This simulator was created as part of work done at [Mila](https://mila.quebec/).

## Installation

Requirements:
- Python 3.5+
- OpenAI Gym
- NumPy
- Pyglet
- PyYAML

You can install all the dependencies with `pip3`:

```
git clone https://github.com/maximecb/gym-miniworld.git
cd gym-miniworld
pip3 install -e .
```

If you run into any problems, take a look at the troubleshooting
section below, and if you're still stuck, please
[open an issue](https://github.com/maximecb/gym-miniworld/issues) on this
repository to let us know something is wrong.

## Usage

TODO

## Design

TODO

### Observations

The observations are single camera images, as numpy arrays of size (210, 160, 3). These arrays contain unsigned 8-bit integer values in the [0, 255] range. This image format was chosen for compatibility, because it matches that of the Atari environments, which every RL framework out there tries to support. It is possible to configure the observation image size by directly instantiating the environment class and setting the appropriate
parameters in the constructor.

### Actions

TODO

### Reward Function

TODO

## Troubleshooting

If you run into problems of any kind, don't hesitate to [open an issue](https://github.com/maximecb/gym-miniworld/issues) on this repository. It's quite possible that you've run into some bug we aren't aware of. Please make sure to give some details about your system configuration (ie: PC or Max, operating system), and to paste the command you used to run the simulator, as well as the complete error message that was produced, if any.

### ImportError: Library "GLU" not found

You may need to manually install packaged needed by Pyglet or OpenAI Gym on your system. The command you need to use will vary depending which OS you are running. For example, to install the glut package on Ubuntu:

```
sudo apt-get install freeglut3-dev
```

And on Fedora:

```
sudo dnf install freeglut-devel

```

### NoSuchDisplayException: Cannot connect to "None"

If you are connected through SSH, or running the simulator in a Docker image, you will need to use xvfb to create a virtual display in order to run the simulator. See the "Running Headless" subsection below.

### Running headless and training in a cloud based environment (AWS)

We recommend using the Ubuntu-based [Deep Learning AMI](https://aws.amazon.com/marketplace/pp/B077GCH38C) to provision your server which comes with all the deep learning libraries.

```
# Install xvfb
sudo apt-get install xvfb mesa-utils -y

# Remove the nvidia display drivers (this doesn't remove the CUDA drivers)
# This is necessary as nvidia display doesn't play well with xvfb
sudo nvidia-uninstall -y

# Sanity check to make sure you still have CUDA driver and its version
nvcc --version

# Start xvfb
Xvfb :1 -screen 0 1024x768x24 -ac +extension GLX +render -noreset &> xvfb.log &

# Export your display id
export DISPLAY=:1

# Check if your display settings are valid
glxinfo

# You are now ready to train
```

### Poor performance, low frame rate

It's possible to improve the performance of the simulator by disabling Pyglet error-checking code. Export this environment variable before running the simulator:

```
export PYGLET_DEBUG_GL=True
```

### Unknown encoder 'libx264' when using gym.wrappers.Monitor

It is possible to use `gym.wrappers.Monitor` to record videos of the agent performing a task. See [examples here](https://www.programcreek.com/python/example/100947/gym.wrappers.Monitor).

The libx264 error is due to a problem with the way ffmpeg is installed on some linux distributions. One possible way to circumvent this is to reinstall ffmpeg using conda:

```
conda install -c conda-forge ffmpeg
```

Alternatively, screencasting programs such as [Kazam](https://launchpad.net/kazam) can be used to record the graphical output of a single window.
