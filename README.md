# MiniWorld (gym-miniworld)

[![Build Status](https://travis-ci.org/maximecb/gym-miniworld.svg?branch=master)](https://travis-ci.org/maximecb/gym-miniworld)

Contents:
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Environments](docs/environments.md)
- [Design and Customization](docs/design.md)
- [Troubleshooting](docs/troubleshooting.md)

## Introduction

MiniWorld is a minimalistic 3D interior environment simulator for reinforcement
learning &amp; robotics research. It can be used to simulate environments with
rooms, doors, hallways and various objects (eg: office and home environments, mazes).
MiniWorld can be seen as an alternative to VizDoom or DMLab. It is written
100% in Python and designed to be easily modified or extended.

<p align="center">
<img src="images/maze_top_view.jpg" width=260></img>
<img src="images/sidewalk_0.jpg" width=260></img>
<img src="images/collecthealth_0.jpg" width=260></img>
</p>

Features:
- Few dependencies, less likely to break, easy to install
- Easy to create your own levels, or modify existing ones
- Good performance, high frame rate, support for multiple processes
- Lightweight, small download, low memory requirements
- Provided under a permissive MIT license
- Comes with a variety of free 3D models and textures
- Fully observable [top-down/overhead view](images/maze_top_view.jpg) available
- [Domain randomization](https://blog.openai.com/generalizing-from-simulation/) support, for sim-to-real transfer
- Ability to [display alphanumeric strings](images/textframe.jpg) on walls
- Ability to produce depth maps matching camera images (RGB-D)

Limitations:
- Graphics are basic, nowhere near photorealism
- Physics are very basic, not sufficient for robot arms or manipulation

Please use this bibtex if you want to cite this repository in your publications:

```
@misc{gym_miniworld,
  author = {Chevalier-Boisvert, Maxime},
  title = {gym-miniworld environment for OpenAI Gym},
  year = {2018},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/maximecb/gym-miniworld}},
}
```

List of publications & submissions using MiniWorld (please open a pull request to add missing entries):
- [DeepAveragers: Offline Reinforcement Learning by Solving Derived Non-Parametric MDPs](https://arxiv.org/abs/2010.08891) (NeurIPS Offline RL Workshop, Oct 2020)
- [Explore then Execute: Adapting without Rewards via Factorized Meta-Reinforcement Learning](https://arxiv.org/abs/2008.02790) (Stanford University, Aug 2020)
- [Pre-trained Word Embeddings for Goal-conditional Transfer Learning in Reinforcement Learning](https://arxiv.org/abs/2007.05196) (University of Antwerp, Jul 2020, ICML 2020 LaReL Workshop)
- [Temporal Abstraction with Interest Functions](https://arxiv.org/abs/2001.00271) (Mila, Feb 2020, AAAI 2020)
- [Avoidance Learning Using Observational Reinforcement Learning](https://arxiv.org/abs/1909.11228) (Mila, McGill, Sept 2019)
- [Visual Hindsight Experience Replay](https://arxiv.org/pdf/1901.11529.pdf) (Georgia Tech, UC Berkeley, Jan 2019)

This simulator was created as part of work done at [Mila](https://mila.quebec/).

## Installation

Requirements:
- Python 3.5+
- OpenAI Gym
- NumPy
- Pyglet (OpenGL 3D graphics)
- GPU for 3D graphics acceleration (optional)

You can install all the dependencies with `pip3`:

```
git clone https://github.com/maximecb/gym-miniworld.git
cd gym-miniworld
pip3 install -e .
```

If you run into any problems, please take a look at the [troubleshooting guide](docs/troubleshooting.md), and if you're still stuck, please
[open an issue](https://github.com/maximecb/gym-miniworld/issues) on this
repository to let us know something is wrong.

## Usage

### Testing

There is a simple UI application which allows you to control the simulation or real robot manually. The `manual_control.py` application will launch the Gym environment, display camera images and send actions (keyboard commands) back to the simulator or robot. The `--env-name` argument specifies which environment to load. See the list of [available environments](docs/environments.md) for more information.

```
./manual_control.py --env-name MiniWorld-Hallway-v0

# Display an overhead view of the environment
./manual_control.py --env-name MiniWorld-Hallway-v0 --top_view
```

There is also a script to run automated tests (`run_tests.py`) and a script to gather performance metrics (`benchmark.py`).

### Reinforcement Learning

To train a reinforcement learning agent, you can use the code provided in the [/pytorch-a2c-ppo-acktr](/pytorch-a2c-ppo-acktr) directory. This code is a modified version of the RL code found in [this repository](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr). I recommend using the PPO algorithm and 16 processes or more. A sample command to launch training is:

```
python3 main.py --algo ppo --num-frames 5000000 --num-processes 16 --num-steps 80 --lr 0.00005 --env-name MiniWorld-Hallway-v0
```

Then, to visualize the results of training, you can run the following command. Note that you can do this while the training process is still running. Also note that if you are running this through SSH, you will need to enable X forwarding to get a display:

```
python3 enjoy.py --env-name MiniWorld-Hallway-v0 --load-dir trained_models/ppo
```
