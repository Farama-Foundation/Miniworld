# PyTorch A2C and PPO deep reinforcement learning algorithms

The `torch_rl` package actually contains the PyTorch implementation of two Actor-Critic deep reinforcement learning algorithms:

- [Synchronous A3C (A2C)](https://arxiv.org/pdf/1602.01783.pdf)
- [Proximal Policy Optimizations (PPO)](https://arxiv.org/pdf/1707.06347.pdf)

The package comes out of the box with starter files:
- generic RL scripts (in `scripts`) to train, visualize and evaluate an agent
- a default agent's model (in `model.py`)
- CSV and Tensorboard logging for easy debugging

These files use the `torch_rl` package to allow you to immediatly train an agent on [MiniGrid](https://github.com/maximecb/gym-minigrid) environments **without having to write any line of code** and they can be easily adapted to other environments.

<p align="center">
    <img width="300" src="README-rsrc/visualize-keycorridor.gif">
</p>

## Features of `torch_rl`

- Support:
    - **Recurrent policies**
    - Reward shaping
    - Wide variety of observation spaces: tensors or dict of tensors
    - Wide variety of action spaces: discrete or continuous
    - Observation preprocessing
- Fast:
    - Multiprocess: parallelising experience collection
    - CUDA

## Installation

You have to clone the repository and then install the package:
```
git clone https://github.com/lcswillems/torch-rl.git
cd torch_rl
pip3 install -e torch_rl
```

To update the package, just do `git pull`. No need to install it again.

The starter files presented just below are best suited for [MiniGrid](https://github.com/maximecb/gym-minigrid) environments that you can install with:
```
git clone https://github.com/maximecb/gym-minigrid.git
cd gym-minigrid
pip3 install -e .
```

But the starter files can be easily adapted to other environments.

## Starter files

The `torch_rl` package comes out of the box with starter files:
- generic RL scripts:
    - `scripts/train.py` to train an agent
    - `scripts/visualize.py` to visualize how an agent behaves
    - `scripts/evaluate.py` to evaluate agent's performances
- a default agent's model: `model.py` (more details below)
- utilitarian classes and functions (in `utils`) used by the scripts

To adapt these files to your needs, you may want to modify:
- `model.py`
- `utils/format.py`

### `scripts/train.py`

An example of use:

`python3 -m scripts.train --algo ppo --env MiniGrid-DoorKey-5x5-v0 --model DoorKey --save-interval 10`

In this use case, the script loads the model in `storage/DoorKey` or creates it if it doesn't exist, then trains it with the PPO algorithm on the MiniGrid DoorKey environment, and saves it every 10 updates in the `storage/DoorKey` directory.

**Note:** You can define a different storage location in the environment variable `TORCH_RL_STORAGE`.

More generally, the script has 2 required arguments:
- `--algo ALGO`: name of the RL algorithm used to train
- `--env ENV`: name of the environment to train on

and a bunch of optional arguments among which:
- `--recurrence N`: gradient will be backpropagated over N timesteps. By default, N = 1. If N > 1, a LSTM is added to the model to have memory.
- `--text`: a GRU is added to the model to handle text input.
- ... (see more using `--help`)

During training, logs are printed in your terminal (and saved in text and CSV format):

<p align="center"><img src="README-rsrc/train-terminal-logs.png"></p>

**Note:** `U` gives the update number, `F` the total number of frames, `FPS` the number of frames per second, `D` the total duration, `rR:μσmM` the mean, std, min and max reshaped return per episode, `F:μσmM` the mean, std, min and max number of frames per episode, `H` the entropy, `V` the value, `pL` the policy loss, `vL` the value loss and `∇` the gradient norm.

During training, logs might also be plotted in Tensorboard if `--tb` is added.

<p><img src="README-rsrc/train-tensorboard.png"></p>

**Note:** `tensorboardX` package is required and can be installed with `pip3 install tensorboardX`.

### `scripts/visualize.py`

An example of use:

`python3 -m scripts.visualize --env MiniGrid-DoorKey-5x5-v0 --model DoorKey`

<p align="center"><img src="README-rsrc/visualize-doorkey.gif"></p>

In this use case, the script displays how the model in `storage/DoorKey` behaves on the MiniGrid DoorKey environment.

More generally, the script has 2 required arguments:
- `--env ENV`: name of the environment to act on.
- `--model MODEL`: name of the trained model.

and a bunch of optional arguments among which:
- `--argmax`: select the action with highest probability
- ... (see more using `--help`)

### `scripts/evaluate.py`

An example of use:

`python3 -m scripts.evaluate --env MiniGrid-DoorKey-5x5-v0 --model DoorKey`

<p align="center"><img src="README-rsrc/evaluate-terminal-logs.png"></p>

In this use case, the script prints in the terminal the performance among 100 episodes of the model in `storage/DoorKey`.

More generally, the script has 2 required arguments:
- `--env ENV`: name of the environment to act on.
- `--model MODEL`: name of the trained model.

and a bunch of optional arguments among which:
- `--episodes N`: number of episodes of evaluation. By default, N = 100.
- ... (see more using `--help`)

### `model.py`

The default model is discribed by the following schema:

<p align="center"><img src="README-rsrc/model.png"></p>

By default, the memory part (in red) and the langage part (in blue) are disabled. They can be enabled by setting to `True` the `use_memory` and `use_text` parameters of the model constructor.

This model can be easily adapted to your needs.

## `torch_rl` package

The package consists of:
- `torch_rl.A2CAlgo` and `torch_rl.PPOAlgo` classes for A2C and PPO algorithms
- `torch_rl.ACModel` and `torch_rl.RecurrentACModel` abstract classes for non-recurrent and recurrent actor-critic models
- a `torch_rl.DictList` class for making dictionnaries of lists list-indexable and hence batch-friendly

### How to use?

Here are detailed the most important parts of the package.

`torch_rl.A2CAlgo` and `torch_rl.PPOAlgo` have 2 methods:
- `__init__` that may take, among the other parameters:
    - an `acmodel` actor-critic model, i.e. an instance of a class inheriting from either `torch_rl.ACModel` or `torch_rl.RecurrentACModel`.
    - a `preprocess_obss` function that transforms a list of observations into a list-indexable object `X` (e.g. a PyTorch tensor). The default `preprocess_obss` function converts observations into a PyTorch tensor.
    - a `reshape_reward` function that takes into parameter an observation `obs`, the action `action` taken, the reward `reward` received and the terminal status `done` and returns a new reward. By default, the reward is not reshaped.
    - a `recurrence` number to specify over how many timesteps gradient is backpropagated. This number is only taken into account if a recurrent model is used and **must divide** the `num_frames_per_agent` parameter and, for PPO, the `batch_size` parameter.
- `update_parameters` that first collects experiences, then update the parameters and finally returns logs.

`torch_rl.ACModel` has 2 abstract methods:
- `__init__` that takes into parameter an `observation_space` and an `action_space`.
- `forward` that takes into parameter N preprocessed observations `obs` and returns a PyTorch distribution `dist` and a tensor of values `value`. The tensor of values **must be** of size N, not N x 1.

`torch_rl.RecurrentACModel` has 3 abstract methods:
- `__init__` that takes into parameter the same parameters than `torch_rl.ACModel`.
- `forward` that takes into parameter the same parameters than `torch_rl.ACModel` along with a tensor of N memories `memory` of size N x M where M is the size of a memory. It returns the same thing than `torch_rl.ACModel` plus a tensor of N memories `memory`.
- `memory_size` that returns the size M of a memory.

**Note:** The `preprocess_obss` function must return a list-indexable object (e.g. a PyTorch tensor). If your observations are dictionnaries, your `preprocess_obss` function may first convert a list of dictionnaries into a dictionnary of lists and then make it list-indexable using the `torch_rl.DictList` class as follow:

```python
>>> d = DictList({"a": [[1, 2], [3, 4]], "b": [[5], [6]]})
>>> d.a
[[1, 2], [3, 4]]
>>> d[0]
DictList({"a": [1, 2], "b": [5]})
```

**Note:** if you use a RNN, you will need to set `batch_first` to `True`.

### Examples

An example of use of `torch_rl.A2CAlgo` and `torch_rl.PPOAlgo` classes is given in `scripts/train.py`.

An example of use of `torch_rl.DictList` is given in the `preprocess_obss` functions of `utils/format.py`.

An example of implementation of `torch_rl.RecurrentACModel` abstract class is defined in `model.py`

Examples of `preprocess_obss` functions are given in `utils/format.py`.