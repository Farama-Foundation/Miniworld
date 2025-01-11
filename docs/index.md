---
hide-toc: true
firstpage:
lastpage:
---

```{toctree}
:hidden:
:caption: Introduction

content/design
content/troubleshooting
content/create_env
content/environments
```

```{toctree}
:hidden:
:caption: Development

release_notes
Github <https://github.com/Farama-Foundation/Miniworld>
Contribute to the Docs <https://github.com/Farama-Foundation/Miniworld/tree/master/docs/>

```

```{project-logo} _static/img/miniworld-text.png
:alt: Miniworld Logo
```

```{project-heading}
Miniworld is a minimalistic 3D interior environment simulator for reinforcement learning & robotics research
```

```{figure} _static/img/miniworld_homepage.gif
    :width: 400px
    :alt: Sequence of observations from Collect Health environment
```

MiniWorld allows environments to be easily edited like Minigrid meets DM Lab. It can simulate environments with rooms, doors, hallways, and various objects (e.g., office and home environments, mazes).

## Installation

```python
pip install miniworld
```

## Usage

The Gymnasium interface allows to initialize and interact with the Miniworld default environments as follows:

```python
import gymnasium as gym
env = gym.make("MiniWorld-OneRoom-v0")
observation, info = env.reset(seed=42)
for _ in range(1000):
   action = policy(observation)  # User-defined policy function
   observation, reward, terminated, truncated, info = env.step(action)

   if terminated or truncated:
      observation, info = env.reset()
env.close()
```

## Citation

To cite this project please use:

```bibtex
@article{MinigridMiniworld23,
  author       = {Maxime Chevalier-Boisvert and Bolun Dai and Mark Towers and Rodrigo de Lazcano and Lucas Willems and Salem Lahlou and Suman Pal and Pablo Samuel Castro and Jordan Terry},
  title        = {Minigrid \& Miniworld: Modular \& Customizable Reinforcement Learning Environments for Goal-Oriented Tasks},
  journal      = {CoRR},
  volume       = {abs/2306.13831},
  year         = {2023},
}
```
