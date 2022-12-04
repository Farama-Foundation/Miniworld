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
```

```{toctree}
:hidden:
:glob:
:caption: Environments

environments/*
```


```{toctree}
:hidden:
:caption: Development

Github <https://github.com/Farama-Foundation/Miniworld>
Contribute to the Docs <https://github.com/Farama-Foundation/Miniworld/tree/master/docs/>

```

# Miniworld


```{figure} _static/environments/collecthealth.jpg
    :width: 400px
    :alt: Observation from Collect Health environment
```

**The Gymnasium interface allows to initialize and interact with the Miniworld default environments as follows:**

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
