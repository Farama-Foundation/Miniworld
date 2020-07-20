import inspect
import gym

from .remotebot import *
from .hallway import *
from .oneroom import *
from .roomobjs import *
from .pickupobjs import *
from .tmaze import *
from .ymaze import *
from .maze import *
from .fourrooms import *
from .threerooms import *
from .wallgap import *
from .sidewalk import *
from .putnext import *
from .collecthealth import *
from .simtorealgoto import *
from .simtorealpush import *

# Registered environment ids
env_ids = []

def register_envs():
    module_name = __name__
    global_vars = globals()

    # Iterate through global names
    for global_name in sorted(list(global_vars.keys())):
        env_class = global_vars[global_name]

        if not inspect.isclass(env_class):
            continue

        if not issubclass(env_class, gym.core.Env):
            continue

        if env_class is MiniWorldEnv:
            continue

        # Register the environment with OpenAI Gym
        gym_id = 'MiniWorld-%s-v0' % (global_name)
        entry_point = '%s:%s' % (module_name, global_name)

        gym.envs.registration.register(
            id=gym_id,
            entry_point=entry_point,
        )

        env_ids.append(gym_id)

        #print('Registered env:', gym_id)

register_envs()
