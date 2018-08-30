import gym

from .hallway import *

def register_envs():
    module_name = __name__
    global_vars = globals()

    # Iterate through global names
    for global_name in sorted(list(global_vars.keys())):
        if not global_name.endswith('Env'):
            continue

        env_name = global_name.split('Env')[0]
        env_class = global_vars[global_name]

        # Register the environment with OpenAI Gym
        gym_id = 'MiniWorld-%s-v0' % (env_name)
        entry_point = '%s:%s' % (module_name, global_name)

        print(entry_point)

        gym.envs.registration.register(
            id=gym_id,
            entry_point=entry_point,
        )

        print('Registered env:', gym_id)

register_envs()
