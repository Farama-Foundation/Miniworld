from setuptools import setup

setup(
    name='gym_miniworld',
    version='2018.8.1',
    keywords='environment, agent, rl, openaigym, openai-gym, gym, robotics, 3d',
    packages=['gym_miniworld', 'gym_miniworld.envs'],
    install_requires=[
        'gym>=0.9.0',
        'numpy>=1.10.0',
        'pyglet',
        #'pyyaml>=3.12',
        #'pybullet>=2.1.0',
    ]
)
