from setuptools import setup

with open("README.md") as fh:
    long_description = ""
    header_count = 0
    for line in fh:
        if line.startswith("##"):
            header_count += 1
        if header_count < 2:
            long_description += line
        else:
            break

setup(
    name="gym_miniworld",
    author="Farama Foundation",
    author_email="jkterry@farama.org",
    version="2020.1.9",
    keywords="environment, agent, rl, gym, robotics, 3d",
    packages=["gym_miniworld", "gym_miniworld.envs"],
    package_data={
        "gym_miniworld": [
            "textures/*.png",
            "textures/chars/*.png",
            "textures/portraits/*.png",
            "meshes/*.mtl",
            "meshes/*.obj",
        ]
    },
    long_description=long_description,
    python_requires=">=3.7, <3.11",
    extras_require={"testing": ["pytest==7.0.1", "torch"]},
    long_description_content_type="text/markdown",
    install_requires=[
        "gym>=0.24.0",
        "numpy>=1.18.0",
        "pyglet>=1.5.11",
    ],
    # Include textures and meshes in the package
    include_package_data=True,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
