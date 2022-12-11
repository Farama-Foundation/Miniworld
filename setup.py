"""Setups up the Miniworld module."""

from setuptools import setup


def get_description():
    """Gets the description from the readme."""
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
    return header_count, long_description


def get_version():
    """Gets the miniworld version."""
    path = "miniworld/__init__.py"
    with open(path) as file:
        lines = file.readlines()

    for line in lines:
        if line.startswith("__version__"):
            return line.strip().split()[-1].strip().strip('"')
    raise RuntimeError("bad version data in __init__.py")


version = get_version()
header_count, long_description = get_description()

setup(
    name="miniworld",
    version=version,
    author="Farama Foundation",
    author_email="contact@farama.org",
    description="Minimalistic 3D interior environment simulator for reinforcement learning & robotics research.",
    url="https://github.com/Farama-Foundation/Miniworld",
    license="Apache",
    license_files=("LICENSE",),
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords=["Environment", "Agent", "RL", "Gym", "Robotics", "3D"],
    python_requires=">=3.7, <3.11",
    packages=["miniworld", "miniworld.envs"],
    package_data={
        "miniworld": [
            "textures/*.png",
            "textures/chars/*.png",
            "textures/portraits/*.png",
            "meshes/*.mtl",
            "meshes/*.obj",
        ]
    },
    extras_require={"testing": ["pytest==7.0.1", "torch"]},
    install_requires=[
        "numpy>=1.18.0",
        "pyglet==1.5.27",
        "gymnasium>=0.26.2",
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
