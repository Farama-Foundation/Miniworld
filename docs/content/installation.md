# Installation

To install `miniworld`, the easiest way is to use `pip`:

```bash
pip install miniworld
```

However, if you would like to build on top of `miniworld`, you would need to install it from source. To do this, first clone the repository:

```bash
git clone https://github.com/Farama-Foundation/Miniworld.git
```

Then, install the package:

```bash
cd Miniworld
python3 -m pip install .
```

If you want to install the package in [development mode](https://setuptools.pypa.io/en/latest/userguide/development_mode.html), use the following command instead:

```bash
python3 -m pip install -e .
```

An installation in development mode (i.e., an editable install) is useful if you want to modify the source code of `miniworld` and have your changes take effect immediately.
