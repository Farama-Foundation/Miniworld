import math
import numpy as np
from collections import namedtuple

# Simulation parameter, with domain randomization range
# The default value is used when domain randomization is disabled
DomainParam = namedtuple('Domainparam', ['default', 'min', 'max', 'type'])

class DomainParams:
    """
    Set of simulation parameters
    """

    def __init__(self):
        # Dictionary of parameters, indexed by name
        self.params = {}

    def add(self, name, default, min=None, max=None, type='float'):
        """
        Register a named parameter
        """

        assert name not in self.params

        if isinstance(default, list):
            default = np.array(default)
        if isinstance(min, list):
            min = np.array(min)
        if isinstance(max, list):
            max = np.array(max)

        if isinstance(default, np.ndarray):
            assert max.shape == default.shape
            assert min.shape == max.shape
            assert np.all(np.greater_equal(max, default))
            assert np.all(np.greater_equal(default, min))
        else:
            assert max >= default
            assert default >= min

        self.params[name] = DomainParam(default, min, max, type)

    def sample(self, rng, name):
        """
        Sample a single parameter
        Note: when rng is None, the default value is returned, which
        has the effect of turning off domain randomization
        """

        assert name in self.params
        p = self.params[name]

        if rng is None:
            return p.default

        if p.type == 'float':
            return rng.float(p.min, p.max)
        elif p.type == 'int':
            return rng.int(p.min, p.max+1)

        assert False

    def sample_many(self, rng, target_obj, param_names):
        """
        Sample a list of parameters
        """

        for name in param_names:
            setattr(target_obj, name, self.sample(rng, name))

# Default simulation parameters
DEFAULT_PARAMS = DomainParams()
DEFAULT_PARAMS.add('sky_color', [0.25, 0.82, 1], [0.1, 0.1, 0.1], [1.0, 1.0, 1.0])
DEFAULT_PARAMS.add('light_pos', [0, 2.5, 0], [-40, 2.5, -40], [40, 5, 40])
DEFAULT_PARAMS.add('light_color', [0.7, 0.7, 0.7], [0.45, 0.45, 0.45], [0.8, 0.8, 0.8])
DEFAULT_PARAMS.add('cam_pitch', 0, -5, 5)
DEFAULT_PARAMS.add('cam_fov_y', 60, 55, 65)
DEFAULT_PARAMS.add('cam_height', 1.5, 1.45, 1.55)
DEFAULT_PARAMS.add('cam_fwd_disp', 0, -0.05, 0.10)
