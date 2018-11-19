import math
import numpy as np
from collections import namedtuple

DomainParam = namedtuple('Domainparam', ['min', 'max', 'type'])

class DomainParams:
    """
    Domain randomization parameters
    """

    def __init__(self):
        # Dictionary of parameters, indexed by name
        self.params = {}

    def add(self, name, min, max=None, type='float'):
        """
        Register a named parameter
        """

        assert name not in self.params

        if isinstance(min, list):
            min = np.array(min)
        if isinstance(max, list):
            max = np.array(max)

        if max is None:
            max = min

        if isinstance(min, np.ndarray):
            assert min.shape == max.shape
        else:
            assert max >= min

        self.params[name] = DomainParam(min, max, type)

    def sample(self, rng, name):
        """
        Sample a single parameter
        """

        assert name in self.params
        p = self.params[name]

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

# Default simulation parameters with domain randomization disabled
DEFAULT_PARAMS = DomainParams()
DEFAULT_PARAMS.add('tex_rand', 0, 0, 'int')
DEFAULT_PARAMS.add('sky_color', [0.25, 0.82, 1])
DEFAULT_PARAMS.add('light_pos', [0, 2.5, 0])
DEFAULT_PARAMS.add('light_color', [0.7, 0.7, 0.7])
DEFAULT_PARAMS.add('cam_pitch', 0)
DEFAULT_PARAMS.add('cam_fov_y', 60)
DEFAULT_PARAMS.add('cam_height', 1.5)

# Default simulation arameters with domain randomization enabled
DEFAULT_PARAMS_RAND = DomainParams()
DEFAULT_PARAMS_RAND.add('tex_rand', 1, 1, 'int')
DEFAULT_PARAMS_RAND.add('sky_color', [0.1, 0.1, 0.1], [0.9, 0.9, 0.9])
DEFAULT_PARAMS_RAND.add('light_pos', [-40, 2.5, -40], [40, 5, 40])
DEFAULT_PARAMS_RAND.add('light_color', [0.45, 0.45, 0.45], [0.8, 0.8, 0.8])
DEFAULT_PARAMS_RAND.add('cam_pitch', -15, 0)
DEFAULT_PARAMS_RAND.add('cam_fov_y', 45, 65)
DEFAULT_PARAMS_RAND.add('cam_height', 1.45, 1.55)
