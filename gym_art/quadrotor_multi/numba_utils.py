import numpy as np
import numpy.random as nr
from numba import njit, types, vectorize, int32, float32, double, boolean
from numba.core.errors import TypingError
from numba.extending import overload
from numba.experimental import jitclass


@overload(np.clip)
def impl_clip(a, a_min, a_max):
    # Check that `a_min` and `a_max` are scalars, and at most one of them is None.
    if not isinstance(a_min, (types.Integer, types.Float, types.NoneType)):
        raise TypingError("a_min must be a_min scalar int/float")
    if not isinstance(a_max, (types.Integer, types.Float, types.NoneType)):
        raise TypingError("a_max must be a_min scalar int/float")
    if isinstance(a_min, types.NoneType) and isinstance(a_max, types.NoneType):
        raise TypingError("a_min and a_max can't both be None")

    if isinstance(a, (types.Integer, types.Float)):
        # `a` is a scalar with a valid type
        if isinstance(a_min, types.NoneType):
            # `a_min` is None
            def impl(a, a_min, a_max):
                return min(a, a_max)
        elif isinstance(a_max, types.NoneType):
            # `a_max` is None
            def impl(a, a_min, a_max):
                return max(a, a_min)
        else:
            # neither `a_min` or `a_max` are None
            def impl(a, a_min, a_max):
                return min(max(a, a_min), a_max)
    elif (
            isinstance(a, types.Array) and
            a.ndim == 1 and
            isinstance(a.dtype, (types.Integer, types.Float))
    ):
        # `a` is a 1D array of the proper type
        def impl(a, a_min, a_max):
            # Allocate an output array using standard numpy functions
            out = np.empty_like(a)
            # Iterate over `a`, calling `np.clip` on every element
            for i in range(a.size):
                # This will dispatch to the proper scalar implementation (as
                # defined above) at *compile time*. There should have no
                # overhead at runtime.
                out[i] = np.clip(a[i], a_min, a_max)
            return out
    else:
        raise TypingError("`a` must be an int/float or a 1D array of ints/floats")

    # The call to `np.clip` has arguments with valid types, return our
    # numba-compatible implementation
    return impl


@vectorize(nopython=True)
def angvel2thrust_numba(w, linearity=0.424):
    return (1 - linearity) * w ** 2 + linearity * w


@njit
def numba_cross(a, b):
    return np.array([a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]])


spec = [
    ('action_dimension', int32),
    ('mu', float32),
    ('theta', float32),
    ('sigma', float32),
    ('state', double[:]),
    ('use_seed', boolean)
]


@jitclass(spec)
class OUNoiseNumba:
    """Ornstein–Uhlenbeck process"""

    def __init__(self, action_dimension, mu=0, theta=0.15, sigma=0.3, use_seed=False):
        """
        @param: mu: mean of noise
        @param: theta: stabilization coeff (i.e. noise return to mean)
        @param: sigma: noise scale coeff
        @param: use_seed: set the random number generator to some specific seed for test
        """
        self.action_dimension = action_dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

        if use_seed:
            nr.seed(2)

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * nr.randn(len(x))
        self.state = x + dx
        return self.state

