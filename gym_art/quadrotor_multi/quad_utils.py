import numpy as np
import numpy.random as nr
from numpy.linalg import norm
from copy import deepcopy
from numba import njit, types
from numba.core.errors import TypingError
from numba.extending import overload


# shorter way to construct a numpy array
@njit
def npa(*args):
    return np.array(args)


@njit
def numba_sum_1(val):
    return np.sum(val, axis=0)


@njit
def numba_sum_2(val):
    return np.sum(val)


@njit
def calculate_motor_tau(dt, tc, mdtu, mdtd, tcd, eps):
    mtu = 4 * dt / (mdtu + eps)
    mtd = 4 * dt / (mdtd + eps)
    mt = mtu * np.ones(4)
    mt[tc < tcd] = mtd
    mt[mt > 1.] = 1.
    return mt, mtu, mtd


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


@njit
def numba_clip(val, min , max):
    return np.clip(val, min, max)


# dict pretty printing
def print_dic(dic, indent=""):
    for key, item in dic.items():
        if isinstance(item, dict):
            print(indent, key + ":")
            print_dic(item, indent=indent+"  ")
        else:
            print(indent, key + ":", item)

# walk dictionary
def walk_dict(node, call):
    for key, item in node.items():
        if isinstance(item, dict):
            walk_dict(item, call)
        else:
            node[key] = call(key, item)

def walk_2dict(node1, node2, call):
    for key, item in node1.items():
        if isinstance(item, dict):
            walk_2dict(item, node2[key], call)
        else:
            node1[key], node2[key] = call(key, item, node2[key])

# numpy's cross is really slow for some reason
def cross(a, b):
    return np.array([a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]])

# returns (normalized vector, original norm)
def normalize(x):
    #n = norm(x)
    n = (x[0]**2 + x[1]**2 + x[2]**2)**0.5#np.sqrt(np.cumsum(np.square(x)))[2]

    if n < 0.00001:
        return x, 0
    return x / n, n

def norm2(x):
    return np.sum(x ** 2)

# uniformly sample from the set of all 3D rotation matrices
def rand_uniform_rot3d():
    randunit = lambda: normalize(np.random.normal(size=(3,)))[0]
    up = randunit()
    fwd = randunit()
    while np.dot(fwd, up) > 0.95:
        fwd = randunit()
    left, _ = normalize(cross(up, fwd))
    # import pdb; pdb.set_trace()
    up = cross(fwd, left)
    rot = np.column_stack([fwd, left, up])
    return rot

# shorter way to construct a numpy array
def npa(*args):
    return np.array(args)

def clamp_norm(x, maxnorm):
    #n = np.linalg.norm(x)
   # n = np.sqrt(np.cumsum(np.square(x)))[2]
    n = (x[0]**2 + x[1]**2 + x[2]**2)**0.5
    return x if n <= maxnorm else (maxnorm / n) * x

# project a vector into the x-y plane and normalize it.
def to_xyhat(vec):
    v = deepcopy(vec)
    v[2] = 0
    v, _ = normalize(v)
    return v

def log_error(err_str, ):
    with open("/tmp/sac/errors.txt", "a") as myfile:
        myfile.write(err_str)
        # myfile.write('###############################################')


def quat2R(qw, qx, qy, qz):
    R = \
    [[1.0 - 2*qy**2 - 2*qz**2,         2*qx*qy - 2*qz*qw,         2*qx*qz + 2*qy*qw],
     [      2*qx*qy + 2*qz*qw,   1.0 - 2*qx**2 - 2*qz**2,         2*qy*qz - 2*qx*qw],
     [      2*qx*qz - 2*qy*qw,         2*qy*qz + 2*qx*qw,   1.0 - 2*qx**2 - 2*qy**2]]
    return np.array(R)

def qwxyz2R(quat):
    return quat2R(qw=quat[0], qx=quat[1], qy=quat[2], qz=quat[3])

def quatXquat(quat, quat_theta):
    ## quat * quat_theta
    noisy_quat = np.zeros(4)
    noisy_quat[0] = quat[0] * quat_theta[0] - quat[1] * quat_theta[1] - quat[2] * quat_theta[2] - quat[3] * quat_theta[3] 
    noisy_quat[1] = quat[0] * quat_theta[1] + quat[1] * quat_theta[0] - quat[2] * quat_theta[3] + quat[3] * quat_theta[2] 
    noisy_quat[2] = quat[0] * quat_theta[2] + quat[1] * quat_theta[3] + quat[2] * quat_theta[0] - quat[3] * quat_theta[1] 
    noisy_quat[3] = quat[0] * quat_theta[3] - quat[1] * quat_theta[2] + quat[2] * quat_theta[1] + quat[3] * quat_theta[0]
    return noisy_quat

def R2quat(rot):
    # print('R2quat: ', rot, type(rot))
    R = rot.reshape([3,3])
    w = np.sqrt(1.0 + R[0,0] + R[1,1] + R[2,2]) / 2.0
    w4 = (4.0 * w)
    x = (R[2,1] - R[1,2]) / w4
    y = (R[0,2] - R[2,0]) / w4
    z = (R[1,0] - R[0,1]) / w4
    return np.array([w,x,y,z])

def rot2D(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, -s], [s, c]])

def rotZ(theta):
    r = np.eye(4)
    r[:2,:2] = rot2D(theta)
    return r

def rpy2R(r, p, y):
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(r), -np.sin(r)],
                    [0, np.sin(r), np.cos(r)]
                    ])
    R_y = np.array([[np.cos(p), 0, np.sin(p)],
                    [0, 1, 0],
                    [-np.sin(p), 0, np.cos(p)]
                    ])
    R_z = np.array([[np.cos(y), -np.sin(y), 0],
                    [np.sin(y), np.cos(y), 0],
                    [0, 0, 1]
                    ])

    R = np.dot(R_z, np.dot( R_y, R_x ))

    return R

def randyaw():
    rotz = np.random.uniform(-np.pi, np.pi)
    return rotZ(rotz)[:3,:3]

def exUxe(e,U):
    """
    Cross product approximation
    exUxe = U - (U @ e) * e, where
    Args:
        e[3,1] - norm vector (assumes the same norm vector for all vectors in the batch U)
        U[3,batch_dim] - set of vectors to perform cross product on
    Returns:
        [3,batch_dim] - batch-wise cross product approximation
    """
    return U - (U.T @ rot_z).T * np.repeat(rot_z, U.shape[1], axis=1)

def cross_vec(v1,v2):
    return np.array([[0, -v1[2], v1[1]], [v1[2], 0, -v1[0]], [-v1[1], v1[0], 0]]) @ v2

def cross_mx4(V1,V2):
    x1 = cross(V1[0,:],V2[0,:])
    x2 = cross(V1[1,:],V2[1,:])
    x3 = cross(V1[2,:],V2[2,:])
    x4 = cross(V1[3,:],V2[3,:])
    return np.array([x1,x2,x3,x4])

def cross_vec_mx4(V1,V2):
    x1 = cross(V1,V2[0,:])
    x2 = cross(V1,V2[1,:])
    x3 = cross(V1,V2[2,:])
    x4 = cross(V1,V2[3,:])
    return np.array([x1,x2,x3,x4])

def dict_update_existing(dic, dic_upd):
    for key in dic_upd.keys():
        if isinstance(dic[key], dict):
            dict_update_existing(dic[key], dic_upd[key])
        else:
            dic[key] = dic_upd[key]

class OUNoise:
    """Ornstein–Uhlenbeck process"""
    def __init__(self, action_dimension, mu=0, theta=0.15, sigma=0.3):
        """
        @param: mu: mean of noise
        @param: theta: stabilization coeff (i.e. noise return to mean)
        @param: sigma: noise scale coeff
        """
        self.action_dimension = action_dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * nr.randn(len(x))
        self.state = x + dx
        return self.state


if __name__ == "__main__":
    ## Cross product test
    import time
    rot_z = np.array([[3],[4],[5]])
    rot_z = rot_z / np.linalg.norm(rot_z)
    v_rotors = np.array([[1,2,3,4],[5,6,7,8],[9,8,7,6]])

    start_time = time.time()
    cr1_ = v_rotors - (v_rotors.T @ rot_z).T * np.repeat(rot_z,4, axis=1)
    print("cr1 time:", time.time() - start_time)

    start_time = time.time()
    cr2 = np.cross(rot_z.T, np.cross(v_rotors.T, np.repeat(rot_z,4, axis=1).T)).T
    print("cr2 time:", time.time() - start_time)
    print("cr1 == cr2:", np.sum(cr1 - cr2) < 1e-10)
