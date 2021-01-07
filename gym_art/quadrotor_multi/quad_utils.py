import numpy as np
import numpy.random as nr
from numba import njit
from numpy.linalg import norm
from copy import deepcopy
from numpy import cos, sin
from scipy import spatial
from copy import deepcopy

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


quat2R_numba = njit()(quat2R)


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


quatXquat_numba = njit()(quatXquat)


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


def spherical_coordinate(x, y):
    return [cos(x) * cos(y), sin(x) * cos(y), sin(y)]


def points_in_sphere(n, x):
    pts = []
    start = (-1. + 1. / (n - 1.))
    increment = (2. - 2. / (n - 1.)) / (n - 1.)
    pi = np.pi
    for j in range(n):
        s = start + j * increment
        pts.append(spherical_coordinate(
            s * x, pi / 2. * np.sign(s) * (1. - np.sqrt(1. - abs(s)))
        ))
    return pts


def generate_points(n=3):
    if n < 3:
        print("The number of goals can not smaller than 3, The system has cast it to 3")
        n = 3
    return points_in_sphere(n, 0.1 + 1.2 * n)


def calculate_collision_matrix(positions, arm):
    dist = spatial.distance_matrix(x=positions, y=positions)
    collision_matrix = (dist < 2 * arm).astype(np.float32)
    np.fill_diagonal(collision_matrix, 0.0)

    # get upper triangular matrix and check if they have collisions and append to all collisions
    upt = np.triu(collision_matrix)
    up_w1 = np.where(upt >= 1)
    all_collisions = []
    for i, val in enumerate(up_w1[0]):
        all_collisions.append((up_w1[0][i], up_w1[1][i]))

    return collision_matrix, all_collisions


def hyperbolic_proximity_penalty(dist_matrix, dt, coeff=0.0):
    '''
    summed coeff/(x^2 + eps) clipped distance penalty between drones
    :param dist_matrix: distance between drones
    :param dt: single time step
    :param coeff: reward scaling hyperparam
    :return: spacing penalty b/w droens
    '''
    costs = -coeff / (np.power(dist_matrix + 1e-7, 2))
    np.fill_diagonal(costs, 0.0)
    costs = np.clip(costs, -10, 0)
    spacing_reward = np.array([np.sum(row) for row in costs])
    spacing_reward = spacing_reward * dt
    return spacing_reward


def compute_col_norm_and_new_velocities(dyn1, dyn2):
    # Ge the collision normal, i.e difference in position
    collision_norm = dyn1.pos - dyn2.pos
    coll_norm_mag = np.linalg.norm(collision_norm)
    collision_norm = collision_norm / (coll_norm_mag + 0.00001 if coll_norm_mag == 0.0 else coll_norm_mag)

    # Get the components of the velocity vectors which are parallel to the collision.
    # The perpendicular component remains the same.
    v1new = np.dot(dyn1.vel, collision_norm)
    v2new = np.dot(dyn2.vel, collision_norm)

    return v1new, v2new, collision_norm


# This function is to change the velocities after a collision happens between two bodies
def perform_collision_between_drones(dyn1, dyn2):
    v1new, v2new, collision_norm = compute_col_norm_and_new_velocities(dyn1, dyn2)

    # Solve for the new velocities using the elastic collision equations. It's really simple when the
    dyn1.vel += (v2new - v1new) * collision_norm
    dyn2.vel += (v1new - v2new) * collision_norm

    # Now adding two different random components,
    # One that preserves momentum in opposite directions
    # Second that does not preserve momentum
    cons_rand_val = np.random.normal(0, 0.8, 3)
    dyn1.vel += cons_rand_val + np.random.normal(0, 0.15, 3)
    dyn2.vel += -cons_rand_val + np.random.normal(0, 0.15, 3)

    # Random forces for omega
    omega_max = 7 * np.pi  # this will amount to max 3.5 revolutions per second
    eps = 1e-5
    new_omega = np.random.uniform(low=-1, high=1, size=(3,))  # random direction in 3D space
    while all(np.abs(new_omega) < eps):
        new_omega = np.random.uniform(low=-1, high=1, size=(3,))  # just to make sure we don't get a 0-vector

    new_omega /= np.linalg.norm(new_omega) + eps  # normalize

    new_omega_magn = np.random.uniform(low=omega_max / 2, high=omega_max)  # random magnitude of the force
    new_omega *= new_omega_magn

    # add the disturbance to drone's angular velocities while preserving angular momentum
    dyn1.omega += new_omega
    dyn2.omega -= new_omega


def perform_collision_with_obstacle(obs, drone_dyn):
    v1new, v2new, collision_norm = compute_col_norm_and_new_velocities(obs, drone_dyn)
    drone_dyn.vel += (v1new - v2new) * collision_norm

    # Now adding random force components
    drone_dyn.vel += np.random.normal(0, 0.8, 3)
    drone_dyn.omega += np.random.normal(0, 0.8, 3)


class OUNoise:
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


if __name__ == "__main__":
    ## Cross product test
    import time
    rot_z = np.array([[3],[4],[5]])
    rot_z = rot_z / np.linalg.norm(rot_z)
    v_rotors = np.array([[1,2,3,4],[5,6,7,8],[9,8,7,6]])

    start_time = time.time()
    cr1 = v_rotors - (v_rotors.T @ rot_z).T * np.repeat(rot_z,4, axis=1)
    print("cr1 time:", time.time() - start_time)

    start_time = time.time()
    cr2 = np.cross(rot_z.T, np.cross(v_rotors.T, np.repeat(rot_z,4, axis=1).T)).T
    print("cr2 time:", time.time() - start_time)
    print("cr1 == cr2:", np.sum(cr1 - cr2) < 1e-10)
