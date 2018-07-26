import numpy as np
from numpy.linalg import norm
from copy import deepcopy

import gym
from gym import spaces

GRAV = 9.81


# numpy's cross is really slow for some reason
def cross(a, b):
    return np.array([a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]])

# returns (normalized vector, original norm)
def normalize(x):
    n = norm(x)
    if n < 0.00001:
        return x, 0
    return x / n, n

def norm2(x):
    return np.sum(x ** 2)

# uniformly sample from the set of all 3D rotation matrices
def rand_uniform_rot3d(np_random):
    randunit = lambda: normalize(np_random.normal(size=(3,)))[0]
    up = randunit()
    fwd = randunit()
    while np.dot(fwd, up) > 0.95:
        fwd = randunit()
    left = normalize(cross(up, fwd))
    up = cross(fwd, left)
    rot = np.column_stack([fwd, left, up])
    return rot

# shorter way to construct a numpy array
def npa(*args):
    return np.array(args)

def clamp_norm(x, maxnorm):
    n = np.linalg.norm(x)
    return x if n <= maxnorm else (maxnorm / n) * x

# project a vector into the x-y plane and normalize it.
def to_xyhat(vec):
    v = deepcopy(vec)
    v[2] = 0
    v, _ = normalize(v)
    return v

# like raw motor control, but shifted such that a zero action
# corresponds to the amount of thrust needed to hover.
class ShiftedMotorControl(object):
    def __init__(self, dynamics):
        pass

    def action_space(self, dynamics):
        # make it so the zero action corresponds to hovering
        low = -1.0 * np.ones(4)
        high = (dynamics.thrust_to_weight - 1.0) * np.ones(4)
        return spaces.Box(low, high)

    # modifies the dynamics in place.
    def step(self, dynamics, action, dt):
        action = (action + 1.0) / dynamics.thrust_to_weight
        action[action < 0] = 0
        action[action > 1] = 1
        dynamics.step(action, dt)

class RawControl(object):
    def __init__(self, dynamics, zero_action_middle=True):
        self.zero_action_middle = zero_action_middle
        pass

    def action_space(self, dynamics):
        if not self.zero_action_middle:
            # Range of actions 0 .. 1
            self.low = np.zeros(4)
            self.bias = 0
            self.scale = 1.0
        else:
            # Range of actions -1 .. 1
            self.low = -np.ones(4)
            self.bias =  1.0
            self.scale = 0.5
        self.high = np.ones(4)
        return spaces.Box(self.low, self.high)

    # modifies the dynamics in place.
    def step(self, dynamics, action, goal, dt):
        action = self.scale * (action + self.bias)
        action = np.clip(action, a_min=self.low, a_max=self.high)
        dynamics.step(action, dt)


# jacobian of (acceleration magnitude, angular acceleration)
#       w.r.t (normalized motor thrusts) in range [0, 1]
def quadrotor_jacobian(dynamics):
    torque = dynamics.thrust * dynamics.prop_crossproducts.T
    torque[2,:] = dynamics.torque * dynamics.prop_ccw
    thrust = dynamics.thrust * np.ones((1,4))
    dw = (1.0 / dynamics.inertia)[:,None] * torque
    dv = thrust / dynamics.mass
    J = np.vstack([dv, dw])
    assert np.linalg.cond(J) < 25.0
    return J


# P-only linear controller on angular velocity.
# direct (ignoring motor lag) control of thrust magnitude.
class OmegaThrustControl(object):
    def __init__(self, dynamics):
        jacobian = quadrotor_jacobian(dynamics)
        self.Jinv = np.linalg.inv(jacobian)

    def action_space(self, dynamics):
        circle_per_sec = 2 * np.pi
        max_rp = 5 * circle_per_sec
        max_yaw = 1 * circle_per_sec
        min_g = -1.0
        max_g = dynamics.thrust_to_weight - 1.0
        low  = np.array([min_g, -max_rp, -max_rp, -max_yaw])
        high = np.array([max_g,  max_rp,  max_rp,  max_yaw])
        return spaces.Box(low, high)

    # modifies the dynamics in place.
    def step(self, dynamics, action, dt):
        kp = 5.0 # could be more aggressive
        omega_err = dynamics.omega - action[1:]
        dw_des = -kp * omega_err
        acc_des = GRAV * (action[0] + 1.0)
        des = np.append(acc_des, dw_des)
        thrusts = np.matmul(self.Jinv, des)
        thrusts[thrusts < 0] = 0
        thrusts[thrusts > 1] = 1
        dynamics.step(thrusts, dt)


# TODO: this has not been tested well yet.
class VelocityYawControl(object):
    def __init__(self, dynamics):
        jacobian = quadrotor_jacobian(dynamics)
        self.Jinv = np.linalg.inv(jacobian)

    def action_space(self, dynamics):
        vmax = 20.0 # meters / sec
        dymax = 4 * np.pi # radians / sec
        high = np.array([vmax, vmax, vmax, dymax])
        return spaces.Box(-high, high)

    def step(self, dynamics, action, dt):
        # needs to be much bigger than in normal controller
        # so the random initial actions in RL create some signal
        kp_v = 5.0
        kp_a, kd_a = 100.0, 50.0

        e_v = dynamics.vel - action[:3]
        acc_des = -kp_v * e_v + npa(0, 0, GRAV)

        # rotation towards the ideal thrust direction
        # see Mellinger and Kumar 2011
        R = dynamics.rot
        zb_des, _ = normalize(acc_des)
        yb_des, _ = normalize(cross(zb_des, R[:,0]))
        xb_des    = cross(yb_des, zb_des)
        R_des = np.column_stack((xb_des, yb_des, zb_des))

        def vee(R):
            return np.array([R[2,1], R[0,2], R[1,0]])
        e_R = 0.5 * vee(np.matmul(R_des.T, R) - np.matmul(R.T, R_des))
        omega_des = np.array([0, 0, action[3]])
        e_w = dynamics.omega - omega_des

        dw_des = -kp_a * e_R - kd_a * e_w
        # we want this acceleration, but we can only accelerate in one direction!
        thrust_mag = np.dot(acc_des, dynamics.rot[:,2])

        des = np.append(thrust_mag, dw_des)
        thrusts = np.matmul(self.Jinv, des)
        thrusts = np.clip(thrusts, a_min=0.0, a_max=1.0)
        dynamics.step(thrusts, dt)


# this is an "oracle" policy to drive the quadrotor towards a goal
# using the controller from Mellinger et al. 2011
class NonlinearPositionController(object):
    def __init__(self, dynamics):
        jacobian = quadrotor_jacobian(dynamics)
        self.Jinv = np.linalg.inv(jacobian)

    # modifies the dynamics in place.
    def step(self, dynamics, goal, dt, action=None):
        kp_p, kd_p = 4.5, 3.5
        kp_a, kd_a = 200.0, 50.0

        to_goal = goal - dynamics.pos
        goal_dist = norm(to_goal)
        e_p = -clamp_norm(to_goal, 4.0)
        e_v = dynamics.vel
        acc_des = -kp_p * e_p - kd_p * e_v + np.array([0, 0, GRAV])

        if goal_dist > 2.0 * dynamics.arm:
            # point towards goal
            xc_des = to_xyhat(to_goal)
        else:
            # keep current
            xc_des = to_xyhat(dynamics.rot[:,0])

        # rotation towards the ideal thrust direction
        # see Mellinger and Kumar 2011
        zb_des, _ = normalize(acc_des)
        yb_des, _ = normalize(cross(zb_des, xc_des))
        xb_des    = cross(yb_des, zb_des)
        R_des = np.column_stack((xb_des, yb_des, zb_des))
        R = dynamics.rot

        def vee(R):
            return np.array([R[2,1], R[0,2], R[1,0]])
        e_R = 0.5 * vee(np.matmul(R_des.T, R) - np.matmul(R.T, R_des))
        e_R[2] *= 0.2 # slow down yaw dynamics
        e_w = dynamics.omega

        dw_des = -kp_a * e_R - kd_a * e_w
        # we want this acceleration, but we can only accelerate in one direction!
        thrust_mag = np.dot(acc_des, R[:,2])

        des = np.append(thrust_mag, dw_des)
        thrusts = np.matmul(self.Jinv, des)
        thrusts[thrusts < 0] = 0
        thrusts[thrusts > 1] = 1
        dynamics.step(thrusts, dt)

    def action_space(self, dynamics):
        circle_per_sec = 2 * np.pi
        max_rp = 5 * circle_per_sec
        max_yaw = 1 * circle_per_sec
        min_g = -1.0
        max_g = dynamics.thrust_to_weight - 1.0
        low  = np.array([min_g, -max_rp, -max_rp, -max_yaw])
        high = np.array([max_g,  max_rp,  max_rp,  max_yaw])
        return spaces.Box(low, high)


# TODO:
# class AttitudeControl,
# refactor common parts of VelocityYaw and NonlinearPosition