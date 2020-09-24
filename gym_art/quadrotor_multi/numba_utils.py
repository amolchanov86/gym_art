import numpy as np
import numpy.random as nr
from numba import njit, types, vectorize, int32, float32, float64, double, boolean
from numba.core.errors import TypingError
from numba.extending import overload
from numba.experimental import jitclass
import math


@njit
def set_seed():
    nr.seed(2)


@njit
def add_noise_to_measurement(measurement, val_1, val_2):
    nr.seed(2)
    return measurement + nr.normal(val_1[0], val_1[1], size=val_1[2]) + nr.uniform(val_2[0], val_2[1], size=val_2[2])


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
def numba_clip(val, mn, mx):
    return np.clip(val, mn, mx)


# @njit
# def quat_from_small_angle_numba(theta):
#     assert theta.shape == (3,)
#
#     q_squared = np.linalg.norm(theta) ** 2 / 4.0
#     if q_squared < 1:
#         q_theta = np.array([(1 - q_squared) ** 0.5, theta[0] * 0.5, theta[1] * 0.5, theta[2] * 0.5])
#     else:
#         w = 1.0 / (1 + q_squared) ** 0.5
#         f = 0.5 * w
#         q_theta = np.array([w, theta[0] * f, theta[1] * f, theta[2] * f])
#
#     q_theta = q_theta / np.linalg.norm(q_theta)
#     return q_theta


# @njit
# def rot2quat_numba(rot):
#     assert rot.shape == (3, 3)
#
#     trace = np.trace(rot)
#     if trace > 0:
#         S = (trace + 1.0) ** 0.5 * 2
#         qw = 0.25 * S
#         qx = (rot[2][1] - rot[1][2]) / S
#         qy = (rot[0][2] - rot[2][0]) / S
#         qz = (rot[1][0] - rot[0][1]) / S
#     elif rot[0][0] > rot[1][1] and rot[0][0] > rot[2][2]:
#         S = (1.0 + rot[0][0] - rot[1][1] - rot[2][2]) ** 0.5 * 2
#         qw = (rot[2][1] - rot[1][2]) / S
#         qx = 0.25 * S
#         qy = (rot[0][1] + rot[1][0]) / S
#         qz = (rot[0][2] + rot[2][0]) / S
#     elif rot[1][1] > rot[2][2]:
#         S = (1.0 + rot[1][1] - rot[0][0] - rot[2][2]) ** 0.5 * 2
#         qw = (rot[0][2] - rot[2][0]) / S
#         qx = (rot[0][1] + rot[1][0]) / S
#         qy = 0.25 * S
#         qz = (rot[1][2] + rot[2][1]) / S
#     else:
#         S = (1.0 + rot[2][2] - rot[0][0] - rot[1][1]) ** 0.5 * 2
#         qw = (rot[1][0] - rot[0][1]) / S
#         qx = (rot[0][2] + rot[2][0]) / S
#         qy = (rot[1][2] + rot[2][1]) / S
#         qz = 0.25 * S
#
#     return np.array([qw, qx, qy, qz])


# @njit
# def quatXquat_numba(quat, quat_theta):
#     ## quat * quat_theta
#     noisy_quat = np.zeros(4)
#     noisy_quat[0] = quat[0] * quat_theta[0] - quat[1] * quat_theta[1] - quat[2] * quat_theta[2] - quat[3] * quat_theta[
#         3]
#     noisy_quat[1] = quat[0] * quat_theta[1] + quat[1] * quat_theta[0] - quat[2] * quat_theta[3] + quat[3] * quat_theta[
#         2]
#     noisy_quat[2] = quat[0] * quat_theta[2] + quat[1] * quat_theta[3] + quat[2] * quat_theta[0] - quat[3] * quat_theta[
#         1]
#     noisy_quat[3] = quat[0] * quat_theta[3] - quat[1] * quat_theta[2] + quat[2] * quat_theta[1] + quat[3] * quat_theta[
#         0]
#     return noisy_quat

@njit
def quatXquat_numba_new(quat, quat_theta):
    ## quat * quat_theta
    noisy_quat = np.zeros(4, dtype=quat.dtype)
    noisy_quat[0] = quat[0] * quat_theta[0] - quat[1] * quat_theta[1] - quat[2] * quat_theta[2] - quat[3] * quat_theta[
        3]
    noisy_quat[1] = quat[0] * quat_theta[1] + quat[1] * quat_theta[0] - quat[2] * quat_theta[3] + quat[3] * quat_theta[
        2]
    noisy_quat[2] = quat[0] * quat_theta[2] + quat[1] * quat_theta[3] + quat[2] * quat_theta[0] - quat[3] * quat_theta[
        1]
    noisy_quat[3] = quat[0] * quat_theta[3] - quat[1] * quat_theta[2] + quat[2] * quat_theta[1] + quat[3] * quat_theta[
        0]
    return noisy_quat


# @njit
# def quat2R_numba(qw, qx, qy, qz):
#     R = \
#         [[1.0 - 2 * qy ** 2 - 2 * qz ** 2, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
#          [2 * qx * qy + 2 * qz * qw, 1.0 - 2 * qx ** 2 - 2 * qz ** 2, 2 * qy * qz - 2 * qx * qw],
#          [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1.0 - 2 * qx ** 2 - 2 * qy ** 2]]
#     return np.array(R)


@njit
def prod_with_norm(x, norm_vals):
    nr.seed(2)
    return x * nr.normal(loc=norm_vals[0], scale=norm_vals[1], size=norm_vals[2])


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

####
# Fomr here, is just for testing purposes, will be removed in future commint
####

sensor_noise_spec = [
    ('pos_norm_std', float32),
    ('pos_unif_range', float32),
    ('vel_norm_std', float32),
    ('vel_unif_range', float32),

    ('quat_norm_std', float32),
    ('quat_unif_range', float32),

    ('gyro_noise_density', float32),
    ('gyro_random_walk', float32),
    ('gyro_bias_correlation_time', float32),
    ('gyro_norm_std', float32),

    ('gyro_bias', float64[:]),

    ('acc_static_noise_std', float32),
    ('acc_dynamic_noise_ratio', float32),

    ('bypass', boolean),
]


@jitclass(sensor_noise_spec)
class SensorNoiseNumba:
    def __init__(self, pos_norm_std=0.005, pos_unif_range=0.,
                        vel_norm_std=0.01, vel_unif_range=0.,
                        quat_norm_std=0., quat_unif_range=0., gyro_norm_std=0.,
                        gyro_noise_density=0.000175, gyro_random_walk=0.0105,
                        gyro_bias_correlation_time=1000., bypass=False,
                        acc_static_noise_std=0.002, acc_dynamic_noise_ratio=0.005):
        """
        Args:
            pos_norm_std (float): std of pos gaus noise component
            pos_unif_range (float): range of pos unif noise component
            vel_norm_std (float): std of linear vel gaus noise component
            vel_unif_range (float): range of linear vel unif noise component
            quat_norm_std (float): std of rotational quaternion noisy angle gaus component
            quat_unif_range (float): range of rotational quaternion noisy angle gaus component
            gyro_gyro_noise_density: gyroscope noise, MPU-9250 spec
            gyro_random_walk: gyroscope noise, MPU-9250 spec
            gyro_bias_correlation_time: gyroscope noise, MPU-9250 spec
            # gyro_gyro_turn_on_bias_sigma: gyroscope noise, MPU-9250 spec (val 0.09)
            bypass: no noise
        """

        self.pos_norm_std = pos_norm_std
        self.pos_unif_range = pos_unif_range

        self.vel_norm_std = vel_norm_std
        self.vel_unif_range = vel_unif_range

        self.quat_norm_std = quat_norm_std
        self.quat_unif_range = quat_unif_range

        self.gyro_noise_density = gyro_noise_density
        self.gyro_random_walk = gyro_random_walk
        self.gyro_bias_correlation_time = gyro_bias_correlation_time
        self.gyro_norm_std = gyro_norm_std
        # self.gyro_turn_on_bias_sigma = gyro_turn_on_bias_sigma
        self.gyro_bias = np.zeros(3)

        self.acc_static_noise_std = acc_static_noise_std
        self.acc_dynamic_noise_ratio = acc_dynamic_noise_ratio

        self.bypass = bypass

    def add_noise(self, pos, vel, rot, omega, acc, dt):
        # if self.bypass:
        #     return pos, vel, rot, omega, acc
        # """
        # Args:
        #     pos: ground truth of the position in world frame
        #     vel: ground truth if the linear velocity in world frame
        #     rot: ground truth of the orientation in rotational matrix / quaterions / euler angles
        #     omega: ground truth of the angular velocity in body frame
        #     dt: integration step
        # """
        assert pos.shape == (3,)
        assert vel.shape == (3,)
        assert omega.shape == (3,)

        # add noise to position measurement
        noisy_pos = pos + \
                    nr.normal(loc=0., scale=self.pos_norm_std, size=3) + \
                    nr.uniform(-self.pos_unif_range, self.pos_unif_range, 3)

        # add noise to linear velocity
        noisy_vel = vel + \
                    nr.normal(loc=0., scale=self.vel_norm_std, size=3) + \
                    nr.uniform(-self.vel_unif_range, self.vel_unif_range, 3)

        ## Noise in omega
        if self.gyro_norm_std != 0.:
            noisy_omega = self.add_noise_to_omega(omega, dt)
        else:
            noisy_omega = omega + \
                          nr.normal(loc=0., scale=self.gyro_noise_density, size=3)

        ## Noise in rotation
        theta = nr.normal(0, self.quat_norm_std, size=3) + \
                nr.uniform(-self.quat_unif_range, self.quat_unif_range, size=3)

        if rot.shape == (3,):
            ## Euler angles (xyz: roll=[-pi, pi], pitch=[-pi/2, pi/2], yaw = [-pi, pi])
            noisy_rot = rot + theta
            # noisy_rot = np.clip(rot + theta,
            #                     a_min=[-np.pi, -np.pi / 2, -np.pi],
            #                     a_max=[np.pi, np.pi / 2, np.pi])
            pass
        elif rot.shape == (3, 3):
            ## Rotation matrix
            quat_theta = quat_from_small_angle_numba(theta)
            quat = rot2quat_numba(rot)
            noisy_quat = quatXquat_numba_new(quat, quat_theta)
            noisy_rot = quat2R_numba(noisy_quat[0], noisy_quat[1], noisy_quat[2], noisy_quat[3])
        elif rot.shape == (4,):
            # noisy_rot = [0]
            pass
            ## Quaternion
            # quat_theta = quat_from_small_angle_numba(theta)
            # noisy_rot = quatXquat_numba_new(rot, quat_theta)
        else:
            # noisy_rot = [0]
            # raise ValueError("ERROR: SensNoise: Unknown rotation type: " + str(rot))
            pass

        ## Accelerometer noise
        noisy_acc = acc + nr.normal(loc=0., scale=self.acc_static_noise_std, size=3) + \
                    (acc * nr.normal(loc=0., scale=self.acc_dynamic_noise_ratio, size=3))
        # print("Vals")
        # print(noisy_pos, noisy_vel, noisy_rot, noisy_omega, noisy_acc)
        # print("\n")
        return noisy_pos, noisy_vel, noisy_rot, noisy_omega, noisy_acc

    def add_noise_to_omega(self, omega, dt):
        assert omega.shape == (3,)

        sigma_g_d = self.gyro_noise_density / (dt**0.5)
        sigma_b_g_d = (-(sigma_g_d**2) * (self.gyro_bias_correlation_time / 2) * (math.exp(-2*dt/self.gyro_bias_correlation_time) - 1))**0.5
        pi_g_d = math.exp(-dt / self.gyro_bias_correlation_time)

        self.gyro_bias = pi_g_d * self.gyro_bias + sigma_b_g_d * nr.normal(0, 1, 3)
        return omega + self.gyro_bias + self.gyro_random_walk * nr.normal(0, 1, 3)

# @overload(np.clip)
# def np_clip(a, a_min, a_max, out=None):
#     def np_clip_impl(a, a_min, a_max, out=None):
#         if out is None:
#             out = np.empty_like(a)
#         for i in range(len(a)):
#             if a[i] < a_min:
#                 out[i] = a_min
#             elif a[i] > a_max:
#                 out[i] = a_max
#             else:
#                 out[i] = a[i]
#         return out
#     return np_clip_impl


# def step1_numba_with_c_rot_drag(self, thrust_cmds, dt):
#     thrust_cmds = numba_clip(thrust_cmds, 0., 1.)
#
#     # Filtering the thruster and adding noise
#     # I use the multiplier 4, since 4*T ~ time for a step response to finish, where
#     # T is a time constant of the first-order filter
#     motor_tau, self.motor_tau_up, self.motor_tau_down = calculate_motor_tau(dt, thrust_cmds,
#                                                                             self.motor_damp_time_up,
#                                                                             self.motor_damp_time_down,
#                                                                             self.thrust_cmds_damp, EPS)
#
#     # Since NN commands thrusts we need to convert to rot vel and back
#     # WARNING: Unfortunately if the linearity != 1 then filtering using square root is not quite correct
#     # since it likely means that you are using rotational velocities as an input instead of the thrust and hence
#     # you are filtering square roots of angular velocities
#     thrust_rot = thrust_cmds ** 0.5
#     self.thrust_rot_damp = motor_tau * (thrust_rot - self.thrust_rot_damp) + self.thrust_rot_damp
#     self.thrust_cmds_damp = self.thrust_rot_damp ** 2
#
#     # Adding noise
#     thrust_noise = thrust_cmds * self.thrust_noise.noise()
#     self.thrust_cmds_damp = numba_clip(self.thrust_cmds_damp + thrust_noise, 0.0, 1.0)
#
#     thrusts = self.thrust_max * angvel2thrust_numba(self.thrust_cmds_damp, self.motor_linearity)
#     # Prop crossproduct give torque directions
#     self.torques = self.prop_crossproducts * thrusts[:, None]  # (4,3)=(props, xyz)
#
#     # additional torques along z-axis caused by propeller rotations
#     self.torques[:, 2] += self.torque_max * self.prop_ccw * self.thrust_cmds_damp
#
#     # net torque: sum over propellers
#     thrust_torque = numba_sum_1(self.torques)
#
#     ###################################
#     ## Rotor drag and Rolling forces and moments
#     ## See Ref[1] Sec:2.1 for detailes
#
#     # self.C_rot_drag = 0.0028
#     # self.C_rot_roll = 0.003 # 0.0003
#     if self.C_rot_drag != 0 or self.C_rot_roll != 0:
#         # self.vel = np.zeros_like(self.vel)
#         # v_rotors[3,4]  = (rot[3,3] @ vel[3,])[3,] + (omega[3,] x prop_pos[4,3])[4,3]
#         # v_rotors = self.rot.T @ self.vel + np.cross(self.omega, self.model.prop_pos)
#         vel_body = self.rot.T @ self.vel
#         v_rotors = vel_body + cross_vec_mx4(self.omega, self.model.prop_pos)
#         # assert v_rotors.shape == (4,3)
#         v_rotors[:, 2] = 0.  # Projection to the rotor plane
#
#         # Drag/Roll of rotors (both in body frame)
#         rotor_drag_fi = - self.C_rot_drag * np.sqrt(self.thrust_cmds_damp)[:, None] * v_rotors  # [4,3]
#         rotor_drag_force = numba_sum_1(rotor_drag_fi)
#         # rotor_drag_ti = np.cross(rotor_drag_fi, self.model.prop_pos)#[4,3] x [4,3]
#         rotor_drag_ti = cross_mx4(rotor_drag_fi, self.model.prop_pos)  # [4,3] x [4,3]
#         rotor_drag_torque = numba_sum_1(rotor_drag_ti)
#
#         rotor_roll_torque = - self.C_rot_roll * self.prop_ccw[:, None] * np.sqrt(self.thrust_cmds_damp)[:,
#                                                                          None] * v_rotors  # [4,3]
#         rotor_roll_torque = numba_sum_1(rotor_roll_torque)
#         rotor_visc_torque = rotor_drag_torque + rotor_roll_torque
#
#         ## Constraints (prevent numerical instabilities)
#         vel_norm = np.linalg.norm(vel_body)
#         rdf_norm = np.linalg.norm(rotor_drag_force)
#         rdf_norm_clip = numba_clip(rdf_norm, 0., vel_norm * self.mass / (2 * dt))
#         if rdf_norm > EPS:
#             rotor_drag_force = (rotor_drag_force / rdf_norm) * rdf_norm_clip
#
#         # omega_norm = np.linalg.norm(self.omega)
#         rvt_norm = np.linalg.norm(rotor_visc_torque)
#         rvt_norm_clipped = numba_clip(rvt_norm, 0.,
#                                       np.linalg.norm(self.omega * self.inertia) / (2 * dt))
#         if rvt_norm > EPS:
#             rotor_visc_torque = (rotor_visc_torque / rvt_norm) * rvt_norm_clipped
#     else:
#         rotor_visc_torque = rotor_drag_torque = rotor_drag_force = rotor_roll_torque = np.zeros(3)
#
#     ###################################
#     ## (Square) Damping using torques (in case we would like to add damping using torques)
#     # damping_torque = - 0.3 * self.omega * np.fabs(self.omega)
#     self.torque = thrust_torque + rotor_visc_torque
#     thrust = npa_numba(0, 0, numba_sum_2(thrusts))
#
#     #########################################################
#     ## ROTATIONAL DYNAMICS
#
#     ###################################
#     ## Integrating rotations (based on current values)
#     # omega_vec = np.matmul(self.rot, self.omega)  # Change from body2world frame
#     self.rot = integrate_rot(self.rot, np.float64(self.omega), self.eye, dt)
#
#     # omega_vec = np.matmul(self.rot, self.omega)
#     # self.rot = integrate_rot_1(self.rot, self.eye, omega_vec, dt)
#     # wx, wy, wz = omega_vec
#     # omega_norm = np.linalg.norm(omega_vec)
#     # if omega_norm != 0:
#     #     # See [7]
#     #     K = np.array([[0, -wz, wy], [wz, 0, -wx], [-wy, wx, 0]]) / omega_norm
#     #     rot_angle = omega_norm * dt
#     #     dRdt = self.eye + np.sin(rot_angle) * K + (1. - np.cos(rot_angle)) * (K @ K)
#     #     self.rot = dRdt @ self.rot
#
#     ## SVD is not strictly required anymore. Performing it rarely, just in case
#     self.since_last_svd += dt
#     if self.since_last_svd > self.since_last_svd_limit:
#         ## Perform SVD orthogonolization
#         u, s, v = np.linalg.svd(self.rot)
#         self.rot = np.matmul(u, v)
#         self.since_last_svd = 0
#
#     ###################################
#     ## COMPUTING OMEGA UPDATE
#
#     ## Damping using velocities (I find it more stable numerically)
#     ## Linear damping
#
#     # This is only for linear damping of angular velocity.
#     # omega_damp = 0.999
#     # self.omega = omega_damp * self.omega + dt * omega_dot
#
#     self.omega_dot = ((1.0 / self.inertia) *
#                       (cross(-self.omega, self.inertia * self.omega) + self.torque))
#
#     ## Quadratic damping
#     # 0.03 corresponds to roughly 1 revolution per sec
#     omega_damp_quadratic = numba_clip(self.damp_omega_quadratic * self.omega ** 2, 0.0, 1.0)
#
#     self.omega = self.omega + (1.0 - omega_damp_quadratic) * dt * self.omega_dot
#     self.omega = numba_clip(self.omega, -self.omega_max, self.omega_max)
#
#     ## When use square damping on torques - use simple integration
#     ## since damping is accounted as part of the net torque
#     # self.omega += dt * omega_dot
#
#     #########################################################
#     # TRANSLATIONAL DYNAMICS
#
#     ## Room constraints
#     # mask = np.logical_or(self.pos <= self.room_box[0], self.pos >= self.room_box[1])
#
#     ## Computing position
#     self.pos = self.pos + dt * self.vel
#
#     # Clipping if met the obstacle and nullify velocities (not sure what to do about accelerations)
#     self.pos_before_clip = self.pos.copy()
#     self.pos = np.clip(self.pos, a_min=self.room_box[0], a_max=self.room_box[1])
#     self.vel[np.equal(self.pos, self.pos_before_clip)] = 0.
#
#     ## Computing accelerations
#     acc = [0, 0, -GRAV] + (1.0 / self.mass) * np.matmul(self.rot, (thrust + rotor_drag_force))
#     # acc[mask] = 0. #If we leave the room - stop accelerating
#     self.acc = acc
#
#     ## Computing velocities
#     self.vel = (1.0 - self.vel_damp) * self.vel + dt * acc
#     # self.vel[mask] = 0. #If we leave the room - stop flying
#
#     ## Accelerometer measures so called "proper acceleration"
#     # that includes gravity with the opposite sign
#     self.accelerometer = np.matmul(self.rot.T, acc + [0, 0, self.gravity])
