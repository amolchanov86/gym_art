#!/usr/bin/env python
import numpy as np
from numpy.random import normal
from numpy.random import uniform
import matplotlib.pyplot as plt
from math import exp
from gym_art.quadrotor_single.quad_utils import quat2R, quatXquat
from numba import njit


def quat_from_small_angle(theta):
    assert theta.shape == (3,)

    q_squared = np.linalg.norm(theta)**2 / 4.0
    if q_squared < 1:
        q_theta = np.array([(1 - q_squared)**0.5, theta[0] * 0.5, theta[1] * 0.5, theta[2] * 0.5])
    else:
        w = 1.0 / (1 + q_squared)**0.5
        f = 0.5 * w
        q_theta = np.array([w, theta[0] * f, theta[1] * f, theta[2] * f])

    q_theta = q_theta / np.linalg.norm(q_theta)
    return q_theta

'''
http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
'''
def rot2quat(rot):
	assert rot.shape == (3, 3)

	trace = np.trace(rot)
	if trace > 0:
		S = (trace + 1.0)**0.5 * 2
		qw = 0.25 * S
		qx = (rot[2][1] - rot[1][2]) / S
		qy = (rot[0][2] - rot[2][0]) / S
		qz = (rot[1][0] - rot[0][1]) / S
	elif rot[0][0] > rot[1][1] and rot[0][0] > rot[2][2]:
		S = (1.0 + rot[0][0] - rot[1][1] - rot[2][2])**0.5 * 2
		qw = (rot[2][1] - rot[1][2]) / S
		qx = 0.25 * S
		qy = (rot[0][1] + rot[1][0]) / S
		qz = (rot[0][2] + rot[2][0]) / S
	elif rot[1][1] > rot[2][2]:
		S = (1.0 + rot[1][1] - rot[0][0] - rot[2][2])**0.5 * 2
		qw = (rot[0][2] - rot[2][0]) / S
		qx = (rot[0][1] + rot[1][0]) / S
		qy = 0.25 * S
		qz = (rot[1][2] + rot[2][1]) / S
	else:
		S = (1.0 + rot[2][2] - rot[0][0] - rot[1][1])**0.5 * 2
		qw = (rot[1][0] - rot[0][1]) / S
		qx = (rot[0][2] + rot[2][0]) / S
		qy = (rot[1][2] + rot[2][1]) / S
		qz = 0.25 * S

	return np.array([qw, qx, qy, qz])

class SensorNoise:
    def __init__(self, pos_norm_std=0.005, pos_unif_range=0.,
                        vel_norm_std=0.01, vel_unif_range=0.,
                        quat_norm_std=0., quat_unif_range=0., gyro_norm_std=0.,
                        gyro_noise_density=0.000175, gyro_random_walk=0.0105,
                        gyro_bias_correlation_time=1000., bypass=False,
                        acc_static_noise_std=0.002, acc_dynamic_noise_ratio=0.005,
                        use_numba=False):
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

        if use_numba:
            self.quat_from_small_angle_numba = njit()(quat_from_small_angle)
            self.rot2quat_numba = njit()(rot2quat)
            self.quat2R_numba = njit()(quat2R)
            self.quatXquat_numba = njit()(quatXquat)

    def add_noise(self, pos, vel, rot, omega, acc, dt):
        if self.bypass:
            return pos, vel, rot, omega, acc
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
                    normal(loc=0., scale=self.pos_norm_std, size=3) + \
                    uniform(low=-self.pos_unif_range, high=self.pos_unif_range, size=3)


        # add noise to linear velocity
        noisy_vel = vel + \
                    normal(loc=0., scale=self.vel_norm_std, size=3) + \
                    uniform(low=-self.vel_unif_range, high=self.vel_unif_range, size=3)

        ## Noise in omega
        if self.gyro_norm_std != 0.:
            noisy_omega = self.add_noise_to_omega(omega, dt)
        else:
            noisy_omega = omega + \
                    normal(loc=0., scale=self.gyro_noise_density, size=3)

        ## Noise in rotation
        theta = normal(0, self.quat_norm_std, size=3) + \
                uniform(-self.quat_unif_range, self.quat_unif_range, size=3)

        if rot.shape == (3,):
            ## Euler angles (xyz: roll=[-pi, pi], pitch=[-pi/2, pi/2], yaw = [-pi, pi])
            noisy_rot = np.clip(rot + theta,
                a_min=[-np.pi, -np.pi/2, -np.pi],
                a_max=[ np.pi,  np.pi/2,  np.pi])
        elif rot.shape == (3,3):
            ## Rotation matrix
            quat_theta = quat_from_small_angle(theta)
            quat = rot2quat(rot)
            noisy_quat = quatXquat(quat, quat_theta)
            noisy_rot = quat2R(noisy_quat[0], noisy_quat[1], noisy_quat[2], noisy_quat[3])
        elif rot.shape == (4,):
            ## Quaternion
            quat_theta = quat_from_small_angle(theta)
            noisy_rot = quatXquat(rot, quat_theta)
        else:
            raise ValueError("ERROR: SensNoise: Unknown rotation type: " + str(rot))

        ## Accelerometer noise
        noisy_acc = acc + normal(loc=0., scale=self.acc_static_noise_std, size=3) + \
                    acc * normal(loc=0., scale=self.acc_dynamic_noise_ratio, size=3)

        return noisy_pos, noisy_vel, noisy_rot, noisy_omega, noisy_acc

    def add_noise_numba(self, pos, vel, rot, omega, acc, dt):
        if self.bypass:
            return pos, vel, rot, omega, acc
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

        noisy_pos, noisy_vel, noisy_omega, noisy_acc, theta = add_noise_to_vel_acc_pos_omega_rot(pos, vel, omega, acc,
                                                             pos_rand_var=(self.pos_norm_std, self.pos_unif_range),
                                                             vel_rand_var=(self.vel_norm_std, self.vel_unif_range),
                                                             omega_rand_var=self.gyro_noise_density,
                                                             acc_rand_var=(self.acc_static_noise_std,
                                                                           self.acc_dynamic_noise_ratio),
                                                             rot_rand_var=(self.quat_norm_std, self.quat_unif_range))

        # Noise in omega
        if self.gyro_norm_std != 0.:
            noisy_omega = self.add_noise_to_omega(omega, dt)

        if rot.shape == (3,):
            ## Euler angles (xyz: roll=[-pi, pi], pitch=[-pi/2, pi/2], yaw = [-pi, pi])
            noisy_rot = np.clip(rot + theta,
                                a_min=[-np.pi, -np.pi / 2, -np.pi],
                                a_max=[np.pi, np.pi / 2, np.pi])
        elif rot.shape == (3, 3):
            ## Rotation matrix
            quat_theta = self.quat_from_small_angle_numba(theta)
            quat = self.rot2quat_numba(rot)
            noisy_quat = self.quatXquat_numba(quat, quat_theta)
            noisy_rot = self.quat2R_numba(noisy_quat[0], noisy_quat[1], noisy_quat[2], noisy_quat[3])
        elif rot.shape == (4,):
            ## Quaternion
            quat_theta = self.quat_from_small_angle_numba(theta)
            noisy_rot = self.quatXquat_numba(rot, quat_theta)
        else:
            raise ValueError("ERROR: SensNoise: Unknown rotation type: " + str(rot))

        return noisy_pos, noisy_vel, noisy_rot, noisy_omega, noisy_acc

    ## copy from rotorS imu plugin
    def add_noise_to_omega(self, omega, dt):
        assert omega.shape == (3,)

        sigma_g_d = self.gyro_noise_density / (dt**0.5)
        sigma_b_g_d = (-(sigma_g_d**2) * (self.gyro_bias_correlation_time / 2) * (exp(-2*dt/self.gyro_bias_correlation_time) - 1))**0.5
        pi_g_d = exp(-dt / self.gyro_bias_correlation_time)

        self.gyro_bias = pi_g_d * self.gyro_bias + sigma_b_g_d * normal(0, 1, 3)
        return omega + self.gyro_bias + self.gyro_random_walk * normal(0, 1, 3) # + self.gyro_turn_on_bias_sigma * normal(0, 1, 3)


@njit
def add_noise_to_vel_acc_pos_omega_rot(pos, vel, omega, acc, pos_rand_var, vel_rand_var, omega_rand_var,
                         acc_rand_var, rot_rand_var):

    # add noise to position measurement
    noisy_pos = pos + \
                normal(loc=0., scale=pos_rand_var[0], size=3) + \
                uniform(-pos_rand_var[1], pos_rand_var[1], 3)

    # Add noise to linear velocity
    noisy_vel = vel + \
                normal(loc=0., scale=vel_rand_var[0], size=3) + \
                uniform(-vel_rand_var[1], vel_rand_var[1], 3)

    # Noise in omega
    noisy_omega = omega + \
                  normal(loc=0., scale=omega_rand_var, size=3)

    # Noise in rotation
    theta = normal(loc=0, scale=rot_rand_var[0], size=3) + \
            uniform(-rot_rand_var[1], rot_rand_var[1], 3)

    # Accelerometer noise
    noisy_acc = acc + normal(loc=0., scale=acc_rand_var[0], size=3) + \
                (acc * normal(loc=0., scale=acc_rand_var[1], size=3))

    return noisy_pos, noisy_vel, noisy_omega, noisy_acc, theta


if __name__ == "__main__":
    sens = SensorNoise()
    import time
    start_time = time.time()
    sens.add_noise(np.zeros(3), np.zeros(3), np.eye(3), np.zeros(3), 0.005)
    print("Noise generation time: ", time.time() - start_time)
