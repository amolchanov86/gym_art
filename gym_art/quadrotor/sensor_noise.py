import numpy as np
from numpy.random import normal 
from numpy.random import uniform
import matplotlib.pyplot as plt
from math import exp

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

class SensorNoise:
	def __init__(self, pos_norm_std=0., pos_unif_range=0., 
					   vel_norm_std=0., vel_unif_range=0., 
					   quat_norm_std=0., quat_unif_range=0., 
					   gyro_noise_density=0.000175, gyro_random_walk=0.0105, 
                       gyro_bias_correlation_time=1000., gyro_turn_on_bias_sigma=0.09): 
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
            gyro_gyro_turn_on_bias_sigma: gyroscope noise, MPU-9250 spec
		"""

		self.pos_norm_std = pos_norm_std
		self.pos_unif_range = pos_unif_range

		self.vel_norm_std = vel_norm_std
		self.vel_unif_range = vel_unif_range

		self.quat_norm_std = quat_norm_std
		self.noise_uniform_theta = noise_uniform_theta

		self.gyro_noise_density = gyro_noise_density
		self.gyro_random_walk = gyro_random_walk
		self.gyro_bias_correlation_time = gyro_bias_correlation_time
		self.gyro_turn_on_bias_sigma = gyro_turn_on_bias_sigma
		self.gyro_bias = np.zeros(3)


	def noise(self, pos, vel, rot, omega, dt):
        """
        Args: 
            pos: ground truth of the position in world frame
            vel: ground truth if the linear velocity in world frame
            rot: ground truth of the orientation in rotational matrix
            omega: ground truth of the angular velocity in body frame
            dt: integration step
        """
		assert pos.shape == (3,)
		assert vel.shape == (3,)
		assert rot.shape == (3,3)
		assert omega.shape == (3,)

		# add noise to position measurement
        noisy_pos = pos + \
                    normal(loc=0., self.pos_norm_std, size=3) + \
                    uniform(low=-self.pos_unif_range, high=self.pos_unif_range, size=3)


		# add noise to linear velocity
        noisy_vel = vel + \
                    normal(loc=0., self.vel_norm_std, size=3) + \
                    uniform(low=-self.vel_unif_range, high=self.vel_unif_range, size=3)

		# add noise to orientation
		quat = rot2quat(rot)
        theta = normal(0, self.quat_norm_std) + \
                uniform(-self.quat_unif_range, self.quat_unif_range)
		
		# convert theta to quaternion
		quat_theta = quat_from_small_angle(theta)

		noisy_quat = np.zeros(4)
		## quat * quat_theta
		noisy_quat[0] = quat[0] * quat_theta[0] - quat[1] * quat_theta[1] - quat[2] * quat_theta[2] - quat[3] * quat_theta[3] 
		noisy_quat[1] = quat[0] * quat_theta[1] + quat[1] * quat_theta[0] - quat[2] * quat_theta[3] + quat[3] * quat_theta[2] 
		noisy_quat[2] = quat[0] * quat_theta[2] + quat[1] * quat_theta[3] + quat[2] * quat_theta[0] - quat[3] * quat_theta[1] 
		noisy_quat[3] = quat[0] * quat_theta[3] - quat[1] * quat_theta[2] + quat[2] * quat_theta[1] + quat[3] * quat_theta[0]

		noisy_rot = quat2rot(noisy_quat)
		noisy_omega = self.add_noise_to_omega(omega, dt)

		return noisy_pos, noisy_vel, noisy_rot, noisy_omega

	## copy from rotorS imu plugin
	def add_noise_to_omega(self, omega, dt):
		assert omega.shape == (3,)

		sigma_g_d = self.gyro_noise_density / (dt**0.5)
		sigma_b_g_d = (-(sigma_g_d**2) * (self.gyro_bias_correlation_time / 2) * (exp(-2*dt/self.gyro_bias_correlation_time) - 1))**0.5
		pi_g_d = exp(-dt / self.gyro_bias_correlation_time)

		self.gyro_bias = pi_g_d * self.gyro_bias + sigma_b_g_d * normal(0, 1, 3)
		omega = omega + self.gyro_bias + self.gyro_random_walk * normal(0, 1, 3) + self.gyro_turn_on_bias_sigma * normal(0, 1, 3)

		return omega