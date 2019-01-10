#!/usr/bin/env python
"""
Quadrotor simulation for OpenAI Gym, with components reusable elsewhere.
Also see: D. Mellinger, N. Michael, V.Kumar. 
Trajectory Generation and Control for Precise Aggressive Maneuvers with Quadrotors
http://journals.sagepub.com/doi/pdf/10.1177/0278364911434236
"""
import argparse
import logging
import numpy as np
from numpy.linalg import norm
import sys
from copy import deepcopy
import matplotlib.pyplot as plt

import gym
from gym import spaces
from gym.utils import seeding
import gym.envs.registration as gym_reg

from garage.core import Serializable

from gym_art.quadrotor.quadrotor_control import *
from gym_art.quadrotor.quadrotor_visualization import *
from gym_art.quadrotor.quad_utils import *
import transforms3d as t3d

logger = logging.getLogger(__name__)

GRAV = 9.81
TILES = 256 # number of tiles used for the obstacle map
EPS = 1e-6 #small constant to avoid divisions by 0 and log(0)

# overall TODO:
# - SVD does not converge from time to time
# - fix front face CCW to enable culling
# - add texture coords to primitives
# - oracle policy for obstacles (already have for free space)
# - non-flat floor
# - fog

# Original goal-seeking reward. I don't really use it anymore
# since compute_reward() is mainly used
def goal_seeking_reward(dynamics, goal, action, dt, crashed, time_remain):
    if not crashed:
        # log to create a sharp peak at the goal
        dist = np.linalg.norm(goal - dynamics.pos)
        # loss_pos = np.log(dist + 0.1) + 0.1 * dist
        loss_pos = dist

        # dynamics_pos = dynamics.pos
        # print('dynamics.pos', dynamics.pos)

        # penalize altitude above this threshold
        max_alt = 6.0
        loss_alt = np.exp(2*(dynamics.pos[2] - max_alt))

        # penalize amount of control effort
        loss_effort = 0.001 * np.linalg.norm(action)

        # loss velocity
        dx = goal - dynamics.pos
        dx = dx / (np.linalg.norm(dx) + EPS)
        vel_direct = dynamics.vel / (np.linalg.norm(dynamics.vel) + EPS)
        vel_proj = np.dot(dx, vel_direct)
        # print('vel_proj:', vel_proj)
        loss_vel_proj = -dt *(vel_proj - 1.0)
        # print('loss_vel_proj:', loss_vel_proj)
        loss_crash = 0
    else:
        loss_pos = 0
        loss_alt = 0
        loss_effort = 0
        loss_vel_proj = 0
        loss_crash = dt * time_remain * 100 + 100

    reward = -dt * np.sum([loss_pos, loss_effort, loss_alt, loss_vel_proj, loss_crash])
    rew_info = {'rew_crash': -loss_crash, 'rew_altitude': -loss_alt, 'rew_action': -loss_effort, 'rew_pos': -loss_pos, 'rew_vel_proj': -loss_vel_proj}

    # print('reward: ', reward, ' pos:', dynamics.pos, ' action', action)
    # print('pos', dynamics.pos)
    if np.isnan(reward) or not np.isfinite(reward):
        for key, value in locals().items():
            print('%s: %s \n' % (key, str(value)))
        raise ValueError('QuadEnv: reward is Nan')

    return reward, rew_info

# simple simulation of quadrotor dynamics.
class QuadrotorDynamics(object):
    # mass unit: kilogram
    # arm_length unit: meter
    # inertia unit: kg * m^2, 3-element vector representing diagonal matrix
    # thrust_to_weight is the total, it will be divided among the 4 props
    # torque_to_thrust is ratio of torque produced by prop to thrust
    # thrust_noise is noise2signal ratio of the thrust noise, Ex: 0.05 = 5% of the current signal
    def __init__(self, mass, 
        arm_length, 
        inertia, 
        thrust_to_weight=2.0, 
        torque_to_thrust=0.05, 
        dynamics_steps_num=1, 
        room_box=None,
        dim_mode="3D",
        thrust_noise_ratio=0.0):
        assert np.isscalar(mass)
        assert np.isscalar(arm_length)
        assert inertia.shape == (3,)
        # This hack allows parametrize calling dynamics multiple times
        # without expensive for-loops

        self.dim_mode = dim_mode
        if self.dim_mode == '1D':
            self.control_mx = np.ones([4,1])
        elif self.dim_mode == '2D':
            self.control_mx = np.array([[1.,0.],[1.,0.],[0.,1.],[0.,1.]])
        elif self.dim_mode == '3D':
            self.control_mx = np.eye(4)
        else:
            raise ValueError('QuadEnv: Unknown dimensionality mode %s' % self.dim_mode)


        self.step = getattr(self, 'step%d' % dynamics_steps_num)
        if room_box is None:
            self.room_box = np.array([[-10., -10., 0.], [10., 10., 10.]])
        else:
            self.room_box = np.array(room_box).copy()

        self.vel_damp = 0.999
        self.damp_omega = 0.015
        self.mass = mass
        self.arm = arm_length
        self.inertia = inertia
        self.thrust_to_weight = thrust_to_weight
        self.thrust_max = GRAV * mass * thrust_to_weight / 4.0
        self.torque_max = torque_to_thrust * self.thrust_max # propeller torque scales
        self.thrust_noise_ratio = thrust_noise_ratio
        scl = arm_length / norm([1.,1.,0.])

        # Unscaled (normalized) propeller positions
        self.prop_pos = scl * np.array([
            [1.,  1., -1., -1.],
            [1., -1., -1.,  1.],
            [0.,  0.,  0.,  0.]]).T # row-wise easier with np
        # unit: meters^2 ??? maybe wrong
        self.prop_crossproducts = np.cross(self.prop_pos, [0., 0., 1.])
        # 1 for props turning CCW, -1 for CW
        self.prop_ccw = np.array([1., -1., 1., -1.])
        self.prop_ccw_mx = np.zeros([3,4]) # Matrix allows using matrix multiplication
        self.prop_ccw_mx[2,:] = self.prop_ccw 
        self.since_last_svd = 0

        ## Forced dynamics auxiliary matrices
        #Prop crossproduct give torque directions
        self.G_omega_thrust = self.thrust_max * self.prop_crossproducts.T # [3,4] @ [4,1]
        # additional torques along z-axis caused by propeller rotations
        self.C_omega_prop = self.torque_max * self.prop_ccw_mx  # [3,4] @ [4,1] = [3,1]
        self.G_omega = (1.0 / self.inertia)[:,None] * (self.G_omega_thrust + self.C_omega_prop)

        # Allows to sum-up thrusts as a linear matrix operation
        self.thrust_sum_mx = np.zeros([3,4]) # [0,0,F_sum].T
        self.thrust_sum_mx[2,:] = 1# [0,0,F_sum].T

        # sigma = 0.2 gives roughly max noise of -1 .. 1
        self.thrust_noise = OUNoise(4, sigma=0.2*self.thrust_noise_ratio)

    # pos, vel, in world coords (meters)
    # rotation is 3x3 matrix (body coords) -> (world coords)
    # omega is angular velocity (radians/sec) in body coords, i.e. the gyroscope
    def set_state(self, position, velocity, rotation, omega, thrusts=np.zeros((4,))):
        for v in (position, velocity, omega):
            assert v.shape == (3,)
        assert thrusts.shape == (4,)
        assert rotation.shape == (3,3)
        self.pos = deepcopy(position)
        self.vel = deepcopy(velocity)
        self.acc = np.zeros(3)
        self.accelerometer = np.array([0, 0, GRAV])
        self.rot = deepcopy(rotation)
        self.omega = deepcopy(omega.astype(np.float32))
        self.thrusts = deepcopy(thrusts)

    # generate a random state (meters, meters/sec, radians/sec)
    def random_state(self, np_random, box, vel_max=15.0, omega_max=2*np.pi):
        pos = np_random.uniform(low=-box, high=box, size=(3,))
        vel = np_random.uniform(low=-vel_max, high=vel_max, size=(3,))
        omega = np_random.uniform(low=-omega_max, high=omega_max, size=(3,))
        rot = rand_uniform_rot3d(np_random)
        self.set_state(pos, vel, rot, omega)

    # multiple dynamics steps
    def step2(self, thrust_cmds, dt):
        self.step1(thrust_cmds, dt)
        self.step1(thrust_cmds, dt)


    # multiple dynamics steps
    def step4(self, thrust_cmds, dt):
        self.step1(thrust_cmds, dt)
        self.step1(thrust_cmds, dt)
        self.step1(thrust_cmds, dt)
        self.step1(thrust_cmds, dt)
        # print('DYN: state:', self.state_vector(), 'thrust:', thrust_cmds, 'dt', dt)

    # multiple dynamics steps
    def step8(self, thrust_cmds, dt):
        self.step1(thrust_cmds, dt)
        self.step1(thrust_cmds, dt)
        self.step1(thrust_cmds, dt)
        self.step1(thrust_cmds, dt)
        self.step1(thrust_cmds, dt)
        self.step1(thrust_cmds, dt)
        self.step1(thrust_cmds, dt)
        self.step1(thrust_cmds, dt)

    ## Step function integrates based on current derivative values (best fits affine dynamics model)
    # thrust_cmds is motor thrusts given in normalized range [0, 1].
    # 1 represents the max possible thrust of the motor.
    ## Frames:
    # pos - global
    # vel - global
    # rot - global
    # omega - body frame
    # goal_pos - global
    def step1(self, thrust_cmds, dt):
        # import pdb; pdb.set_trace()
        # uncomment for debugging. they are slow
        #assert np.all(thrust_cmds >= 0)
        #assert np.all(thrust_cmds <= 1)

        ###################################
        ## Convert the motor commands to a force and moment on the body
        thrust_noise = thrust_cmds * self.thrust_noise.noise()
        thrust_cmds = np.clip(thrust_cmds + thrust_noise, 0.0, 1.0)
        thrusts = self.thrust_max * thrust_cmds
        #Prop crossproduct give torque directions
        torques = self.prop_crossproducts * thrusts[:,None] # (4,3)=(props, xyz)

        # additional torques along z-axis caused by propeller rotations
        torques[:, 2] += self.torque_max * self.prop_ccw * thrust_cmds 

        # net torque: sum over propellers
        thrust_torque = np.sum(torques, axis=0) 

        ###################################
        ## (Square) Damping using torques (in case we would like to add damping using torques)
        # damping_torque = - 0.3 * self.omega * np.fabs(self.omega)
        damping_torque = 0.0
        torque =  thrust_torque + damping_torque
        thrust = npa(0,0,np.sum(thrusts))

        #########################################################
        ## ROTATIONAL DYNAMICS

        ###################################
        ## Integrating rotations (based on current values)
        omega_vec = np.matmul(self.rot, self.omega) # Change from body2world frame
        x, y, z = omega_vec
        omega_mat_deriv = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])

        # ROtation matrix derivative
        dRdt = np.matmul(omega_mat_deriv, self.rot)
        self.rot += dt * dRdt

        # Occasionally orthogonalize the rotation matrix
        # It is necessary, since integration falls apart over time, thus
        # R matrix becomes non orthogonal (inconsistent)
        self.since_last_svd += 1
        if self.since_last_svd > 25:
            try:
                u, s, v = np.linalg.svd(self.rot)
                self.rot = np.matmul(u, v)
                self.since_last_svd = 0
            except Exception as e:
                print('Rotation Matrix: ', self.rot, ' actions: ', thrust_cmds)
                log_error('##########################################################')
                for key, value in locals().items():
                    log_error('%s: %s \n' %(key, str(value)))
                    print('%s: %s \n' %(key, str(value)))
                raise ValueError("QuadrotorEnv ERROR: SVD did not converge: " + str(e))
                # log_error('QuadrotorEnv: ' + str(e) + ': ' + 'Rotation matrix: ' + str(self.rot))

        ###################################
        ## COMPUTING OMEGA UPDATE

        ## Damping using velocities (I find it more stable numerically)
        ## Linear damping

        # This is only for linear dampling of angular velocity.
        # omega_damp = 0.999   
        # self.omega = omega_damp * self.omega + dt * omega_dot

        omega_dot = ((1.0 / self.inertia) *
            (cross(-self.omega, self.inertia * self.omega) + torque))

        ## Quadratic damping
        # 0.03 corresponds to roughly 1 revolution per sec
        omega_damp_quadratic = np.clip(0.015 * self.omega ** 2, a_min=0.0, a_max=1.0)
        self.omega = self.omega + (1.0 - omega_damp_quadratic) * dt * omega_dot

        ## When use square damping on torques - use simple integration
        ## since damping is accounted as part of the net torque
        # self.omega += dt * omega_dot

        #########################################################
        # TRANSLATIONAL DYNAMICS

        ## Room constraints
        mask = np.logical_or(self.pos <= self.room_box[0], self.pos >= self.room_box[1])
               
        ## Computing position
        self.pos = self.pos + dt * self.vel

        # Clipping if met the obstacle and nullify velocities (not sure what to do about accelerations)
        self.pos_before_clip = self.pos.copy()
        self.pos = np.clip(self.pos, a_min=self.room_box[0], a_max=self.room_box[1])
        # self.vel[np.equal(self.pos, self.pos_before_clip)] = 0.

        ## Computing accelerations
        acc = [0, 0, -GRAV] + (1.0 / self.mass) * np.matmul(self.rot, thrust)
        # acc[mask] = 0. #If we leave the room - stop accelerating
        self.acc = acc

        ## Computing velocities
        self.vel = self.vel_damp * self.vel + dt * acc
        # self.vel[mask] = 0. #If we leave the room - stop flying

        ## Accelerometer measures so called "proper acceleration" 
        # that includes gravity with the opposite sign
        self.accelerometer = np.matmul(self.rot.T, acc + [0, 0, GRAV])

    #######################################################
    ## AFFINE DYNAMICS REPRESENTATION:
    # s = dt*(F(s) + G(s)*u)
    
    ## Unforced dynamics (integrator, damping_deceleration)
    def F(self, s, dt):
        xyz  = s[0:3]
        Vxyz = s[3:6]
        rot = s[6:15].reshape([3,3])
        omega = s[15:18]
        goal = s[18:21]

        ###############################
        ## Linear position change
        dx = deepcopy(Vxyz)

        ###############################
        ## Linear velocity change
        dV = (self.vel_damp * Vxyz - Vxyz) / dt + np.array([0, 0, -GRAV])

        ###############################
        ## Angular orientation change
        omega_vec = np.matmul(rot, omega) # Change from body2world frame
        wx, wy, wz = omega_vec
        omega_mat_deriv = np.array([[0, -wz, wy], [wz, 0, -wx], [-wy, wx, 0]])

        # ROtation matrix derivative
        dR = np.matmul(omega_mat_deriv, rot).flatten()

        ###############################
        ## Angular rate change
        F_omega = (1.0 / self.inertia) * (cross(-omega, self.inertia * omega))
        omega_damp_quadratic = np.clip(self.damp_omega * omega ** 2, a_min=0.0, a_max=1.0)
        dOmega = (1.0 - omega_damp_quadratic) * F_omega

        ###############################
        ## Goal change
        dgoal = np.zeros_like(goal)

        return np.concatenate([dx, dV, dR, dOmega, dgoal])


    ## Forced affine dynamics (controlling acceleration only)
    def G(self, s):
        xyz  = s[0:3]
        Vxyz = s[3:6]
        rot = s[6:15].reshape([3,3])
        omega = s[15:18]
        goal = s[18:21]

        ###############################
        ## dx, dV, dR, dgoal
        dx = np.zeros([3,4])
        dV = (rot / self.mass ) @ (self.thrust_max * self.thrust_sum_mx)
        dR = np.zeros([9,4])
        dgoal = np.zeros([3,4])
        
        ###############################
        ## Angular acceleration
        omega_damp_quadratic = np.clip(self.damp_omega * omega ** 2, a_min=0.0, a_max=1.0)
        dOmega = (1.0 - omega_damp_quadratic)[:,None] * self.G_omega
        
        return np.concatenate([dx, dV, dR, dOmega, dgoal], axis=0) @ self.control_mx


    # return eye, center, up suitable for gluLookAt representing onboard camera
    def look_at(self):
        degrees_down = 45.0
        R = self.rot
        # camera slightly below COM
        eye = self.pos + np.matmul(R, [0, 0, -0.02])
        theta = np.radians(degrees_down)
        to, _ = normalize(np.cos(theta) * R[:,0] - np.sin(theta) * R[:,2])
        center = eye + to
        up = cross(to, R[:,1])
        return eye, center, up

    def state_vector(self):
        return np.concatenate([
            self.pos, self.vel, self.rot.flatten(), self.omega])

    def action_space(self):
        low = np.zeros(4)
        high = np.ones(4)
        return spaces.Box(low, high)


def default_dynamics(sim_steps, room_box, dim_mode, noise_scale):
    # similar to AscTec Hummingbird
    # TODO: dictionary of dynamics of real quadrotors
    mass = 0.5
    arm_length = 0.33 / 2.0
    inertia = mass * npa(0.01, 0.01, 0.02)
    thrust_to_weight = 2.0
    return QuadrotorDynamics(mass, arm_length, inertia,
        thrust_to_weight=thrust_to_weight, 
        dynamics_steps_num=sim_steps, 
        room_box=room_box, 
        dim_mode=dim_mode,
        thrust_noise_ratio=noise_scale)



# reasonable reward function for hovering at a goal and not flying too high
def compute_reward(dynamics, goal, action, dt, crashed, time_remain):
    ##################################################
    ## log to create a sharp peak at the goal
    dist = np.linalg.norm(goal - dynamics.pos)
    loss_pos = np.log(dist + 0.1) + 0.1 * dist
    # loss_pos = dist

    # dynamics_pos = dynamics.pos
    # print('dynamics.pos', dynamics.pos)

    ##################################################
    ## penalize altitude above this threshold
    # max_alt = 6.0
    # loss_alt = np.exp(2*(dynamics.pos[2] - max_alt))

    ##################################################
    # penalize amount of control effort
    loss_effort = 0.01 * np.linalg.norm(action)

    ##################################################
    ## loss velocity
    dx = goal - dynamics.pos
    dx = dx / (np.linalg.norm(dx) + EPS)
    
    ## normalized
    # vel_direct = dynamics.vel / (np.linalg.norm(dynamics.vel) + EPS)
    # vel_proj = np.dot(dx, vel_direct)
    
    vel_direct = dynamics.vel / (np.linalg.norm(dynamics.vel) + EPS)
    vel_magn = np.clip(np.linalg.norm(dynamics.vel),-1, 1)
    vel_clipped = vel_magn * vel_direct 
    vel_proj = np.dot(dx, vel_clipped)

    loss_vel_proj = - dist * vel_proj
    # print('vel_proj:', vel_proj)
    # print('loss_vel_proj:', loss_vel_proj)

    ##################################################
    ## Loss orientation
    loss_orient = -dynamics.rot[2,2] #Projection of the z-body axis to z-world axis


    ##################################################
    ## loss crash
    loss_crash = float(crashed)


    # reward = -dt * np.sum([loss_pos, loss_effort, loss_alt, loss_vel_proj, loss_crash])
    # rew_info = {'rew_crash': -loss_crash, 'rew_altitude': -loss_alt, 'rew_action': -loss_effort, 'rew_pos': -loss_pos, 'rew_vel_proj': -loss_vel_proj}


    reward = -dt * np.sum([
        loss_pos, 
        loss_effort, 
        loss_crash, 
        loss_vel_proj,
        loss_orient
        ])
    

    rew_info = {
    'rew_pos': -loss_pos, 
    'rew_action': -loss_effort, 
    'rew_crash': -loss_crash, 
    #'rew_altitude': -loss_alt, 
    'rew_vel_proj': -loss_vel_proj,
    "rew_orient": -loss_orient
    }



    # print('reward: ', reward, ' pos:', dynamics.pos, ' action', action)
    # print('pos', dynamics.pos)
    if np.isnan(reward) or not np.isfinite(reward):
        for key, value in locals().items():
            print('%s: %s \n' % (key, str(value)))
        raise ValueError('QuadEnv: reward is Nan')

    return reward, rew_info



# Gym environment for quadrotor seeking the origin
# with no obstacles and full state observations
class QuadrotorEnv(gym.Env, Serializable):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self, raw_control=True, raw_control_zero_middle=True, dim_mode='3D', tf_control=False, sim_steps=4,
                obs_repr="state_xyz_vxyz_rot_omega", ep_time=3, thrust_noise_ratio=0.):
        np.seterr(under='ignore')
        """
        @param obs_repr: options: state_xyz_vxyz_rot_omega, state_xyz_vxyz_quat_omega
        """
        self.room_box = np.array([[-10, -10, 0], [10, 10, 10]])
        self.obs_repr = obs_repr
        self.state_vector = getattr(self, obs_repr)

        self.dynamics = default_dynamics(sim_steps, room_box=self.room_box, dim_mode=dim_mode, noise_scale=thrust_noise_ratio)
        # self.controller = ShiftedMotorControl(self.dynamics)
        # self.controller = OmegaThrustControl(self.dynamics) ## The last one used
        # self.controller = VelocityYawControl(self.dynamics)
        self.scene = None
        # self.oracle = NonlinearPositionController(self.dynamics)
        self.dim_mode = dim_mode
        if self.dim_mode =='1D':
            self.viewpoint = 'side'
        else:
            self.viewpoint = 'chase'

        if raw_control:
            if self.dim_mode == '1D':
                self.controller = VerticalControl(self.dynamics, zero_action_middle=raw_control_zero_middle)
            elif self.dim_mode == '2D':
                self.controller = VertPlaneControl(self.dynamics, zero_action_middle=raw_control_zero_middle)
            elif self.dim_mode == '3D':
                self.controller = RawControl(self.dynamics, zero_action_middle=raw_control_zero_middle)
            else:
                raise ValueError('QuadEnv: Unknown dimensionality mode %s' % self.dim_mode)
        else:
            self.controller = NonlinearPositionController(self.dynamics, tf_control=tf_control)

        self.action_space = self.controller.action_space(self.dynamics)

        ## Former way to get obs space
        # # pos, vel, rot, rot vel
        # obs_dim = 3 + 3 + 9 + 3 + 3 # xyz, Vxyz, R, Omega, goal_xyz
        # # TODO tighter bounds on some variables
        # obs_high = 100 * np.ones(obs_dim)
        # # rotation mtx guaranteed to be orthogonal
        # obs_high[6:6+9] = 1
        # self.observation_space = spaces.Box(-obs_high, obs_high)

        self.observation_space = self.get_observation_space()


        # TODO get this from a wrapper
        self.ep_time = ep_time #In seconds
        self.dt = 1.0 / 100.0
        self.sim_steps = sim_steps
        self.ep_len = int(self.ep_time / (self.dt * self.sim_steps))
        self.tick = 0
        # self.dt = 1.0 / 50.0
        self.crashed = False

        self._seed()

        # size of the box from which initial position will be randomly sampled
        # if box_scale > 1.0 then it will also growevery episode
        self.box = 2.0
        self.box_scale = 1.0 #scale the initialbox by this factor eache episode

        self._reset()

        if self.spec is None:
            self.spec = gym_reg.EnvSpec(id='Quadrotor-v0', max_episode_steps=self.ep_len)
        
        # Always call Serializable constructor last
        Serializable.quick_init(self, locals())

    def state_xyz_vxyz_rot_omega(self):
        return np.concatenate([self.dynamics.state_vector(), self.goal[:3]])

    def state_xyz_vxyz_quat_omega(self):
        self.quat = R2quat(self.dynamics.rot)
        return np.concatenate([self.dynamics.pos, self.dynamics.vel, self.quat, self.dynamics.omega, self.goal[:3]])

    def state_xyz_vxyz_euler_omega(self):
        self.euler = t3d.euler.mat2euler(self.dynamics.rot)
        return np.concatenate([self.dynamics.pos, self.dynamics.vel, self.euler, self.dynamics.omega, self.goal[:3]])

    def get_observation_space(self):
        self.wall_offset = 0.3
        if self.obs_repr == "state_xyz_vxyz_rot_omega":
            ## Creating observation space
            # pos, vel, rot, rot vel
            self.obs_comp_sizes = [3, 3, 9, 3, 3]
            self.obs_comp_names = ["xyz", "Vxyz", "R", "Omega", "goal_xyz"]
            obs_dim = np.sum(self.obs_comp_sizes)
            # TODO tighter bounds on some variables
            obs_high =  np.ones(obs_dim)
            obs_low  = -np.ones(obs_dim)
            # xyz room constraints
            obs_high[0:3] = self.room_box[1]
            obs_low[0:3]  = self.room_box[0]

            # xyz room constraints
            obs_high[18:21] = self.room_box[1] - self.wall_offset
            obs_low[18:21]  = self.room_box[0] + self.wall_offset


        elif self.obs_repr == "state_xyz_vxyz_euler_omega":
             ## Creating observation space
            # pos, vel, rot, rot vel
            self.obs_comp_sizes = [3, 3, 3, 3, 3]
            self.obs_comp_names = ["xyz", "Vxyz", "euler", "Omega", "goal_xyz"]
            obs_dim = np.sum(self.obs_comp_sizes)
            # TODO tighter bounds on some variables
            obs_high =  np.ones(obs_dim)
            obs_low  = -np.ones(obs_dim)
            # xyz room constraints
            obs_high[0:3] = self.room_box[1]
            obs_low[0:3]  = self.room_box[0]

            # Euler angles
            obs_high[6:9] = np.pi*obs_high[6:9] 
            obs_low[6:9]  = np.pi*obs_low[6:9]

            # goal xyz room offseted
            obs_high[12:15] = self.room_box[1] - self.wall_offset
            obs_low[12:15]  = self.room_box[0] + self.wall_offset           

        elif self.obs_repr == "state_xyz_vxyz_quat_omega":
             ## Creating observation space
            # pos, vel, rot, rot vel
            self.obs_comp_sizes = [3, 3, 4, 3, 3]
            self.obs_comp_names = ["xyz", "Vxyz", "quat", "Omega", "goal_xyz"]
            obs_dim = np.sum(self.obs_comp_sizes)
            # TODO tighter bounds on some variables
            obs_high =  np.ones(obs_dim)
            obs_low  = -np.ones(obs_dim)
            # xyz room constraints
            obs_high[0:3] = self.room_box[1]
            obs_low[0:3]  = self.room_box[0]

            # goal xyz room offseted
            obs_high[13:16] = self.room_box[1] - self.wall_offset
            obs_low[13:16]  = self.room_box[0] + self.wall_offset  


        self.observation_space = spaces.Box(obs_low, obs_high)
        return self.observation_space

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        # print('actions: ', action)
        # if not self.crashed:
        # print('goal: ', self.goal, 'goal_type: ', type(self.goal))
        self.controller.step_func(dynamics=self.dynamics,
                                action=action,
                                goal=self.goal,
                                dt=self.dt,
                                observation=np.expand_dims(self.state_vector(), axis=0))
        # self.oracle.step(self.dynamics, self.goal, self.dt)
        self.crashed = self.scene.update_state(self.dynamics)
        self.crashed = self.crashed or not np.array_equal(self.dynamics.pos,
                                                      np.clip(self.dynamics.pos,
                                                              a_min=self.room_box[0],
                                                              a_max=self.room_box[1]))

        self.time_remain = self.ep_len - self.tick
        reward, rew_info = compute_reward(self.dynamics, self.goal, action, self.dt, self.crashed, self.time_remain)
        self.tick += 1
        done = self.tick > self.ep_len #or self.crashed
        sv = self.state_vector()

        # print('state', sv, 'goal', self.goal)
        # print('vel', sv[3], sv[4], sv[5])
        return sv, reward, done, {'rewards': rew_info}

    def _reset(self):
        if self.scene is None:
            self.scene = Quadrotor3DScene(None, self.dynamics.arm,
                640, 480, resizable=True, obstacles=False, viewpoint=self.viewpoint)

        self.goal = np.array([0., 0., 2.])
        # print('reset goal: ', self.goal)
        x, y, z = self.np_random.uniform(-self.box, self.box, size=(3,)) + self.goal
        if self.dim_mode == '1D':
            x = self.goal[0]
            y = self.goal[1]
        elif self.dim_mode == '2D':
            y = self.goal[1]
        if z < 0.25 : z = 0.25
        if self.box < 10:
            # from 0.5 to 10 after 100k episodes
            nextbox = self.box * self.box_scale
            if int(4*nextbox) > int(4*self.box):
                print("box:", nextbox)
            self.box = nextbox
        pos = npa(x, y, z)
        vel = omega = npa(0, 0, 0)

        def randrot():
            rotz = np.random.uniform(-np.pi, np.pi)
            return r3d.rotz(rotz)[:3,:3]

        if self.dim_mode == '1D' or self.dim_mode == '2D':
            rotation = np.eye(3)
        else:
            # make sure we're sort of pointing towards goal
            rotation = randrot()
            while np.dot(rotation[:,0], to_xyhat(-pos)) < 0.5:
                rotation = randrot()

        self.dynamics.set_state(pos, vel, rotation, omega)

        self.scene.reset(self.goal, self.dynamics)
        self.scene.update_state(self.dynamics)

        self.crashed = False
        self.tick = 0
        # if self.ep_len < 1000:
        #     self.ep_len += 0.01 # len 1000 after 100k episodes

        state = self.state_vector()
        #That helps to avoid including goals xyz into the observation space
        # print('state', state)
        return state

    def _render(self, mode='human', close=False):
        return self.scene.render_chase(mode=mode)
    
    def reset(self):
        return self._reset()

    def render(self, mode='human', **kwargs):
        return self._render(mode, **kwargs)
    
    def step(self, action):
        return self._step(action)


def test_rollout():
    #############################
    # Init plottting
    fig = plt.figure(1)
    # ax = plt.subplot(111)
    plt.show(block=False)

    render = True
    plot_step = 50
    time_limit = 25
    render_each = 2
    rollouts_num = 10
    plot_obs = False

    env = QuadrotorEnv(raw_control=False, sim_steps=4)

    env.max_episode_steps = time_limit
    print('Reseting env ...')

    try:
        print('Observation space:', env.observation_space.low, env.observation_space.high)
        print('Action space:', env.action_space.low, env.action_space.high)
    except:
        print('Observation space:', env.observation_space.spaces[0].low, env.observation_space[0].spaces[0].high)
        print('Action space:', env.action_space[0].spaces[0].low, env.action_space[0].spaces[0].high)
    # input('Press any key to continue ...')

    action = [0.5, 0.5, 0.5, 0.5]
    rollouts_id = 0

    while rollouts_id < rollouts_num:
        rollouts_id += 1
        s = env.reset()
        ## Diagnostics
        observations = []
        velocities = []

        t = 0
        while True:
            if render and (t % render_each == 0): env.render()
            s, r, done, info = env.step(action)
            observations.append(s)
            # print('Step: ', t, ' Obs:', s)

            if t % plot_step == 0:
                plt.clf()

                if plot_obs:
                    observations_arr = np.array(observations)
                    # print('observations array shape', observations_arr.shape)
                    dimenstions = observations_arr.shape[1]
                    for dim in range(dimenstions):
                        plt.plot(observations_arr[:, dim])
                    plt.legend([str(x) for x in range(observations_arr.shape[1])])

                plt.pause(0.05) #have to pause otherwise does not draw
                plt.draw()
            if done: break
            t += 1
    # print('Rollouts are done!')
    # plt.pause(2.0)
    # plt.waitforbuttonpress()
    input("Press Enter to continue...")

def main(argv):
    # parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-m',"--mode",
        type=int,
        default=0,
        help="Test mode: "
             "0 - rollout with default controller"
    )
    # parser.add_argument(
    #     '-e',"--env_id",
    #     type=int,
    #     default=0,
    #     help="Env ID: "
    #          "0 - Quad"
    # )
    args = parser.parse_args()

    if args.mode == 0:
        print('Running test rollout ...')
        test_rollout()

if __name__ == '__main__':
    main(sys.argv)
