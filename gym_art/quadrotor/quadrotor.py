#!/usr/bin/env python
"""
Quadrotor simulation for OpenAI Gym, with components reusable elsewhere.
Also see: D. Mellinger, N. Michael, V.Kumar. 
Trajectory Generation and Control for Precise Aggressive Maneuvers with Quadrotors
http://journals.sagepub.com/doi/pdf/10.1177/0278364911434236

Developers:
James Preiss, Artem Molchanov, Tao Chen 
"""
import argparse
import logging
import numpy as np
from numpy.linalg import norm
import sys
from copy import deepcopy
import matplotlib.pyplot as plt
import time

import gym
from gym import spaces
from gym.utils import seeding
import gym.envs.registration as gym_reg

import transforms3d as t3d

from gym_art.quadrotor.quadrotor_control import *
from gym_art.quadrotor.quadrotor_obstacles import *
from gym_art.quadrotor.quadrotor_visualization import *
from gym_art.quadrotor.quad_utils import *
from gym_art.quadrotor.inertia import QuadLink
from gym_art.quadrotor.sensor_noise import SensorNoise

try:
    from garage.core import Serializable
except:
    print("WARNING: garage.core.Serializable is not found. Substituting with a dummy class")
    class Serializable:
        def __init__(self):
            pass
        def quick_init(self, locals_in):
            pass


logger = logging.getLogger(__name__)

GRAV = 9.81 #default gravitational constant
EPS = 1e-6 #small constant to avoid divisions by 0 and log(0)

## WARN:
# - linearity is set to 1 always, by means of check_quad_param_limits(). 
# The def. value of linarity for CF is set to 1 as well (due to firmware nonlinearity compensation)

## TODO:
# -
# -
# -

def crazyflie_params():
    ## See: http://mikehamer.info/assets/papers/Crazyflie%20Modelling.pdf
    ## Geometric parameters for Inertia and the model
    geom_params = {}
    geom_params["body"] = {"l": 0.03, "w": 0.03, "h": 0.004, "m": 0.005}
    geom_params["payload"] = {"l": 0.035, "w": 0.02, "h": 0.008, "m": 0.01}
    geom_params["arms"] = {"l": 0.022, "w":0.005, "h":0.005, "m":0.001}
    geom_params["motors"] = {"h":0.02, "r":0.0035, "m":0.0015}
    geom_params["propellers"] = {"h":0.002, "r":0.022, "m":0.00075}
    
    geom_params["motor_pos"] = {"xyz": [0.065/2, 0.065/2, 0.]}
    geom_params["arms_pos"] = {"angle": 45., "z": 0.}
    geom_params["payload_pos"] = {"xy": [0., 0.], "z_sign": 1}
    # z_sing corresponds to location (+1 - on top of the body, -1 - on the bottom of the body)

    ## Damping parameters
    damp_params = {"vel": 0.001, "omega_quadratic": 0.015}

    ## Noise parameters
    noise_params = {}
    noise_params["thrust_noise_ratio"] = 0.01

    ## Motor parameters
    motor_params = {"thrust_to_weight" : 1.9, #2.18
                    "torque_to_thrust": 0.006, #0.005964552
                    "linearity": 1. #0.424
                    }

    ## Summarizing
    params = {
        "geom": geom_params, 
        "damp": damp_params, 
        "noise": noise_params,
        "motor": motor_params
    }
    return params


def defaultquad_params():
    # Similar to AscTec Hummingbird: http://www.asctec.de/en/uav-uas-drones-rpas-roav/asctec-hummingbird/
    ## Geometric parameters for Inertia and the model
    geom_params = {}
    geom_params["body"] = {"l": 0.1, "w": 0.1, "h": 0.085, "m": 0.5}
    geom_params["payload"] = {"l": 0.12, "w": 0.12, "h": 0.04, "m": 0.1}
    geom_params["arms"] = {"l": 0.1, "w":0.015, "h":0.015, "m":0.025} #0.17 total arm
    geom_params["motors"] = {"h":0.02, "r":0.025, "m":0.02}
    geom_params["propellers"] = {"h":0.001, "r":0.1, "m":0.009}
    
    geom_params["motor_pos"] = {"xyz": [0.12, 0.12, 0.]}
    geom_params["arms_pos"] = {"angle": 45., "z": 0.}
    geom_params["payload_pos"] = {"xy": [0., 0.], "z_sign": -1}
    # z_sing corresponds to location (+1 - on top of the body, -1 - on the bottom of the body)
    
    ## Damping parameters
    damp_params = {"vel": 0.001, "omega_quadratic": 0.015}

    ## Noise parameters
    noise_params = {}
    noise_params["thrust_noise_ratio"] = 0.01
    
    ## Motor parameters
    motor_params = {"thrust_to_weight" : 2.8,
                    "torque_to_thrust": 0.05,
                    "linearity": 1.0
                    }

    ## Summarizing
    params = {
        "geom": geom_params, 
        "damp": damp_params, 
        "noise": noise_params,
        "motor": motor_params
    }
    return params

def clip_params_positive(params):
    def clip_positive(key, item):
        return np.clip(item, a_min=0., a_max=None)
    walk_dict(params, clip_positive)
    return params

def check_quad_param_limits(params, params_init=None):
    ## Body parameters (like lengths and masses) are always positive
    for key in ["body", "payload", "arms", "motors", "propellers"]:
        params["geom"][key] = clip_params_positive(params["geom"][key])

    params["geom"]["motor_pos"]["xyz"][:2] = np.clip(params["geom"]["motor_pos"]["xyz"][:2], a_min=0.005, a_max=None)
    body_w = params["geom"]["body"]["w"]
    params["geom"]["payload_pos"]["xy"] = np.clip(params["geom"]["payload_pos"]["xy"], a_min=-body_w/4., a_max=body_w/4.)    
    params["geom"]["arms_pos"]["angle"] = np.clip(params["geom"]["arms_pos"]["angle"], a_min=0., a_max=90.)    
    
    ## Damping parameters
    params["damp"]["vel"] = np.clip(params["damp"]["vel"], a_min=0.00000, a_max=1.)
    params["damp"]["omega_quadratic"] = np.clip(params["damp"]["omega_quadratic"], a_min=0.00000, a_max=1.)
    
    ## Motor parameters
    params["motor"]["thrust_to_weight"] = np.clip(params["motor"]["thrust_to_weight"], a_min=1.2, a_max=None)
    params["motor"]["torque_to_thrust"] = np.clip(params["motor"]["torque_to_thrust"], a_min=0.001, a_max=1.)
    params["motor"]["linearity"] = np.clip(params["motor"]["linearity"], a_min=0.999, a_max=1.)

    ## Make sure propellers make sense in size
    if params_init is not None:
        r0 = params_init["geom"]["propellers"]["r"]
        t2w, t2w0 = params_init["motor"]["thrust_to_weight"], params["motor"]["thrust_to_weight"]
        params["geom"]["propellers"]["r"] = r0 * (t2w/t2w0)**0.5

    return params

def get_dyn_randomization_params(quad_params, noise_ratio=0., noise_ratio_params=None):
    """
    The function updates noise params
    Args:
        noise_ratio (float): ratio of change relative to the nominal values
        noise_ratio_params (dict): if for some parameters you want to have different ratios relative to noise_ratio,
            you can provided it through this dictionary
    Returns:
        noise_params dictionary
    """
    ## Setting the initial noise ratios (nominal ones)
    noise_params = deepcopy(quad_params)
    def set_noise_ratio(key, item):
        if isinstance(item, str):
            return None
        else:
            return noise_ratio
    
    walk_dict(noise_params, set_noise_ratio)

    ## Updating noise ratios
    if noise_ratio_params is not None:
        noise_params.update(noise_ratio_params)
    return noise_params


def perturb_dyn_parameters(params, noise_params, sampler="normal"):
    """
    The function samples around nominal parameters provided noise parameters
    Args:
        params (dict): dictionary of quadrotor parameters
        noise_params (dict): dictionary of noise parameters with the same hierarchy as params, but
            contains ratio of deviation from the params
    Returns:
        dict: modified parameters
    """
    ## Sampling parameters
    def sample_normal(key, param_val, ratio):
        #2*ratio since 2std contain 98% of all samples
        param_val_sample = np.random.normal(loc=param_val, scale=np.abs((ratio/2)*np.array(param_val)))
        return param_val_sample, ratio
    
    def sample_uniform(key, param_val, ratio):
        param_val = np.array(param_val)
        return np.random.uniform(low=param_val - param_val*ratio, high=param_val + param_val*ratio), ratio

    sample_param = locals()["sample_" + sampler]

    params_new = deepcopy(params)
    walk_2dict(params_new, noise_params, sample_param)

    ## Fixing a few parameters if they go out of allowed limits
    params_new = check_quad_param_limits(params_new, params)
    # print_dic(params_new)

    return params_new

def sample_dyn_parameters():
    """
    The function samples parameters for all possible quadrotors
    Args:
        scale (float): scale of sampling
    Returns:
        dict: sampled quadrotor parameters
    """
    ###################################################################
    ## DENSITIES (body, payload, arms, motors, propellers)
    # Crazyflie estimated body / payload / arms / motors / props density: 1388.9 / 1785.7 / 1777.8 / 1948.8 / 246.6 kg/m^3
    # Hummingbird estimated body / payload / arms / motors/ props density: 588.2 / 173.6 / 1111.1 / 509.3 / 246.6 kg/m^3
    geom_params = {}
    dens_val = np.random.uniform(
        low=[500., 150., 500., 500., 240.], 
        high=[1500., 1800., 1800., 1800., 250.])
    
    geom_params["body"] = {"density": dens_val[0]}
    geom_params["payload"] = {"density": dens_val[1]}
    geom_params["arms"] = {"density": dens_val[2]}
    geom_params["motors"] = {"density": dens_val[3]}
    geom_params["propellers"] = {"density": dens_val[4]}

    ###################################################################
    ## GEOMETRIES
    # MOTORS (and overal size)
    total_w = np.random.uniform(low=0.07, high=0.3)
    total_l = np.clip(np.random.normal(loc=1., scale=0.25), a_min=1.0, a_max=None) * total_w
    motor_z = np.random.normal(loc=0., scale=total_w / 8.)
    geom_params["motor_pos"] = {"xyz": [total_w / 2., total_l / 2., motor_z]}
    geom_params["motors"]["r"] = total_w * np.random.normal(loc=0.05, scale=0.005)
    geom_params["motors"]["h"] = geom_params["motors"]["r"] * np.random.normal(loc=1.0, scale=0.05)
    
    # BODY
    geom_params["body"]["w"] =  np.random.normal(loc=0.4, scale=0.1) * total_w
    geom_params["body"]["l"] =  np.clip(np.random.normal(loc=1., scale=0.25), a_min=1.0, a_max=None) * geom_params["body"]["w"]
    geom_params["body"]["h"] =  np.random.uniform(low=0.1, high=1.5) * geom_params["body"]["w"]

    # PAYLOAD
    pl_scl = np.random.uniform(low=0.25, high=1.0, size=3)
    geom_params["payload"]["w"] =  pl_scl[0] * geom_params["body"]["w"]
    geom_params["payload"]["l"] =  pl_scl[1] * geom_params["body"]["l"]
    geom_params["payload"]["h"] =  pl_scl[2] * geom_params["body"]["h"]
    geom_params["payload_pos"] = {
            "xy": np.random.normal(loc=0., scale=geom_params["body"]["w"] / 8., size=2), 
            "z_sign": np.sign(np.random.uniform(low=-1, high=1))}
    # z_sing corresponds to location (+1 - on top of the body, -1 - on the bottom of the body)

    # ARMS
    geom_params["arms"]["w"] = total_w * np.random.normal(loc=0.05, scale=0.005)
    geom_params["arms"]["h"] = total_w * np.random.normal(loc=0.05, scale=0.005)
    geom_params["arms_pos"] = {"angle": np.random.normal(loc=45., scale=10.), "z": motor_z - geom_params["motors"]["h"]/2.}
    
    # PROPS
    thrust_to_weight = np.random.uniform(low=1.8, high=2.8)
    geom_params["propellers"]["h"] = 0.01
    geom_params["propellers"]["r"] = (0.3) * total_w * (thrust_to_weight / 2.0)**0.5
    
    ## Damping parameters
    damp_vel_scale = np.random.uniform(low=0.01, high=2.)
    damp_omega_scale = damp_vel_scale * np.random.uniform(low=0.75, high=1.25)
    damp_params = {
        "vel": 0.001 * damp_vel_scale, 
        "omega_quadratic": 0.015 * damp_omega_scale}

    ## Noise parameters
    noise_params = {}
    noise_params["thrust_noise_ratio"] = np.random.uniform(low=0.001, high=0.02) #0.01
    
    ## Motor parameters
    motor_params = {"thrust_to_weight" : thrust_to_weight,
                    "torque_to_thrust": np.random.uniform(low=0.003, high=0.03), #0.05 originally
                    "linearity": 1.0
                    # "linearity": np.random.normal(loc=0.5, scale=0.1)
                    }

    ## Summarizing
    params = {
        "geom": geom_params, 
        "damp": damp_params, 
        "noise": noise_params,
        "motor": motor_params
    }

    ## Checking everything
    params = check_quad_param_limits(params=params)
    return params

class QuadrotorDynamics(object):
    """
    Simple simulation of quadrotor dynamics.
    mass unit: kilogram
    arm_length unit: meter
    inertia unit: kg * m^2, 3-element vector representing diagonal matrix
    thrust_to_weight is the total, it will be divided among the 4 props
    torque_to_thrust is ratio of torque produced by prop to thrust
    thrust_noise_ratio is noise2signal ratio of the thrust noise, Ex: 0.05 = 5% of the current signal
      It is an approximate ratio, i.e. the upper bound could still be higher, due to how OU noise operates
    Coord frames: x configuration:
     - x axis between arms looking forward [x - configuration]
     - y axis pointing to the left
     - z axis up 
    TODO:
    - only diagonal inertia is used at the moment
    """
    def __init__(self, model_params,
        room_box=None,
        dynamics_steps_num=1, 
        dim_mode="3D",
        gravity=GRAV):

        self.dynamics_steps_num = dynamics_steps_num
        ###############################################################
        ## PARAMETERS 
        self.prop_ccw = np.array([-1., 1., -1., 1.])
        # cw = 1 ; ccw = -1 [ccw, cw, ccw, cw]
        # Reference: https://docs.google.com/document/d/1wZMZQ6jilDbj0JtfeYt0TonjxoMPIgHwYbrFrMNls84/edit
        self.omega_max = 40. #rad/s The CF sensor can only show 35 rad/s (2000 deg/s), we allow some extra
        self.vxyz_max = 3. #m/s
        self.gravity = gravity

        ###############################################################
        ## Internal State variables
        self.since_last_ort_check = 0 #counter
        self.since_last_ort_check_limit = 0.04 #when to check for non-orthogonality
        
        self.rot_nonort_limit = 0.01 # How much of non-orthogonality in the R matrix to tolerate
        self.rot_nonort_coeff_maxsofar = 0. # Statistics on the max number of nonorthogonality that we had
        
        self.since_last_svd = 0 #counter
        self.since_last_svd_limit = 0.5 #in sec - how ofthen mandatory orthogonalization should be applied

        self.eye = np.eye(3)
        ###############################################################
        ## Initializing model
        self.update_model(model_params)
        
        ## Sanity checks
        assert self.inertia.shape == (3,)

        ###############################################################
        ## OTHER PARAMETERS
        if room_box is None:
            self.room_box = np.array([[-10., -10., 0.], [10., 10., 10.]])
        else:
            self.room_box = np.array(room_box).copy()

        ## Selecting 1D, Planar or Full 3D modes
        self.dim_mode = dim_mode
        if self.dim_mode == '1D':
            self.control_mx = np.ones([4,1])
        elif self.dim_mode == '2D':
            self.control_mx = np.array([[1.,0.],[1.,0.],[0.,1.],[0.,1.]])
        elif self.dim_mode == '3D':
            self.control_mx = np.eye(4)
        else:
            raise ValueError('QuadEnv: Unknown dimensionality mode %s' % self.dim_mode)

        ## Selecting how many sim steps should be done b/w controller calls
        # i.e. controller frequency
        self.step = getattr(self, 'step%d' % dynamics_steps_num)

    @staticmethod
    def angvel2thrust(w, linearity=0.424):
        """
        Args:
            linearity (float): linearity factor factor [0 .. 1].
            CrazyFlie: linearity=0.424
        """
        return  (1 - linearity) * w**2 + linearity * w

    def update_model(self, model_params):
        self.model = QuadLink(params=model_params["geom"])
        self.model_params = model_params

        ###############################################################
        ## PARAMETERS FOR RANDOMIZATION
        self.mass = self.model.m
        self.inertia = np.diagonal(self.model.I_com)
        self.thrust_to_weight = self.model_params["motor"]["thrust_to_weight"]
        self.torque_to_thrust = self.model_params["motor"]["torque_to_thrust"]
        self.motor_linearity = self.model_params["motor"]["linearity"]
        self.thrust_noise_ratio = self.model_params["noise"]["thrust_noise_ratio"]
        self.vel_damp = self.model_params["damp"]["vel"]
        self.damp_omega_quadratic = self.model_params["damp"]["omega_quadratic"]

        ###############################################################
        ## COMPUTED (Dependent) PARAMETERS
        self.thrust_max = GRAV * self.mass * self.thrust_to_weight / 4.0
        self.torque_max = self.torque_to_thrust * self.thrust_max # propeller torque scales

        # Propeller positions in X configurations
        self.prop_pos = self.model.prop_pos

        # unit: meters^2 ??? maybe wrong
        self.prop_crossproducts = np.cross(self.prop_pos, [0., 0., 1.])
        self.prop_ccw_mx = np.zeros([3,4]) # Matrix allows using matrix multiplication
        self.prop_ccw_mx[2,:] = self.prop_ccw 

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

        self.arm = np.linalg.norm(self.model.motor_xyz[:2])

        self.step = getattr(self, 'step%d' % self.dynamics_steps_num)

    # pos, vel, in world coords (meters)
    # rotation is 3x3 matrix (body coords) -> (world coords)dt
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
    def random_state(self, box, vel_max=15.0, omega_max=2*np.pi):
        pos = np.random.uniform(low=-box, high=box, size=(3,))
        vel = np.random.uniform(low=-vel_max, high=vel_max, size=(3,))
        omega = np.random.uniform(low=-omega_max, high=omega_max, size=(3,))
        rot = rand_uniform_rot3d()
        return pos, vel, rot, omega
        # self.set_state(pos, vel, rot, omega)

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
    def step6(self, thrust_cmds, dt):
        self.step1(thrust_cmds, dt)
        self.step1(thrust_cmds, dt)
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

    # multiple dynamics steps
    def step10(self, thrust_cmds, dt):
        self.step1(thrust_cmds, dt)
        self.step1(thrust_cmds, dt)
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
    # from numba import jit, autojit
    # @autojit
    def step1(self, thrust_cmds, dt):
        # print("thrust_cmds:", thrust_cmds)
        # uncomment for debugging. they are slow
        #assert np.all(thrust_cmds >= 0)
        #assert np.all(thrust_cmds <= 1)

        ###################################
        ## Convert the motor commands to a force and moment on the body
        thrust_noise = thrust_cmds * self.thrust_noise.noise()
        thrust_cmds = np.clip(thrust_cmds + thrust_noise, 0.0, 1.0)

        thrusts = self.thrust_max * self.angvel2thrust(thrust_cmds, linearity=self.motor_linearity)
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
        self.since_last_svd += dt
        self.since_last_ort_check += dt
        if self.since_last_ort_check >= self.since_last_ort_check_limit:
            self.since_last_ort_check = 0.
            nonort_coeff = np.sum(np.abs(self.rot @ self.rot.T - self.eye))
            self.rot_nonort_coeff_maxsofar = max(nonort_coeff, self.rot_nonort_coeff_maxsofar)
            if nonort_coeff > self.rot_nonort_limit or self.since_last_svd > self.since_last_svd_limit:
                ## Perform SVD corrdetions
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

        # This is only for linear damping of angular velocity.
        # omega_damp = 0.999   
        # self.omega = omega_damp * self.omega + dt * omega_dot

        omega_dot = ((1.0 / self.inertia) *
            (cross(-self.omega, self.inertia * self.omega) + torque))

        ## Quadratic damping
        # 0.03 corresponds to roughly 1 revolution per sec
        omega_damp_quadratic = np.clip(self.damp_omega_quadratic * self.omega ** 2, a_min=0.0, a_max=1.0)
        self.omega = self.omega + (1.0 - omega_damp_quadratic) * dt * omega_dot
        self.omega = np.clip(self.omega, a_min=-self.omega_max, a_max=self.omega_max)

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
        self.vel = (1.0 - self.vel_damp) * self.vel + dt * acc
        # self.vel[mask] = 0. #If we leave the room - stop flying

        ## Accelerometer measures so called "proper acceleration" 
        # that includes gravity with the opposite sign
        self.accelerometer = np.matmul(self.rot.T, acc + [0, 0, self.gravity])

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
        dV = ((1.0 - self.vel_damp) * Vxyz - Vxyz) / dt + np.array([0, 0, -GRAV])

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
        omega_damp_quadratic = np.clip(self.damp_omega_quadratic * omega ** 2, a_min=0.0, a_max=1.0)
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
        omega_damp_quadratic = np.clip(self.damp_omega_quadratic * omega ** 2, a_min=0.0, a_max=1.0)
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
        return spaces.Box(low, high, dtype=np.float32)


# reasonable reward function for hovering at a goal and not flying too high
def compute_reward_weighted(dynamics, goal, action, dt, crashed, time_remain, rew_coeff):
    ##################################################
    ## log to create a sharp peak at the goal
    dist = np.linalg.norm(goal - dynamics.pos)
    loss_pos = rew_coeff["pos"] * (np.log(dist + 0.1) + 0.1 * dist)
    # loss_pos = dist

    # dynamics_pos = dynamics.pos
    # print('dynamics.pos', dynamics.pos)

    ##################################################
    ## penalize altitude above this threshold
    # max_alt = 6.0
    # loss_alt = np.exp(2*(dynamics.pos[2] - max_alt))

    ##################################################
    # penalize amount of control effort
    loss_effort = rew_coeff["effort"] * np.linalg.norm(action)

    ##################################################
    ## loss velocity
    # dx = goal - dynamics.pos
    # dx = dx / (np.linalg.norm(dx) + EPS)
    
    ## normalized    
    # vel_direct = dynamics.vel / (np.linalg.norm(dynamics.vel) + EPS)
    # vel_magn = np.clip(np.linalg.norm(dynamics.vel),-1, 1)
    # vel_clipped = vel_magn * vel_direct 
    # vel_proj = np.dot(dx, vel_clipped)
    # loss_vel_proj = - rew_coeff["vel_proj"] * dist * vel_proj

    loss_vel_proj = 0. 

    ##################################################
    ## Loss orientation
    loss_orient = -rew_coeff["orient"] * dynamics.rot[2,2] 
    # Projection of the z-body axis to z-world axis
    # Negative, because the larger the projection the smaller the loss (i.e. the higher the reward)

    ##################################################
    ## Loss yaw
    loss_yaw = -rew_coeff["yaw"] * dynamics.rot[0,0]

    ##################################################
    ## Loss for constant uncontrolled rotation around vertical axis
    loss_spin_z  = rew_coeff["spin_z"]  * np.abs(dynamics.omega[2])
    loss_spin_xy = rew_coeff["spin_xy"] * np.linalg.norm(dynamics.omega[:2])

    ##################################################
    ## loss crash
    loss_crash = rew_coeff["crash"] * float(crashed)

    # reward = -dt * np.sum([loss_pos, loss_effort, loss_alt, loss_vel_proj, loss_crash])
    # rew_info = {'rew_crash': -loss_crash, 'rew_altitude': -loss_alt, 'rew_action': -loss_effort, 'rew_pos': -loss_pos, 'rew_vel_proj': -loss_vel_proj}

    reward = -dt * np.sum([
        loss_pos, 
        loss_effort, 
        loss_crash, 
        loss_vel_proj,
        loss_orient,
        loss_yaw,
        loss_spin_z,
        loss_spin_xy,
        ])
    

    rew_info = {
    "rew_main": -loss_pos,
    'rew_pos': -loss_pos, 
    'rew_action': -loss_effort, 
    'rew_crash': -loss_crash, 
    'rew_vel_proj': -loss_vel_proj,
    "rew_orient": -loss_orient,
    "rew_yaw": -loss_yaw,
    "rew_spin_z": -loss_spin_z,
    "rew_spin_xy": -loss_spin_xy,
    }

    # print('reward: ', reward, ' pos:', dynamics.pos, ' action', action)
    # print('pos', dynamics.pos)
    if np.isnan(reward) or not np.isfinite(reward):
        for key, value in locals().items():
            print('%s: %s \n' % (key, str(value)))
        raise ValueError('QuadEnv: reward is Nan')

    return reward, rew_info


####################################################################################################################################################################
## ENV
# Gym environment for quadrotor seeking the origin with no obstacles and full state observations.
# NOTES:
# - room size of the env and init state distribution are not the same !
#   It is done for the reason of having static (and preferably short) episode length, since for some distance it would be impossible to reach the goal
class QuadrotorEnv(gym.Env, Serializable):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self, dynamics_params="defaultquad", dynamics_change=None, 
                dynamics_randomize_every=None, dynamics_randomization_ratio=0., dynamics_randomization_ratio_params=None,
                raw_control=True, raw_control_zero_middle=True, dim_mode='3D', tf_control=False, sim_freq=200., sim_steps=2,
                obs_repr="xyz_vxyz_rot_omega", ep_time=4, obstacles_num=0, room_size=10, init_random_state=False, 
                rew_coeff=None, sense_noise=None, verbose=False, gravity=GRAV):
        np.seterr(under='ignore')
        """
        Args:
            dynamics_params: [str or dict] loading dynamics params by name or by providing a dictionary. 
                If "random": dynamics will be randomized completely (see sample_dyn_parameters() )
                If dynamics_randomize_every is None: it will be randomized only once at the beginning.
                One can randomize dynamics during the end of any episode using resample_dynamics()
                WARNING: randomization during an episode is not supported yet. Randomize ONLY before calling reset().
            dynamics_change: [dict] update to dynamics parameters relative to dynamics_params provided
            dynamics_randomize_every: [int] how often (trajectories) perform randomization
            dynamics_randomization_ratio: [float] randomization ratio relative to the nominal values of parameters
            dynamics_randomization_ratio_params: [dict] if a few dyn params require custom randomization ratios - provide them in this dict
            raw_control: [bool] use raw cantrol or the Mellinger controller as a default
            raw_control_zero_middle: [bool] meaning that control will be [-1 .. 1] rather than [0 .. 1]
            dim_mode: [str] Dimensionality of the env. Options: 1D(just a vertical stabilization), 2D(vertical plane), 3D(normal)
            tf_control: [bool] creates Mellinger controller using TensorFlow
            sim_freq (float): frequency of simulation
            sim_steps: [int] how many simulation steps for each control step
            obs_repr: [str] options: xyz_vxyz_rot_omega, xyz_vxyz_quat_omega
            ep_time: [float] episode time in simulated seconds. This parameter is used to compute env max time length in steps.
            obstacles_num: [int] number of obstacle in the env
            room_size: [int] env room size. Not the same as the initialization box to allow shorter episodes
            init_random_state: [bool] use random state initialization or horizontal initialization with 0 velocities
            rew_coeff: [dict] weights for different reward components (see compute_weighted_reward() function)
            sens_noise (dict or str): sensor noise parameters. If None - no noise. If "default" then the default params are loaded. Otherwise one can provide specific params.
        """
        ## ARGS
        self.init_random_state = init_random_state
        self.room_size = room_size
        self.obs_repr = obs_repr
        self.sim_steps = sim_steps
        self.dim_mode = dim_mode
        self.raw_control_zero_middle = raw_control_zero_middle
        self.tf_control = tf_control
        self.dynamics_randomize_every = dynamics_randomize_every
        self.verbose = verbose
        self.obstacles_num = obstacles_num
        self.raw_control = raw_control
        self.scene = None
        self.update_sense_noise(sense_noise=sense_noise)
        self.gravity = gravity
        
        ## PARAMS
        self.max_init_vel = 1.5 # m/s
        self.max_init_omega = 6 * np.pi #rad/s
        self.room_box = np.array([[-self.room_size, -self.room_size, 0], [self.room_size, self.room_size, self.room_size]])
        self.state_vector = getattr(self, "state_" + self.obs_repr)
        ## WARN: If you
        # size of the box from which initial position will be randomly sampled
        # if box_scale > 1.0 then it will also growevery episode
        self.box = 2.0
        self.box_scale = 1.0 #scale the initialbox by this factor eache episode

        ## Statistics vars
        self.traj_count = 0

        ###############################################################################
        ## DYNAMICS (and randomization)
        if dynamics_params == "random":
            self.dynamics_params_def = None
            self.dynamics_params = sample_dyn_parameters()
        else:
            ## Setting the quad dynamics params
            if isinstance(dynamics_params, str):
                self.dynamics_params_def = globals()[dynamics_params + "_params"]()
            elif isinstance(dynamics_params, dict):
                # This option is good when you only partially provide parameters of the model
                # For example if you are making some sort of a search, from the initial model
                self.dynamics_params_def = copy.deepcopy(dynamics_params)
            
            ## Now, updating if we are providing modifications
            if dynamics_change is not None:
                self.dynamics_params_def.update(dynamics_change)

            ## Setting randomization params
            if self.dynamics_randomize_every is not None:
                self.dyn_randomization_params = get_dyn_randomization_params(
                        quad_params=self.dynamics_params_def,
                        noise_ratio=dynamics_randomization_ratio,
                        noise_ratio_params=dynamics_randomization_ratio_params) 
                if self.verbose:
                    print("###############################################")
                    print("DYN RANDOMIZATION PARAMS:")
                    print_dic(self.dyn_randomization_params)
                    print("###############################################")
            self.dynamics_params = self.dynamics_params_def

        ## Updating dynamics
        dyn_upd_start_time = time.time()
        self.update_dynamics(dynamics_params=self.dynamics_params)
        print("QuadEnv: Dyn update time: ", time.time() - dyn_upd_start_time)

        ###############################################################################
        ## OBSERVATIONS
        self.observation_space = self.get_observation_space()

        ################################################################################
        ## DIMENSIONALITY
        if self.dim_mode =='1D':
            self.viewpoint = 'side'
        else:
            self.viewpoint = 'chase'

        ################################################################################
        ## EPISODE PARAMS
        # TODO get this from a wrapper
        self.ep_time = ep_time #In seconds
        self.dt = 1.0 / sim_freq
        self.metadata["video.frames_per_second"] = sim_freq / self.sim_steps
        self.ep_len = int(self.ep_time / (self.dt * self.sim_steps))
        self.tick = 0
        self.crashed = False
        self.control_freq = sim_freq / sim_steps

        #########################################
        ## REWARDS PARAMS
        self.rew_coeff = {
            "pos": 1, "effort": 0.01, "crash": 1, 
            "vel_proj": 0, 
            "orient": 1, "yaw": 0,
            "spin_z": 0.5, "spin_xy": 0.5}

        if rew_coeff is not None: 
            assert isinstance(rew_coeff, dict)
            assert set(rew_coeff.keys()).issubset(set(self.rew_coeff.keys()))
            self.rew_coeff.update(rew_coeff)

        #########################################
        ## RESET
        self._seed()
        self._reset()

        if self.spec is None:
            self.spec = gym_reg.EnvSpec(id='Quadrotor-v0', max_episode_steps=self.ep_len)
        
        # Always call Serializable constructor last
        Serializable.quick_init(self, locals())

    def save_dyn_params(self, filename):
        import yaml
        with open(filename, 'w') as yaml_file:
            def numpy_convert(key, item):
                return str(item)
            self.dynamics_params_converted = copy.deepcopy(self.dynamics_params)
            walk_dict(self.dynamics_params_converted, numpy_convert)
            yaml_file.write(yaml.dump(self.dynamics_params_converted, default_flow_style=False))

    def update_sense_noise(self, sense_noise):
        if isinstance(sense_noise, dict):
            self.sense_noise = SensorNoise(**sense_noise)
        elif isinstance(sense_noise, str):
            if sense_noise == "default":
                self.sense_noise = SensorNoise(bypass=False)
            else:
                ValueError("ERROR: QuadEnv: sense_noise parameter is of unknown type: " + str(sense_noise))
        elif sense_noise is None:
            self.sense_noise = SensorNoise(bypass=True)
        else:
            raise ValueError("ERROR: QuadEnv: sense_noise parameter is of unknown type: " + str(sense_noise))


    def update_dynamics(self, dynamics_params):
        ################################################################################
        ## DYNAMICS
        ## Then loading the dynamics
        self.dynamics_params = dynamics_params
        self.dynamics = QuadrotorDynamics(model_params=dynamics_params, 
                        dynamics_steps_num=self.sim_steps, room_box=self.room_box, dim_mode=self.dim_mode,
                        gravity=self.gravity)
        
        if self.verbose:
            print("#################################################")
            print("Dynamics params loaded:")
            print_dic(dynamics_params)
            print("#################################################")

        ################################################################################
        ## SCENE
        if self.obstacles_num > 0:
            self.obstacles = _random_obstacles(None, obstacles_num, self.room_size, self.dynamics.arm)
        else:
            self.obstacles = None

        ################################################################################
        ## CONTROL
        if self.raw_control:
            if self.dim_mode == '1D': # Z axis only
                self.controller = VerticalControl(self.dynamics, zero_action_middle=self.raw_control_zero_middle)
            elif self.dim_mode == '2D': # X and Z axes only
                self.controller = VertPlaneControl(self.dynamics, zero_action_middle=self.raw_control_zero_middle)
            elif self.dim_mode == '3D':
                self.controller = RawControl(self.dynamics, zero_action_middle=self.raw_control_zero_middle)
            else:
                raise ValueError('QuadEnv: Unknown dimensionality mode %s' % self.dim_mode)
        else:
            self.controller = NonlinearPositionController(self.dynamics, tf_control=self.tf_control)

        ################################################################################
        ## ACTIONS
        self.action_space = self.controller.action_space(self.dynamics)

        ################################################################################
        ## STATE VECTOR FUNCTION
        self.state_vector = getattr(self, "state_" + self.obs_repr)

    def state_xyz_vxyz_rot_omega(self):        
        pos, vel, rot, omega = self.sense_noise.add_noise(
            pos=self.dynamics.pos,
            vel=self.dynamics.vel,
            rot=self.dynamics.rot,
            omega=self.dynamics.omega,
            dt=self.dt
        )
        # return np.concatenate([pos - self.goal[:3], vel, rot.flatten(), omega, (pos[2],)])
        return np.concatenate([pos - self.goal[:3], vel, rot.flatten(), omega])

    def state_xyz_vxyz_quat_omega(self):
        self.quat = R2quat(self.dynamics.rot)
        pos, vel, quat, omega = self.sense_noise.add_noise(
            pos=self.dynamics.pos,
            vel=self.dynamics.vel,
            rot=self.quat,
            omega=self.dynamics.omega,
            dt=self.dt
        )
        return np.concatenate([pos - self.goal[:3], vel, quat, omega])

    def state_xyz_vxyz_euler_omega(self):
        self.euler = t3d.euler.mat2euler(self.dynamics.rot)
        pos, vel, quat, omega = self.sense_noise.add_noise(
            pos=self.dynamics.pos,
            vel=self.dynamics.vel,
            rot=self.euler,
            omega=self.dynamics.omega,
            dt=self.dt
        )       
        return np.concatenate([pos - self.goal[:3], vel, euler, omega])

    def get_observation_space(self):
        self.wall_offset = 0.3
        if self.obs_repr == "xyz_vxyz_rot_omega":
            ## Creating observation space
            # pos, vel, rot, rot vel
            self.obs_comp_sizes = [3, 3, 9, 3]
            self.obs_comp_names = ["xyz", "Vxyz", "R", "Omega"]
            obs_dim = np.sum(self.obs_comp_sizes)
            obs_high =  np.ones(obs_dim)
            obs_low  = -np.ones(obs_dim)
            
            # xyz room constraints
            obs_high[0:3] = self.room_box[1] - self.room_box[0] #i.e. full room size
            obs_low[0:3]  = -obs_high[0:3]

            # Vxyz
            obs_high[3:6] = self.dynamics.vxyz_max * obs_high[3:6]
            obs_low[3:6]  = self.dynamics.vxyz_max * obs_low[3:6] 

            # R
            # indx range: 6:15

            # Omega
            obs_high[15:18] = self.dynamics.omega_max * obs_high[15:18]
            obs_low[15:18]  = self.dynamics.omega_max * obs_low[15:18] 

            # z - distance to ground
            # obs_high[-1] = self.room_box[1][2] 
            # obs_low[-1] = self.room_box[0][2]

            # # xyz room constraints
            # obs_high[18:21] = self.room_box[1] - self.wall_offset
            # obs_low[18:21]  = self.room_box[0] + self.wall_offset

        elif self.obs_repr == "xyz_vxyz_euler_omega":
             ## Creating observation space
            # pos, vel, rot, rot vel
            self.obs_comp_sizes = [3, 3, 3, 3]
            self.obs_comp_names = ["xyz", "Vxyz", "euler", "Omega"]
            obs_dim = np.sum(self.obs_comp_sizes)
            obs_high =  np.ones(obs_dim)
            obs_low  = -np.ones(obs_dim)
            
            # xyz room constraints
            obs_high[0:3] = self.room_box[1] - self.room_box[0] #i.e. full room size
            obs_low[0:3]  = -obs_high[0:3]

            # Vxyz
            obs_high[3:6] = self.dynamics.vxyz_max * obs_high[3:6]
            obs_low[3:6]  = self.dynamics.vxyz_max * obs_low[3:6] 

            # Euler angles
            obs_high[6:9] = np.pi*obs_high[6:9] 
            obs_low[6:9]  = np.pi*obs_low[6:9]

            # Omega
            obs_high[9:12] = self.dynamics.omega_max * obs_high[9:12]
            obs_low[9:12]  = self.dynamics.omega_max * obs_low[9:12] 

            # z - distance to ground
            # obs_high[-1] = self.room_box[1][2] 
            # obs_low[-1] = self.room_box[0][2]           

        elif self.obs_repr == "state_xyz_vxyz_quat_omega":
             ## Creating observation space
            # pos, vel, rot, rot vel
            self.obs_comp_sizes = [3, 3, 4, 3]
            self.obs_comp_names = ["xyz", "Vxyz", "quat", "Omega"]
            obs_dim = np.sum(self.obs_comp_sizes)
            obs_high =  np.ones(obs_dim)
            obs_low  = -np.ones(obs_dim)

            # xyz room constraints
            obs_high[0:3] = self.room_box[1] - self.room_box[0] #i.e. full room size
            obs_low[0:3]  = -obs_high[0:3]

            # Vxyz
            obs_high[3:6] = self.dynamics.vxyz_max * obs_high[3:6]
            obs_low[3:6]  = self.dynamics.vxyz_max * obs_low[3:6] 

            # Quat
            # indx range: 6:9

            # Omega
            obs_high[9:12] = self.dynamics.omega_max * obs_high[9:12]
            obs_low[9:12]  = self.dynamics.omega_max * obs_low[9:12]           

            # z - distance to ground
            # obs_high[-1] = self.room_box[1][2] 
            # obs_low[-1] = self.room_box[0][2]    


        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)
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
        # self.scene.update_state(self.dynamics, self.goal)

        if self.obstacles is not None:
            self.crashed = self.obstacles.detect_collision(self.dynamics)
        else:
            self.crashed = self.dynamics.pos[2] <= self.dynamics.arm
        self.crashed = self.crashed or not np.array_equal(self.dynamics.pos,
                                                      np.clip(self.dynamics.pos,
                                                              a_min=self.room_box[0],
                                                              a_max=self.room_box[1]))

        self.time_remain = self.ep_len - self.tick
        try:
            reward, rew_info = compute_reward_weighted(self.dynamics, self.goal, action, self.dt, self.crashed, self.time_remain, 
                                rew_coeff=self.rew_coeff)
        except:
            import pdb; pdb.set_trace()
        self.tick += 1
        done = self.tick > self.ep_len #or self.crashed
        sv = self.state_vector()

        self.traj_count += int(done)
        # print('state', sv, 'goal', self.goal)
        # print('state', sv)
        # print('vel', sv[3], sv[4], sv[5])
        # print(sv, reward, done, rew_info)
        return sv, reward, done, {'rewards': rew_info}

    def resample_dynamics(self):
        """
        Allows manual dynamics resampling when needed.
        WARNING: 
            - Randomization dyring an episode is not supported
            - MUST call reset() after this function
        """
        if self.dynamics_params_def is None:
            self.dynamics_params = sample_dyn_parameters()
        else:
            ## Generating new params
            self.dynamics_params = perturb_dyn_parameters(
                params=copy.deepcopy(self.dynamics_params_def), 
                noise_params=self.dyn_randomization_params
                )
        ## Updating params
        self.update_dynamics(dynamics_params=self.dynamics_params)


    def _reset(self):
        ##############################################################
        ## DYNAMICS RANDOMIZATION AND UPDATE       
        if self.dynamics_randomize_every is not None and \
           (self.traj_count + 1) % (self.dynamics_randomize_every) == 0:
           
           self.resample_dynamics()

        ##############################################################
        ## VISUALIZATION
        if self.scene is None:
            self.scene = Quadrotor3DScene(model=self.dynamics.model,
                w=640, h=480, resizable=True, obstacles=self.obstacles, viewpoint=self.viewpoint)
        else:
            self.scene.update_model(self.dynamics.model)

        ##############################################################
        ## GOAL
        self.goal = np.array([0., 0., 2.])

        ## CURRICULUM (NOT REALLY NEEDED ANYMORE)
        # from 0.5 to 10 after 100k episodes (a form of curriculum)
        if self.box < 10:
            self.box = self.box * self.box_scale
        x, y, z = self.np_random.uniform(-self.box, self.box, size=(3,)) + self.goal
        
        if self.dim_mode == '1D':
            x, y = self.goal[0], self.goal[1]
        elif self.dim_mode == '2D':
            y = self.goal[1]
        #Since being near the groud means crash we have to start above
        if z < 0.25 : z = 0.25 
        pos = npa(x, y, z)

        ##############################################################
        ## INIT STATE
        ## Initializing rotation and velocities
        if self.init_random_state:
            if self.dim_mode == '1D':
                omega, rotation = npa(0, 0, 0), np.eye(3)
                vel = np.array([0., 0., self.max_init_vel * np.random.rand()])
            elif self.dim_mode == '2D':
                omega = npa(0, self.max_init_omega * np.random.rand(), 0)
                vel = self.max_init_vel * np.random.rand(3)
                vel[1] = 0.
                c, s, theta = np.cos(theta), np.sin(theta), np.pi * np.random.rand()
                rotation = np.array(((c, 0., -s), (0., 1., 0.), (s, 0., c)))
            else:
                # It already sets the state internally
                _, vel, rotation, omega = self.dynamics.random_state(box=self.room_size, vel_max=self.max_init_vel, omega_max=self.max_init_omega)
        else:
            ## INIT HORIZONTALLY WITH 0 VEL and OMEGA
            vel, omega = npa(0, 0, 0), npa(0, 0, 0)

            if self.dim_mode == '1D' or self.dim_mode == '2D':
                rotation = np.eye(3)
            else:
                # make sure we're sort of pointing towards goal (for mellinger controller)
                rotation = randyaw()
                while np.dot(rotation[:,0], to_xyhat(-pos)) < 0.5:
                    rotation = randyaw()
        
        # Setting the generated state
        # print("QuadEnv: init: pos/vel/rot/omega:", pos, vel, rotation, omega)
        self.init_state = [pos, vel, rotation, omega]
        self.dynamics.set_state(pos, vel, rotation, omega)

        # Resetting scene to reflect the state we have just set in dynamics
        self.scene.reset(self.goal, self.dynamics)
        # self.scene.update_state(self.dynamics)

        # Reseting some internal state (counters, etc)
        self.crashed = False
        self.tick = 0

        state = self.state_vector()
        return state

    def _render(self, mode='human', close=False):
        return self.scene.render_chase(dynamics=self.dynamics, goal=self.goal, mode=mode)
    
    def reset(self):
        return self._reset()

    def render(self, mode='human', **kwargs):
        return self._render(mode, **kwargs)
    
    def step(self, action):
        return self._step(action)


def test_rollout(quad, dyn_randomize_every=None, dyn_randomization_ratio=None, 
    render=True, traj_num=10, plot_step=None, plot_dyn_change=True,
    sense_noise=None):
    import tqdm
    #############################
    # Init plottting
    if plot_step is not None:
        fig = plt.figure(1)
        # ax = plt.subplot(111)
        plt.show(block=False)

    # render = True
    # plot_step = 50
    time_limit = 25
    render_each = 2
    rollouts_num = traj_num
    plot_obs = False

    env = QuadrotorEnv(dynamics_params=quad, raw_control=False, raw_control_zero_middle=True, sim_steps=4, 
        dynamics_randomize_every=dyn_randomize_every, dynamics_randomization_ratio=dyn_randomization_ratio,
        sense_noise=sense_noise)

    env.max_episode_steps = time_limit
    print('Reseting env ...')

    try:
        print('Observation space:', env.observation_space.low, env.observation_space.high)
        print('Action space:', env.action_space.low, env.action_space.high)
    except:
        print('Observation space:', env.observation_space.spaces[0].low, env.observation_space[0].spaces[0].high)
        print('Action space:', env.action_space[0].spaces[0].low, env.action_space[0].spaces[0].high)
    # input('Press any key to continue ...')

    ## Collected statistics for dynamics
    dyn_param_names = [
        "mass",
        "inertia",
        "thrust_to_weight",
        "torque_to_thrust",
        "thrust_noise_ratio",
        "vel_damp",
        "damp_omega_quadratic"
    ]

    dyn_param_stats = [[] for i in dyn_param_names]

    action = np.array([0.0, 0.5, 0.0, 0.5])
    rollouts_id = 0

    start_time = time.time()
    # while rollouts_id < rollouts_num:
    for rollouts_id in tqdm.tqdm(range(rollouts_num)):
        rollouts_id += 1
        s = env.reset()
        ## Diagnostics
        observations = []
        velocities = []

        ## Collecting dynamics params
        if plot_dyn_change:
            for par_i, par in enumerate(dyn_param_names):
                dyn_param_stats[par_i].append(np.array(getattr(env.dynamics, par)).flatten())
                # print(par, dyn_param_stats[par_i][-1])

        t = 0
        while True:
            if render and (t % render_each == 0): env.render()
            s, r, done, info = env.step(action)
            observations.append(s)
            print('Step: ', t, ' Obs:', env.dynamics.rot[0,0])

            if plot_step is not None and t % plot_step == 0:
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

    if plot_dyn_change:
        dyn_par_normvar = []
        dyn_par_means = []
        dyn_par_var = []
        plt.figure(2, figsize=(10, 10))
        for par_i, par in enumerate(dyn_param_stats):
            plt.subplot(3, 3, par_i+1)
            par = np.array(par)

            ## Compute stats
            # print(dyn_param_names[par_i], par)
            dyn_par_means.append(np.mean(par, axis=0))
            dyn_par_var.append(np.std(par, axis=0))
            dyn_par_normvar.append(dyn_par_var[-1] / dyn_par_means[-1])

            if par.shape[1] > 1:
                for vi in range(par.shape[1]):
                    plt.plot(par[:, vi])
            else:
                plt.plot(par)
            # plt.title(dyn_param_names[par_i] + "\n Normvar: %s" % str(dyn_par_normvar[-1]))
            plt.title(dyn_param_names[par_i])
            print(dyn_param_names[par_i], "NormVar: ", dyn_par_normvar[-1])
    
    print("##############################################################")
    print("Total time: ", time.time() - start_time )

    # print('Rollouts are done!')
    # plt.pause(2.0)
    # plt.waitforbuttonpress()
    if plot_step is not None or plot_dyn_change:
        plt.show(block=False)
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
    parser.add_argument(
        '-q',"--quad",
        default="defaultquad",
        help="Quadrotor model to use: \n" + 
            "- defaultquad \n" + 
            "- crazyflie \n" +
            "- random"
    )
    parser.add_argument(
        '-dre',"--dyn_randomize_every",
        type=int,
        help="How often (in terms of trajectories) to perform randomization"
    )
    parser.add_argument(
        '-drr',"--dyn_randomization_ratio",
        type=float,
        default=0.5,
        help="Randomization ratio for random sampling of dynamics parameters"
    )
    parser.add_argument(
        '-r',"--render",
        action="store_false",
        help="Use this flag to turn off rendering"
    )
    parser.add_argument(
        '-trj',"--traj_num",
        type=int,
        default=10,
        help="Number of trajectories to run"
    )
    parser.add_argument(
        '-plt',"--plot_step",
        type=int,
        help="Plot step"
    )
    parser.add_argument(
        '-pltdyn',"--plot_dyn_change",
        action="store_true",
        help="Plot the dynamics change from trajectory to trajectory?"
    )
    parser.add_argument(
        '-sn',"--sense_noise",
        action="store_true",
        help="Add sensor noise?"
    )
    args = parser.parse_args()

    if args.sense_noise:
        sense_noise="default"
    else:
        sense_noise=None

    if args.mode == 0:
        print('Running test rollout ...')
        test_rollout(
            quad=args.quad, 
            dyn_randomize_every=args.dyn_randomize_every,
            dyn_randomization_ratio=args.dyn_randomization_ratio,
            render=args.render,
            traj_num=args.traj_num,
            plot_step=args.plot_step,
            plot_dyn_change=args.plot_dyn_change,
            sense_noise=sense_noise
        )

if __name__ == '__main__':
    main(sys.argv)
