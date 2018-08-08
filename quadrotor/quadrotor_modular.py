#!/usr/bin/env python
"""
Quadrotor simulation for OpenAI Gym, with components reusable elsewhere.
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

import gym_art.quadrotor.rendering3d as r3d
from gym_art.quadrotor.quadrotor_control import *

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



def log_error(err_str, ):
    with open("/tmp/sac/errors.txt", "a") as myfile:
        myfile.write(err_str)
        # myfile.write('###############################################')


# simple simulation of quadrotor dynamics.
class QuadrotorDynamics(object):
    # mass unit: kilogram
    # arm_length unit: meter
    # inertia unit: kg * m^2, 3-element vector representing diagonal matrix
    # thrust_to_weight is the total, it will be divided among the 4 props
    # torque_to_thrust is ratio of torque produced by prop to thrust
    def __init__(self, mass, arm_length, inertia, thrust_to_weight=2.0, torque_to_thrust=0.05, dynamics_steps_num=1, room_box=None):
        assert np.isscalar(mass)
        assert np.isscalar(arm_length)
        assert inertia.shape == (3,)
        # This hack allows parametrize calling dynamics multiple times
        # without expensive for-loops
        self.step = getattr(self, 'step%d' % dynamics_steps_num)
        if room_box is None:
            self.room_box = np.array([[-10., -10., 0.], [10., 10., 10.]])
        else:
            self.room_box = np.array(room_box).copy()

        self.mass = mass
        self.arm = arm_length
        self.inertia = inertia
        self.thrust_to_weight = thrust_to_weight
        self.thrust = GRAV * mass * thrust_to_weight / 4.0
        self.torque = torque_to_thrust * self.thrust
        self.torque = torque_to_thrust * self.thrust
        scl = arm_length / norm([1.,1.,0.])
        self.prop_pos = scl * np.array([
            [1.,  1., -1., -1.],
            [1., -1., -1.,  1.],
            [0.,  0.,  0.,  0.]]).T # row-wise easier with np
        # unit: meters^2 ??? maybe wrong
        self.prop_crossproducts = np.cross(self.prop_pos, [0., 0., 1.])
        # 1 for props turning CCW, -1 for CW
        self.prop_ccw = np.array([1., -1., 1., -1.])
        self.since_last_svd = 0

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

    # thrust_cmds is motor thrusts given in normalized range [0, 1].
    # 1 represents the max possible thrust of the motor.
    def step1(self, thrust_cmds, dt):
        # import pdb; pdb.set_trace()
        # uncomment for debugging. they are slow
        #assert np.all(thrust_cmds >= 0)
        #assert np.all(thrust_cmds <= 1)

        # convert the motor commands to a force and moment on the body
        thrust_cmds = np.clip(thrust_cmds, 0.0, 1.0)
        thrusts = self.thrust * thrust_cmds
        torques = self.prop_crossproducts * thrusts[:,None]
        print('DYN: torques:', torques, self.prop_crossproducts)
        try:
            torques[:, 2] += self.torque * self.prop_ccw * thrust_cmds
        except Exception as e:
            print('actions: ', thrust_cmds)
            log_error('##########################################################')
            for key, value in locals().items():
                log_error('%s: %s \n' % (key, str(value)))
                print('%s: %s \n' % (key, str(value)))
            raise ValueError("QuadrotorEnv ERROR: SVD did not converge: " + str(e))
        # torques[:,2] += self.torque * self.prop_ccw * thrust_cmds

        thrust_torque = np.sum(torques, axis=0)

        ## Dampling torque
        # damping_torque = - 0.3 * self.omega * np.fabs(self.omega)
        damping_torque = 0.0
        # print('DYN: thrust torque: ', thrust_torque, 'damp_torque', damping_torque, 'omega', self.omega)
        torque =  thrust_torque + damping_torque
        thrust = npa(0,0,np.sum(thrusts))
        # print('thrus_cmds:', thrust_cmds, ' thrusts', thrusts, ' prop_cross', self.prop_crossproducts)

        # TODO add noise

        vel_damp = 0.999
        # omega_damp = 0.999 # This is only for linear dampling of angular velocity. Currently use quadratic damping

        # rotational dynamics
        omega_dot = ((1.0 / self.inertia) *
            (cross(-self.omega, self.inertia * self.omega) + torque))

        ## Linear damping
        # self.omega = omega_damp * self.omega + dt * omega_dot

        ## Quadratic damping
        # 0.03 corresponds to roughly 1 revolution per sec
        omega_damp_quadratic = np.clip(0.015 * self.omega ** 2, a_min=0.0, a_max=1.0)
        self.omega = self.omega + (1.0 - omega_damp_quadratic) * dt * omega_dot

        ## When use square damping on torques - use simple integration
        # self.omega += dt * omega_dot

        omega_vec = np.matmul(self.rot, self.omega)
        x, y, z = omega_vec
        omega_mat_deriv = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])

        dRdt = np.matmul(omega_mat_deriv, self.rot)
        self.rot += dt * dRdt

        # occasionally orthogonalize the rotation matrix
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

        # translational dynamics
        # Room constraints
        mask = np.logical_or(self.pos <= self.room_box[0], self.pos >= self.room_box[1])
        acc = [0, 0, -GRAV] + (1.0 / self.mass) * np.matmul(self.rot, thrust)
        # acc[mask] = 0.
        self.acc = acc
        self.vel = vel_damp * self.vel + dt * acc
        # self.vel[mask] = 0.
        self.pos = self.pos + dt * self.vel
        # self.pos = np.clip(self.pos, a_min=self.room_box[0], a_max=self.room_box[1])

        self.accelerometer = np.matmul(self.rot.T, acc + [0, 0, GRAV])

        # if np.any(np.isnan(self.rot)):
        #     log_error('##########################################################')
        #     for key, value in locals().items():
        #         log_error('%s: %s \n' %(key, str(value)))


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


def default_dynamics(sim_steps, room_box):
    # similar to AscTec Hummingbird
    # TODO: dictionary of dynamics of real quadrotors
    mass = 0.5
    arm_length = 0.33 / 2.0
    inertia = mass * npa(0.01, 0.01, 0.02)
    thrust_to_weight = 2.0
    return QuadrotorDynamics(mass, arm_length, inertia,
        thrust_to_weight=thrust_to_weight, dynamics_steps_num=sim_steps, room_box=room_box)


# reasonable reward function for hovering at a goal and not flying too high
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


# for visualization.
# a rough attempt at a reasonable third-person camera
# that looks "over the quadrotor's shoulder" from behind
class ChaseCamera(object):
    def __init__(self):
        self.view_dist = 4

    def reset(self, goal, pos, vel):
        self.goal = goal
        self.pos_smooth = pos
        self.vel_smooth = vel
        self.right_smooth, _ = normalize(cross(vel, npa(0, 0, 1)))

    def step(self, pos, vel):
        # lowpass filter
        ap = 0.6
        av = 0.999
        ar = 0.9
        self.pos_smooth = ap * self.pos_smooth + (1 - ap) * pos
        self.vel_smooth = av * self.vel_smooth + (1 - av) * vel

        veln, n = normalize(self.vel_smooth)
        up = npa(0, 0, 1)
        ideal_vel, _ = normalize(self.goal - self.pos_smooth)
        if True or np.abs(veln[2]) > 0.95 or n < 0.01 or np.dot(veln, ideal_vel) < 0.7:
            # look towards goal even though we are not heading there
            right, _ = normalize(cross(ideal_vel, up))
        else:
            right, _ = normalize(cross(veln, up))
        self.right_smooth = ar * self.right_smooth + (1 - ar) * right

    # return eye, center, up suitable for gluLookAt
    def look_at(self):
        up = npa(0, 0, 1)
        back, _ = normalize(cross(self.right_smooth, up))
        to_eye, _ = normalize(0.9 * back + 0.3 * self.right_smooth)
        eye = self.pos_smooth + self.view_dist * (to_eye + 0.3 * up)
        center = self.pos_smooth
        return eye, center, up


# for visualization.
# In case we have vertical control only we use a side view camera
class SideCamera(object):
    def __init__(self):
        self.view_dist = 4

    def reset(self, goal, pos, vel):
        self.goal = goal
        self.pos_smooth = pos
        self.vel_smooth = vel
        self.right_smooth, _ = normalize(cross(vel, npa(0, 0, 1)))

    def step(self, pos, vel):
        # lowpass filter
        ap = 0.6
        av = 0.999
        ar = 0.9
        self.pos_smooth = ap * self.pos_smooth + (1 - ap) * pos
        self.vel_smooth = av * self.vel_smooth + (1 - av) * vel

        veln, n = normalize(self.vel_smooth)
        up = npa(0, 0, 1)
        ideal_vel, _ = normalize(self.goal - self.pos_smooth)
        if True or np.abs(veln[2]) > 0.95 or n < 0.01 or np.dot(veln, ideal_vel) < 0.7:
            # look towards goal even though we are not heading there
            right, _ = normalize(cross(ideal_vel, up))
        else:
            right, _ = normalize(cross(veln, up))
        self.right_smooth = ar * self.right_smooth + (1 - ar) * right

    # return eye, center, up suitable for gluLookAt
    def look_at(self):
        up = npa(0, 0, 1)
        back, _ = normalize(cross(self.right_smooth, up))
        to_eye, _ = normalize(0.9 * back + 0.3 * self.right_smooth)
        # eye = self.pos_smooth + self.view_dist * (to_eye + 0.3 * up)
        eye = self.pos_smooth + self.view_dist * np.array([0, 1, 0])
        center = self.pos_smooth
        return eye, center, up


# determine where to put the obstacles such that no two obstacles intersect
# and compute the list of obstacles to collision check at each 2d tile.
def _place_obstacles(np_random, N, box, radius_range, our_radius, tries=5):

    t = np.linspace(0, box, TILES+1)[:-1]
    scale = box / float(TILES)
    x, y = np.meshgrid(t, t)
    pts = np.zeros((N, 2))
    dist = x + np.inf

    radii = np_random.uniform(*radius_range, size=N)
    radii = np.sort(radii)[::-1]
    test_list = [[] for i in range(TILES**2)]

    for i in range(N):
        rad = radii[i]
        ok = np.where(dist.flat > rad)[0]
        if len(ok) == 0:
            if tries == 1:
                print("Warning: only able to place {}/{} obstacles. "
                    "Increase box, decrease radius, or decrease N.")
                return pts[:i,:], radii[:i]
            else:
                return _place_obstacles(N, box, radius_range, tries-1)
        pt = np.unravel_index(np_random.choice(ok), dist.shape)
        pt = scale * np.array(pt)
        d = np.sqrt((x - pt[1])**2 + (y - pt[0])**2) - rad
        # big slop factor for tile size, off-by-one errors, etc
        for ind1d in np.where(d.flat <= 2*our_radius + scale)[0]:
            test_list[ind1d].append(i)
        dist = np.minimum(dist, d)
        pts[i,:] = pt - box/2.0

    # very coarse to allow for binning bugs
    test_list = np.array(test_list).reshape((TILES, TILES))
    #amt_free = sum(len(a) == 0 for a in test_list.flat) / float(test_list.size)
    #print(amt_free * 100, "pct free space")
    return pts, radii, test_list


# generate N obstacles w/ randomized primitive, size, color, TODO texture
# arena: boundaries of world in xy plane
# our_radius: quadrotor's radius
def _random_obstacles(np_random, N, arena, our_radius):
    arena = float(arena)
    # all primitives should be tightly bound by unit circle in xy plane
    boxside = np.sqrt(2)
    box = r3d.box(boxside, boxside, boxside)
    sphere = r3d.sphere(radius=1.0, facets=16)
    cylinder = r3d.cylinder(radius=1.0, height=2.0, sections=32)
    # TODO cone-sphere collision
    #cone = r3d.cone(radius=0.5, height=1.0, sections=32)
    primitives = [box, sphere, cylinder]

    bodies = []
    max_radius = 2.0
    positions, radii, test_list = _place_obstacles(
        np_random, N, arena, (0.5, max_radius), our_radius)
    for i in range(N):
        primitive = np_random.choice(primitives)
        tex_type = r3d.random_textype()
        tex_dark = 0.5 * np_random.uniform()
        tex_light = 0.5 * np_random.uniform() + 0.5
        color = 0.5 * np_random.uniform(size=3)
        heightscl = np.random.uniform(0.5, 2.0)
        height = heightscl * 2.0 * radii[i]
        z = (0 if primitive is cylinder else
            (height/2.0 if primitive is sphere else
            (height*boxside/4.0 if primitive is box
            else np.nan)))
        translation = np.append(positions[i,:], z)
        matrix = np.matmul(r3d.translate(translation), r3d.scale(radii[i]))
        matrix = np.matmul(matrix, np.diag([1, 1, heightscl, 1]))
        body = r3d.Transform(matrix,
            #r3d.ProceduralTexture(tex_type, (tex_dark, tex_light), primitive))
                r3d.Color(color, primitive))
        bodies.append(body)

    return ObstacleMap(arena, bodies, test_list)


# main class for non-visual aspects of the obstacle map.
class ObstacleMap(object):
    def __init__(self, box, bodies, test_lists):
        self.box = box
        self.bodies = bodies
        self.test = test_lists

    def detect_collision(self, dynamics):
        pos = dynamics.pos
        if pos[2] <= dynamics.arm:
            print("collided with terrain")
            return True
        r, c = self.coord2tile(*dynamics.pos[:2])
        if r < 0 or c < 0 or r >= TILES or c >= TILES:
            print("collided with wall")
            return True
        if self.test is not None:
            radius = dynamics.arm + 0.1
            return any(self.bodies[k].collide_sphere(pos, radius)
                for k in self.test[r,c])
        return False

    def sample_start(self, np_random):
        pad = 4
        band = TILES // 8
        return self.sample_freespace((pad, pad + band), np_random)

    def sample_goal(self, np_random):
        pad = 4
        band = TILES // 8
        return self.sample_freespace((-(pad + band), -pad), np_random)

    def sample_freespace(self, rowrange, np_random):
        rfree, cfree = np.where(np.vectorize(lambda t: len(t) == 0)(
            self.test[rowrange[0]:rowrange[1],:]))
        choice = np_random.choice(len(rfree))
        r, c = rfree[choice], cfree[choice]
        r += rowrange[0]
        x, y = self.tile2coord(r, c)
        z = np_random.uniform(1.0, 3.0)
        return np.array([x, y, z])

    def tile2coord(self, r, c):
        #TODO consider moving origin to corner of world
        scale = self.box / float(TILES)
        return scale * np.array([r,c]) - self.box / 2.0

    def coord2tile(self, x, y):
        scale = float(TILES) / self.box
        return np.int32(scale * (np.array([x,y]) + self.box / 2.0))


# using our rendering3d.py to draw the scene in 3D.
# this class deals both with map and mapless cases.
class Quadrotor3DScene(object):
    def __init__(self, np_random, quad_arm, w, h,
        obstacles=True, visible=True, resizable=True, goal_diameter=None, viewpoint='chase'):

        self.window_target = r3d.WindowTarget(w, h, resizable=resizable)
        self.obs_target = r3d.FBOTarget(64, 64)
        self.cam1p = r3d.Camera(fov=90.0)
        self.cam3p = r3d.Camera(fov=45.0)

        self.viepoint = viewpoint
        if self.viepoint == 'chase':
            self.chase_cam = ChaseCamera()
        elif self.viepoint == 'side':
            self.chase_cam = SideCamera()
        self.world_box = 40.0

        diameter = 2 * quad_arm
        if goal_diameter:
            self.goal_diameter = goal_diameter
        else:
            self.goal_diameter = diameter
        self.quad_transform = self._quadrotor_3dmodel(diameter)

        self.shadow_transform = r3d.transform_and_color(
            np.eye(4), (0, 0, 0, 0.4), r3d.circle(0.75*diameter, 32))

        # TODO make floor size or walls to indicate world_box
        floor = r3d.ProceduralTexture(r3d.random_textype(), (0.15, 0.25),
            r3d.rect((1000, 1000), (0, 100), (0, 100)))

        self.goal_transform = r3d.transform_and_color(np.eye(4),
            (0.85, 0.55, 0), r3d.sphere(self.goal_diameter/2, 18))

        self.map = None
        bodies = [r3d.BackToFront([floor, self.shadow_transform]),
            self.goal_transform, self.quad_transform]

        if obstacles:
            N = 20
            self.map = _random_obstacles(np_random, N, self.world_box, quad_arm)
            bodies += self.map.bodies

        world = r3d.World(bodies)
        batch = r3d.Batch()
        world.build(batch)

        self.scene = r3d.Scene(batches=[batch], bgcolor=(0,0,0))
        self.scene.initialize()

    def _quadrotor_3dmodel(self, diam):
        r = diam / 2
        prop_r = 0.3 * diam
        prop_h = prop_r / 15.0

        # "X" propeller configuration, start fwd left, go clockwise
        rr = r * np.sqrt(2)/2
        deltas = ((rr, rr, 0), (rr, -rr, 0), (-rr, -rr, 0), (-rr, rr, 0))
        colors = ((1,0,0), (1,0,0), (0,1,0), (0,1,0))
        def disc(translation, color):
            color = 0.5 * np.array(list(color)) + 0.2
            disc = r3d.transform_and_color(r3d.translate(translation), color,
                r3d.cylinder(prop_r, prop_h, 32))
            return disc
        props = [disc(d, c) for d, c in zip(deltas, colors)]

        arm_thicc = diam / 20.0
        arm_color = (0.6, 0.6, 0.6)
        arms = r3d.transform_and_color(
            np.matmul(r3d.translate((0, 0, -arm_thicc)), r3d.rotz(np.pi / 4)), arm_color,
            [r3d.box(diam/10, diam, arm_thicc), r3d.box(diam, diam/10, arm_thicc)])

        arrow = r3d.Color((0.2, 0.3, 0.9), r3d.arrow(0.12*prop_r, 2.5*prop_r, 16))

        bodies = props + [arms, arrow]
        self.have_state = False
        return r3d.Transform(np.eye(4), bodies)

    # TODO allow resampling obstacles?
    def reset(self, goal, dynamics):
        self.goal_transform.set_transform(r3d.translate(goal[0:3]))
        self.chase_cam.reset(goal[0:3], dynamics.pos, dynamics.vel)
        self.update_state(dynamics)

    def update_state(self, dynamics):
        self.have_state = True
        self.fpv_lookat = dynamics.look_at()
        self.chase_cam.step(dynamics.pos, dynamics.vel)

        matrix = r3d.trans_and_rot(dynamics.pos, dynamics.rot)
        self.quad_transform.set_transform_nocollide(matrix)

        shadow_pos = 0 + dynamics.pos
        shadow_pos[2] = 0.001 # avoid z-fighting
        matrix = r3d.translate(shadow_pos)
        self.shadow_transform.set_transform_nocollide(matrix)

        if self.map is not None:
            collided = self.map.detect_collision(dynamics)
        else:
            collided = dynamics.pos[2] <= dynamics.arm
        return collided

    def render_chase(self):
        assert self.have_state
        self.cam3p.look_at(*self.chase_cam.look_at())
        #self.cam3p.look_at(*self.fpv_lookat)
        r3d.draw(self.scene, self.cam3p, self.window_target)

    def render_obs(self):
        assert self.have_state
        self.cam1p.look_at(*self.fpv_lookat)
        r3d.draw(self.scene, self.cam1p, self.obs_target)
        return self.obs_target.read()


# Gym environment for quadrotor seeking the origin
# with no obstacles and full state observations
class QuadrotorEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self, raw_control=True, dim_mode='3D', tf_control=True, sim_steps=4):
        np.seterr(under='ignore')
        self.room_box = np.array([[-10, -10, 0], [10, 10, 10]])
        self.dynamics = default_dynamics(sim_steps, room_box=self.room_box)
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
                self.controller = VerticalControl(self.dynamics)
            elif self.dim_mode == '2D':
                self.controller = VertPlaneControl(self.dynamics)
            elif self.dim_mode == '3D':
                self.controller = RawControl(self.dynamics)
            else:
                raise ValueError('QuadEnv: Unknown dimensionality mode %s' % self.dim_mode)
        else:
            self.controller = NonlinearPositionController(self.dynamics, tf_control=tf_control)

        self.action_space = self.controller.action_space(self.dynamics)

        # pos, vel, rot, rot vel
        obs_dim = 3 + 3 + 9 + 3 + 3 # xyz, Vxyz, R, Omega, goal_xyz
        # TODO tighter bounds on some variables
        obs_high = 100 * np.ones(obs_dim)
        # rotation mtx guaranteed to be orthogonal
        obs_high[6:6+9] = 1
        self.observation_space = spaces.Box(-obs_high, obs_high)

        # TODO get this from a wrapper
        self.ep_len = 300
        self.tick = 0
        # self.dt = 1.0 / 50.0
        self.dt = 1.0 / 100.0
        self.crashed = False

        self._seed()

        # size of the box from which initial position will be randomly sampled
        # if box_scale > 1.0 then it will also growevery episode
        self.box = 2.0
        self.box_scale = 1.0 #scale the initialbox by this factor eache episode

        self._reset()

        if self.spec is None:
            self.spec = gym_reg.EnvSpec(id='Quadrotor-v0', max_episode_steps=self.ep_len)


    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        # print('actions: ', action)
        if not self.crashed:
            # print('goal: ', self.goal, 'goal_type: ', type(self.goal))
            self.controller.step_func(dynamics=self.dynamics,
                                    action=action,
                                    goal=self.goal,
                                    dt=self.dt,
                                    observation=np.expand_dims(self.dynamics.state_vector(), axis=0))
            # self.oracle.step(self.dynamics, self.goal, self.dt)
            self.crashed = self.scene.update_state(self.dynamics)
            self.crashed = self.crashed or not np.array_equal(self.dynamics.pos,
                                                          np.clip(self.dynamics.pos,
                                                                  a_min=self.room_box[0],
                                                                  a_max=self.room_box[1]))
        self.time_remain = self.ep_len - self.tick
        reward, rew_info = goal_seeking_reward(self.dynamics, self.goal, action, self.dt, self.crashed, self.time_remain)
        self.tick += 1
        done = self.tick > self.ep_len #or self.crashed
        sv = self.dynamics.state_vector()
        sv = np.append(sv, self.goal[:3])

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

        state = self.dynamics.state_vector()
        #That helps to avoid including goals xyz into the observation space
        state = np.append(state, self.goal[:3])
        # print('state', state)
        return state

    def _render(self, mode='human', close=False):
        self.scene.render_chase()



# Gym environment for quadrotor seeking a given goal
# with obstacles and vision + IMU observations
class QuadrotorVisionEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self):
        np.seterr(under='ignore')
        self.dynamics = default_dynamics()
        #self.controller = ShiftedMotorControl(self.dynamics)
        self.controller = OmegaThrustControl(self.dynamics)
        self.action_space = self.controller.action_space(self.dynamics)
        self.scene = None
        self.crashed = False
        self.oracle = NonlinearPositionController(self.dynamics)

        seq_len = 4
        img_w, img_h = 64, 64
        img_space = spaces.Box(-1, 1, (img_h, img_w, seq_len))
        imu_space = spaces.Box(-100, 100, (6, seq_len))
        # vector from us to goal projected onto world plane and rotated into
        # our "looking forward" coordinates, and clamped to a maximal length
        dir_space = spaces.Box(-4, 4, (2, seq_len))
        self.observation_space = spaces.Tuple([img_space, imu_space, dir_space])
        self.img_buf = np.zeros((img_w, img_h, seq_len))
        self.imu_buf = np.zeros((6, seq_len))
        self.dir_buf = np.zeros((2, seq_len))

        # TODO get this from a wrapper
        self.ep_len = 500
        self.tick = 0
        self.dt = 1.0 / 50.0

        self._seed()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        if not self.crashed:
            #self.controller.step(self.dynamics, action, self.dt)
            # print("oracle step")
            self.oracle.step(self.dynamics, self.goal, self.dt)
            self.crashed = self.scene.update_state(self.dynamics)
        reward, rew_info = goal_seeking_reward(self.dynamics, self.goal, action, self.dt, self.crashed)
        self.tick += 1
        done = self.crashed or (self.tick > self.ep_len)

        rgb = self.scene.render_obs()
        # for debugging:
        #rgb = np.flip(rgb, axis=0)
        #plt.imshow(rgb)
        #plt.show()

        grey = (2.0 / 255.0) * np.mean(rgb, axis=2) - 1.0
        #self.img_buf = np.roll(self.img_buf, -1, axis=2)
        self.img_buf[:,:,:-1] = self.img_buf[:,:,1:]
        self.img_buf[:,:,-1] = grey

        imu = np.concatenate([self.dynamics.omega, self.dynamics.accelerometer])
        #self.imu_buf = np.roll(self.imu_buf, -1, axis=1)
        self.imu_buf[:,:-1] = self.imu_buf[:,1:]
        self.imu_buf[:,-1] = imu

        # heading measurement - simplified, #95489c has a more nuanced version
        our_gps = self.dynamics.pos[:2]
        goal_gps = self.goal[:2]
        dir = clamp_norm(goal_gps - our_gps, 4.0)
        #self.dir_buf = np.roll(self.dir_buf, -1, axis=1)
        self.dir_buf[:,:-1] = self.dir_buf[:,1:]
        self.dir_buf[:,-1] = dir

        return (self.img_buf, self.imu_buf, self.dir_buf), reward, done, {'rewards': rew_info}

    def _reset(self):
        if self.scene is None:
            self.scene = Quadrotor3DScene(self.np_random, self.dynamics.arm,
                640, 480, resizable=True)

        self.goal = self.scene.map.sample_goal(self.np_random)
        pos = self.scene.map.sample_start(self.np_random)
        vel = omega = npa(0, 0, 0)
        # for debugging collisions w/ no policy:
        #vel = self.np_random.uniform(-20, 20, size=3)
        #vel[2] = 0

        # make us point towards the goal
        xb = to_xyhat(self.goal - pos)
        zb = npa(0, 0, 1)
        yb = cross(zb, xb)
        rotation = np.column_stack([xb, yb, zb])
        self.dynamics.set_state(pos, vel, rotation, omega)
        self.crashed = False

        self.scene.reset(self.goal, self.dynamics)
        collided = self.scene.update_state(self.dynamics)
        assert not collided

        # fill the buffers with copies of initial state
        w, h, seq_len = self.img_buf.shape
        rgb = self.scene.render_obs()
        grey = (2.0 / 255.0) * np.mean(rgb, axis=2) - 1.0
        self.img_buf = np.tile(grey[:,:,None], (1,1,seq_len))
        imu = np.concatenate([self.dynamics.omega, self.dynamics.accelerometer])
        self.imu_buf = np.tile(imu[:,None], (1,seq_len))

        self.tick = 0
        return (self.img_buf, self.imu_buf)

    def _render(self, mode='human', close=False):
        self.scene.render_chase()


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

    env = QuadrotorEnv(raw_control=False)

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
