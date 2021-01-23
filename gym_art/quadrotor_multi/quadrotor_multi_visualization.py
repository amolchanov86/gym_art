import copy
import numpy as np

import gym_art.quadrotor_multi.rendering3d as r3d

from gym_art.quadrotor_multi.quadrotor_visualization import ChaseCamera, SideCamera, quadrotor_simple_3dmodel, \
    quadrotor_3dmodel
from gym_art.quadrotor_multi.params import quad_color
from gym_art.quadrotor_multi.quad_utils import *
from gym_art.quadrotor_multi.quad_utils import calculate_collision_matrix
from scipy import spatial
from pyglet.window import key

# Global Camera
class GlobalCamera(object):
    def __init__(self, view_dist=2.0):
        self.radius = view_dist
        self.theta = np.pi / 2
        self.phi = 0.0
        self.center = np.array([0., 0., 2.])
        # center, phi, theta
        self.center_list = [[[5, 5, 10.0], 0.93, 1.14],
                            [[5, -5, 10.0], -0.78, 1.0],
                            [[-5., -5., 10.0], -2.2, 0.87],
                            [[-5, 5, 10.0], 2.4, 0.87],
                            ]
        self.center_list_id = 0

    def reset(self, view_dist=2.0, center=np.array([0., 0., 2.])):
        self.center = center

    def step(self, center=np.array([0., 0., 2.])):
        pass

    def look_at(self):
        up = npa(0, 0, 1)
        center = self.center  # pattern center
        eye = center + self.radius * np.array([np.sin(self.theta) * np.cos(self.phi), np.sin(self.theta) * np.sin(self.phi), np.cos(self.theta)])
        return eye, center, up


class Quadrotor3DSceneMulti:
    def __init__(
            self, w, h,
            quad_arm=None, models=None, obstacles=None, visible=True, resizable=True, goal_diameter=None,
            viewpoint='chase', obs_hw=None, obstacle_mode='no_obstacles', room_dims=(10, 10, 10), num_agents=8
    ):
        if obs_hw is None:
            obs_hw = [64, 64]

        self.window_target = None
        self.window_w, self.window_h = w, h
        self.resizable = resizable
        self.viepoint = viewpoint
        self.obs_hw = copy.deepcopy(obs_hw)
        self.visible = visible

        self.quad_arm = quad_arm
        self.obstacles = obstacles
        self.obstacle_mode = obstacle_mode
        self.models = models
        self.room_dims = room_dims

        self.quad_transforms, self.shadow_transforms, self.goal_transforms = [], [], []

        if goal_diameter:
            self.goal_forced_diameter = goal_diameter
        else:
            self.goal_forced_diameter = None

        self.diameter = self.goal_diameter = -1
        self.update_goal_diameter()

        if self.viepoint == 'chase':
            self.chase_cam = ChaseCamera(view_dist=self.diameter * 15)
        elif self.viepoint == 'side':
            self.chase_cam = SideCamera(view_dist=self.diameter * 15)
        elif self.viepoint == 'global':
            self.chase_cam = GlobalCamera(view_dist=2.5)

        self.fpv_lookat = None

        self.scene = None
        self.window_target = None
        self.obs_target = None
        self.video_target = None

        # Save parameters to help transfer from global camera to local camera
        self.goals = None
        self.dynamics = None
        self.num_agents = num_agents
        self.camera_drone_index = 0
        self.camera_rot_step_size = np.pi / 90
        self.camera_zoom_step_size = 0.02

    def update_goal_diameter(self):
        if self.quad_arm is not None:
            self.diameter = 2 * self.quad_arm
        else:
            self.diameter = 2 * np.linalg.norm(self.models[0].params['motor_pos']['xyz'][:2])

        if self.goal_forced_diameter:
            self.goal_diameter = self.goal_forced_diameter
        else:
            self.goal_diameter = self.diameter

    def update_env(self, room_dims):
        self.room_dims = room_dims

    def _make_scene(self):
        self.cam1p = r3d.Camera(fov=90.0)
        self.cam3p = r3d.Camera(fov=45.0)

        self.quad_transforms, self.shadow_transforms, self.goal_transforms, self.collision_transforms, self.obstacle_transforms = [], [], [], [], []

        for i, model in enumerate(self.models):
            if model is not None:
                quad_transform = quadrotor_3dmodel(model, quad_id=i)
            else:
                quad_transform = quadrotor_simple_3dmodel(self.diameter)
            self.quad_transforms.append(quad_transform)

            self.shadow_transforms.append(
                r3d.transform_and_color(np.eye(4), (0, 0, 0, 0.4), r3d.circle(0.75 * self.diameter, 32))
            )
            self.collision_transforms.append(
                r3d.transform_and_color(np.eye(4), (0, 0, 0, 0.0), r3d.sphere(0.75 * self.diameter, 32))
            )

        # TODO make floor size or walls to indicate world_box
        floor = r3d.ProceduralTexture(r3d.random_textype(), (0.15, 0.25),
                                      r3d.rect((100, 100), (0, 100), (0, 100)))
        self.update_goal_diameter()
        self.chase_cam.view_dist = self.diameter * 15

        self.create_goals()

        bodies = [r3d.BackToFront([floor, st]) for st in self.shadow_transforms]
        bodies.extend(self.goal_transforms)
        bodies.extend(self.quad_transforms)
        # visualize walls of the room if True
        if self.visible:
            room = r3d.ProceduralTexture(r3d.random_textype(), (0.15, 0.25), r3d.envBox(*self.room_dims))
            bodies.append(room)

        if self.obstacle_mode != 'no_obstacles':
            self.create_obstacles()
            bodies.extend(self.obstacle_transforms)

        world = r3d.World(bodies)
        batch = r3d.Batch()
        world.build(batch)
        self.scene = r3d.Scene(batches=[batch], bgcolor=(0, 0, 0))
        self.scene.initialize()

        # Collision spheres have to be added in the ending after everything has been rendered, as it transparent
        bodies = []
        bodies.extend(self.collision_transforms)
        world = r3d.World(bodies)
        batch = r3d.Batch()
        world.build(batch)
        self.scene.batches.extend([batch])

    def create_obstacles(self):
        for item in self.obstacles.obstacles:
            color = quad_color[14]
            if item.type == 'cube':
                obstacle_transform = r3d.transform_and_color(np.eye(4), color, r3d.box(item.size, item.size, item.size))
            elif item.type == 'sphere':
                obstacle_transform = r3d.transform_and_color(np.eye(4), color, r3d.sphere(item.size / 2, 18))
            else:
                raise NotImplementedError()

            self.obstacle_transforms.append(obstacle_transform)

    def update_obstacles(self, obstacles):
        for i, g in enumerate(obstacles.obstacles):
            self.obstacle_transforms[i].set_transform(r3d.translate(g.pos))

    def create_goals(self):
        for i in range(len(self.models)):
            color = quad_color[i % len(quad_color)]
            goal_transform = r3d.transform_and_color(np.eye(4), color, r3d.sphere(self.goal_diameter / 2, 18))
            self.goal_transforms.append(goal_transform)

    def update_goals(self, goals):
        for i, g in enumerate(goals):
            self.goal_transforms[i].set_transform(r3d.translate(g[0:3]))

    def update_models(self, models):
        self.models = models

        if self.video_target is not None:
            self.video_target.finish()
            self.video_target = None
        if self.obs_target is not None:
            self.obs_target.finish()
            self.obs_target = None
        if self.window_target:
            self._make_scene()

    def reset(self, goals, dynamics, obstacles, collisions):
        self.goals = goals
        self.dynamics = dynamics

        if self.viepoint == 'global':
            goal = np.mean(goals, axis=0)
            self.chase_cam.reset(view_dist=2.5, center=goal)
        else:
            goal = goals[self.camera_drone_index]  # TODO: make a camera that can look at all drones
            self.chase_cam.reset(goal[0:3], dynamics[self.camera_drone_index].pos, dynamics[self.camera_drone_index].vel)


        self.update_state(dynamics, goals, obstacles, collisions)

    def update_state(self, all_dynamics, goals, obstacles, collisions):
        if self.scene:
            if self.viepoint == 'global':
                goal = np.mean(goals, axis=0)
                self.chase_cam.step(center=goal)
            else:
                self.chase_cam.step(all_dynamics[self.camera_drone_index].pos, all_dynamics[self.camera_drone_index].vel)
                self.fpv_lookat = all_dynamics[self.camera_drone_index].look_at()
            # use this to get trails on the goals and visualize the paths they follow
            # bodies = []
            # bodies.extend(self.goal_transforms)
            # world = r3d.World(bodies)
            # batch = r3d.Batch()
            # world.build(batch)
            # self.scene.batches.extend([batch])

            self.update_goals(goals=goals)
            if self.obstacle_mode != 'no_obstacles':
                self.update_obstacles(obstacles=obstacles)

            for i, dyn in enumerate(all_dynamics):
                matrix = r3d.trans_and_rot(dyn.pos, dyn.rot)
                self.quad_transforms[i].set_transform_nocollide(matrix)

                shadow_pos = 0 + dyn.pos
                shadow_pos[2] = 0.001  # avoid z-fighting
                matrix = r3d.translate(shadow_pos)
                self.shadow_transforms[i].set_transform_nocollide(matrix)

                matrix = r3d.translate(dyn.pos)
                if collisions['drone'][i] > 0.0 or collisions['obstacle'][i] > 0.0 or collisions['ground'][i] > 0.0:
                    # Multiplying by 1 converts bool into float
                    self.collision_transforms[i].set_transform_and_color(matrix, (
                        (collisions['drone'][i] > 0.0) * 1.0, (collisions['obstacle'][i] > 0.0) * 1.0,
                        (collisions['ground'][i] > 0.0) * 1.0, 0.4))
                else:
                    self.collision_transforms[i].set_transform_and_color(matrix, (0, 0, 0, 0.0))

    def render_chase(self, all_dynamics, goals, collisions, mode='human', obstacles=None):
        if mode == 'human':
            if self.window_target is None:
                self.window_target = r3d.WindowTarget(self.window_w, self.window_h, resizable=self.resizable)
                self.window_target.window.on_key_press = self.window_on_key_press
                self.keys = key.KeyStateHandler()
                self.window_target.window.on_key_release = self.window_on_key_release
                self._make_scene()

            self.window_smooth_change_view()
            self.update_state(all_dynamics=all_dynamics, goals=goals, obstacles=obstacles, collisions=collisions)
            self.cam3p.look_at(*self.chase_cam.look_at())
            r3d.draw(self.scene, self.cam3p, self.window_target)
            return None
        elif mode == 'rgb_array':
            if self.video_target is None:
                self.video_target = r3d.FBOTarget(self.window_h, self.window_h)
                self._make_scene()
            self.update_state(all_dynamics=all_dynamics, goals=goals, obstacles=obstacles, collisions=collisions)
            self.cam3p.look_at(*self.chase_cam.look_at())
            r3d.draw(self.scene, self.cam3p, self.video_target)
            return np.flipud(self.video_target.read())

    def window_on_key_press(self, symbol, modifiers):
        # <- LEFT Rotation :
        if symbol == key.LEFT:
            self.chase_cam.phi -= self.camera_rot_step_size
            self.keys[key.LEFT] = 1
        # -> Right Rotation :
        elif symbol == key.RIGHT:
            self.chase_cam.phi += self.camera_rot_step_size
            self.keys[key.RIGHT] = 1
        elif symbol == key.UP:
            self.chase_cam.theta -= self.camera_rot_step_size
            self.keys[key.UP] = 1
        elif symbol == key.DOWN:
            self.chase_cam.theta += self.camera_rot_step_size
            self.keys[key.DOWN] = 1
        elif symbol == key.Z:
            # Zoom In
            self.chase_cam.radius -= self.camera_zoom_step_size
            self.keys[key.Z] = 1
        elif symbol == key.X:
            # Zoom Out
            self.chase_cam.radius += self.camera_zoom_step_size
            self.keys[key.X] = 1
        elif symbol == key.L:
            self.viepoint = 'local'
            self.chase_cam = ChaseCamera(view_dist=self.diameter * 15)
            self.chase_cam.reset(self.goals[0][0:3], self.dynamics[0].pos, self.dynamics[0].vel)
        elif symbol == key.G:
            self.viepoint = 'global'
            self.chase_cam = GlobalCamera(view_dist=2.5)
            goal = np.mean(self.goals, axis=0)
            self.chase_cam.reset(view_dist=2.5, center=goal)
        elif key.NUM_0 <= symbol <= key.NUM_9:
            index = min(symbol - key.NUM_0, self.num_agents-1)
            self.camera_drone_index = index
            self.viepoint = 'local'
            self.chase_cam = ChaseCamera(view_dist=self.diameter * 15)
            self.chase_cam.reset(self.goals[index][0:3], self.dynamics[index].pos, self.dynamics[index].vel)
        elif symbol == key.Q :
            # Decrease the step size of Rotation
            if self.camera_rot_step_size <= np.pi / 18:
                print('Current rotation step size for camera is the minimum!')
            else:
                self.camera_rot_step_size /= 2
        elif symbol == key.P:
            # Increase the step size of Rotation
            if self.camera_rot_step_size >= np.pi / 2:
                print('Current rotation step size for camera is the maximum!')
            else:
                self.camera_rot_step_size *= 2
        elif symbol == key.W:
            # Decrease the step size of Zoom
            if self.camera_zoom_step_size <= 0.1:
                print('Current zoom step size for camera is the minimum!')
            else:
                self.camera_zoom_step_size -= 0.1
        elif symbol == key.O:
            # Increase the step size of Zoom
            if self.camera_zoom_step_size >= 2.0:
                print('Current zoom step size for camera is the maximum!')
            else:
                self.camera_zoom_step_size += 0.1
        elif symbol == key.J:
            self.chase_cam.center += np.array([0., 0., 0.05])
            self.keys[key.J] = 1
        elif symbol == key.N:
            self.chase_cam.center += np.array([0., 0., -0.05])
            self.keys[key.N] = 1
        elif symbol == key.B:
            self.chase_cam.center += np.array([0., -0.05, 0.])
            self.keys[key.B] = 1
        elif symbol == key.M:
            self.chase_cam.center += np.array([0., 0.05, 0.])
            self.keys[key.M] = 1
        elif symbol == key.SPACE:
            self.chase_cam.center, self.chase_cam.phi, self.chase_cam.theta = self.chase_cam.center_list[self.chase_cam.center_list_id]
            self.chase_cam.center_list_id = (self.chase_cam.center_list_id + 1) % len(self.chase_cam.center_list)

    def window_on_key_release(self, symbol, modifiers):
        self.keys = key.KeyStateHandler()

    def window_smooth_change_view(self):
        if self.keys[key.LEFT]:
            self.chase_cam.phi -= self.camera_rot_step_size
        elif self.keys[key.RIGHT]:
            self.chase_cam.phi += self.camera_rot_step_size
        elif self.keys[key.UP]:
            self.chase_cam.theta -= self.camera_rot_step_size
        elif self.keys[key.DOWN]:
            self.chase_cam.theta += self.camera_rot_step_size
        elif self.keys[key.Z]:
            self.chase_cam.radius -= self.camera_zoom_step_size
        elif self.keys[key.X]:
            self.chase_cam.radius += self.camera_zoom_step_size
        elif self.keys[key.J]:
            self.chase_cam.center += np.array([0., 0., 0.05])
        elif self.keys[key.N]:
            self.chase_cam.center += np.array([0., 0., -0.05])
        elif self.keys[key.B]:
            self.chase_cam.center += np.array([0., -0.05, 0.])
        elif self.keys[key.M]:
            self.chase_cam.center += np.array([0., 0.05, 0.])
