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
            viewpoint='chase', obs_hw=None, obstacle_mode='no_obstacles', room_dims=(10, 10, 10), num_agents=8,
            render_speed=1.0, formation_size=-1.0,
    ):
        if obs_hw is None:
            obs_hw = [64, 64]

        self.window_target = None
        self.window_w, self.window_h = w, h
        self.resizable = resizable
        self.viewpoint = viewpoint
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

        if self.viewpoint == 'chase':
            self.chase_cam = ChaseCamera(view_dist=self.diameter * 15)
        elif self.viewpoint == 'side':
            self.chase_cam = SideCamera(view_dist=self.diameter * 15)
        elif self.viewpoint == 'global':
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

        # Aux camera moving
        standard_render_speed = 1.0
        speed_ratio = render_speed / standard_render_speed
        self.camera_rot_step_size = np.pi / 45 * speed_ratio
        self.camera_zoom_step_size = 0.1 * speed_ratio
        self.camera_mov_step_size = 0.1 * speed_ratio
        self.formation_size = formation_size

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
        self._make_scene()

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

        if self.viewpoint == 'global':
            goal = np.mean(goals, axis=0)
            self.chase_cam.reset(view_dist=2.5, center=goal)
        else:
            goal = goals[self.camera_drone_index]  # TODO: make a camera that can look at all drones
            self.chase_cam.reset(goal[0:3], dynamics[self.camera_drone_index].pos, dynamics[self.camera_drone_index].vel)


        self.update_state(dynamics, goals, obstacles, collisions)

    def update_state(self, all_dynamics, goals, obstacles, collisions):
        if self.scene:
            if self.viewpoint == 'global':
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
                self.keys = key.KeyStateHandler()
                self.window_target.window.push_handlers(self.keys)
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

    def window_smooth_change_view(self):
        if len(self.keys) == 0:
            return

        symbol = list(self.keys)
        if key.NUM_0 <= symbol[0] <= key.NUM_9:
            index = min(symbol[0] - key.NUM_0, self.num_agents-1)
            self.camera_drone_index = index
            self.viewpoint = 'local'
            self.chase_cam = ChaseCamera(view_dist=self.diameter * 15)
            self.chase_cam.reset(self.goals[index][0:3], self.dynamics[index].pos, self.dynamics[index].vel)
            return

        if self.keys[key.L]:
            self.viewpoint = 'local'
            self.chase_cam = ChaseCamera(view_dist=self.diameter * 15)
            self.chase_cam.reset(self.goals[0][0:3], self.dynamics[0].pos, self.dynamics[0].vel)
            return
        if self.keys[key.G]:
            self.viewpoint = 'global'
            self.chase_cam = GlobalCamera(view_dist=2.5)
            goal = np.mean(self.goals, axis=0)
            self.chase_cam.reset(view_dist=2.5, center=goal)

        if not isinstance(self.chase_cam, GlobalCamera):
            return

        if self.keys[key.LEFT]:
            # <- Left Rotation :
            self.chase_cam.phi -= self.camera_rot_step_size
        if self.keys[key.RIGHT]:
            # -> Right Rotation :
            self.chase_cam.phi += self.camera_rot_step_size
        if self.keys[key.UP]:
            self.chase_cam.theta -= self.camera_rot_step_size
        if self.keys[key.DOWN]:
            self.chase_cam.theta += self.camera_rot_step_size
        if self.keys[key.Z]:
            # Zoom In
            self.chase_cam.radius -= self.camera_zoom_step_size
        if self.keys[key.X]:
            # Zoom Out
            self.chase_cam.radius += self.camera_zoom_step_size
        if self.keys[key.Q]:
            # Decrease the step size of Rotation
            if self.camera_rot_step_size <= np.pi / 18:
                print('Current rotation step size for camera is the minimum!')
            else:
                self.camera_rot_step_size /= 2
        if self.keys[key.P]:
            # Increase the step size of Rotation
            if self.camera_rot_step_size >= np.pi / 2:
                print('Current rotation step size for camera is the maximum!')
            else:
                self.camera_rot_step_size *= 2
        if self.keys[key.W]:
            # Decrease the step size of Zoom
            if self.camera_zoom_step_size <= 0.1:
                print('Current zoom step size for camera is the minimum!')
            else:
                self.camera_zoom_step_size -= 0.1
        if self.keys[key.O]:
            # Increase the step size of Zoom
            if self.camera_zoom_step_size >= 2.0:
                print('Current zoom step size for camera is the maximum!')
            else:
                self.camera_zoom_step_size += 0.1
        if self.keys[key.J]:
            self.chase_cam.center += np.array([0., 0., self.camera_mov_step_size])
        if self.keys[key.N]:
            self.chase_cam.center += np.array([0., 0., -self.camera_mov_step_size])
        if self.keys[key.B]:
            angle = self.chase_cam.phi + np.pi / 2
            move_step = np.array([np.cos(angle), np.sin(angle), 0]) * self.camera_mov_step_size
            self.chase_cam.center -= move_step
        if self.keys[key.M]:
            angle = self.chase_cam.phi + np.pi / 2
            move_step = np.array([np.cos(angle), np.sin(angle), 0]) * self.camera_mov_step_size
            self.chase_cam.center += move_step

        if self.keys[key.NUM_ADD]:
            self.formation_size += 0.1
        elif self.keys[key.NUM_SUBTRACT]:
            self.formation_size -= 0.1


    def window_on_key_release(self, symbol, modifiers):
        self.keys = key.KeyStateHandler()
        self.window_target.window.push_handlers(self.keys)
