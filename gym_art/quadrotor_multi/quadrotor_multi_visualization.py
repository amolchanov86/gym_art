import copy
import numpy as np

import gym_art.quadrotor_multi.rendering3d as r3d

from gym_art.quadrotor_multi.quadrotor_visualization import ChaseCamera, SideCamera, quadrotor_simple_3dmodel, \
    quadrotor_3dmodel
from gym_art.quadrotor_multi.params import quad_color


class Quadrotor3DSceneMulti:
    def __init__(
            self, w, h,
            quad_arm=None, models=None, obstacles=True, visible=True, resizable=True, goal_diameter=None,
            viewpoint='chase', obs_hw=None,
    ):
        if obs_hw is None:
            obs_hw = [64, 64]

        self.window_target = None
        self.window_w, self.window_h = w, h
        self.resizable = resizable
        self.viepoint = viewpoint
        self.obs_hw = copy.deepcopy(obs_hw)

        self.quad_arm = quad_arm
        self.obstacles = obstacles
        self.models = models

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

        self.fpv_lookat = None

        self.scene = None
        self.window_target = None
        self.obs_target = None
        self.video_target = None

    def update_goal_diameter(self):
        if self.quad_arm is not None:
            self.diameter = 2 * self.quad_arm
        else:
            self.diameter = 2 * np.linalg.norm(self.models[0].params['motor_pos']['xyz'][:2])

        if self.goal_forced_diameter:
            self.goal_diameter = self.goal_forced_diameter
        else:
            self.goal_diameter = self.diameter

    def _make_scene(self):
        self.cam1p = r3d.Camera(fov=90.0)
        self.cam3p = r3d.Camera(fov=45.0)

        self.quad_transforms, self.shadow_transforms, self.goal_transforms = [], [], []

        for i, model in enumerate(self.models):
            if model is not None:
                quad_transform = quadrotor_3dmodel(model, quad_id=i)
            else:
                quad_transform = quadrotor_simple_3dmodel(self.diameter)
            self.quad_transforms.append(quad_transform)

            self.shadow_transforms.append(
                r3d.transform_and_color(np.eye(4), (0, 0, 0, 0.4), r3d.circle(0.75 * self.diameter, 32))
            )

        # TODO make floor size or walls to indicate world_box
        floor = r3d.ProceduralTexture(r3d.random_textype(), (0.15, 0.25),
                                      r3d.rect((1000, 1000), (0, 100), (0, 100)))

        self.update_goal_diameter()
        self.chase_cam.view_dist = self.diameter * 15

        self.create_goals()

        bodies = [r3d.BackToFront([floor, st]) for st in self.shadow_transforms]
        bodies.extend(self.goal_transforms)
        bodies.extend(self.quad_transforms)

        # TODO: obstacles?

        world = r3d.World(bodies)
        batch = r3d.Batch()
        world.build(batch)

        self.scene = r3d.Scene(batches=[batch], bgcolor=(0, 0, 0))
        self.scene.initialize()

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

    def reset(self, goals, dynamics):
        first_goal = goals[0]  # TODO: make a camera that can look at all drones
        self.chase_cam.reset(first_goal[0:3], dynamics[0].pos, dynamics[0].vel)
        self.update_state(dynamics, goals)

    def update_state(self, all_dynamics, goals):
        if self.scene:
            self.chase_cam.step(all_dynamics[0].pos, all_dynamics[0].vel)
            self.fpv_lookat = all_dynamics[0].look_at()

            self.update_goals(goals=goals)

            for i, dyn in enumerate(all_dynamics):
                matrix = r3d.trans_and_rot(dyn.pos, dyn.rot)
                self.quad_transforms[i].set_transform_nocollide(matrix)

                shadow_pos = 0 + dyn.pos
                shadow_pos[2] = 0.001  # avoid z-fighting
                matrix = r3d.translate(shadow_pos)
                self.shadow_transforms[i].set_transform_nocollide(matrix)

    def render_chase(self, all_dynamics, goals, mode='human'):
        if mode == 'human':
            if self.window_target is None:
                self.window_target = r3d.WindowTarget(self.window_w, self.window_h, resizable=self.resizable)
                self._make_scene()
            self.update_state(all_dynamics=all_dynamics, goals=goals)
            self.cam3p.look_at(*self.chase_cam.look_at())
            r3d.draw(self.scene, self.cam3p, self.window_target)
            return None
        elif mode == 'rgb_array':
            if self.video_target is None:
                self.video_target = r3d.FBOTarget(self.window_h, self.window_h)
                self._make_scene()
            self.update_state(all_dynamics=all_dynamics, goals=goals)
            self.cam3p.look_at(*self.chase_cam.look_at())
            r3d.draw(self.scene, self.cam3p, self.video_target)
            return np.flipud(self.video_target.read())

