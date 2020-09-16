import copy
import math

import numpy as np
import scipy as scp
from scipy import spatial

import gym
import random
from gym_art.quadrotor_multi.quadrotor_single import GRAV, QuadrotorSingle
from gym_art.quadrotor_multi.quadrotor_multi_visualization import Quadrotor3DSceneMulti


class QuadrotorEnvMulti(gym.Env):
    def __init__(self,
                 num_agents,
                 dynamics_params='DefaultQuad', dynamics_change=None,
                 dynamics_randomize_every=None, dyn_sampler_1=None, dyn_sampler_2=None,
                 raw_control=True, raw_control_zero_middle=True, dim_mode='3D', tf_control=False, sim_freq=200.,
                 sim_steps=2, obs_repr='xyz_vxyz_R_omega', ep_time=7, obstacles_num=0, room_size=10,
                 init_random_state=False, rew_coeff=None, sense_noise=None, verbose=False, gravity=GRAV,
                 resample_goals=False, t2w_std=0.005, t2t_std=0.0005, excite=False, dynamics_simplification=False,
                 quads_dist_between_goals=0.3, quads_mode='circular_config', multi_agent=True):

        super().__init__()

        self.num_agents = num_agents
        self.multi_agent = multi_agent
        self.envs = []

        for i in range(self.num_agents):
            e = QuadrotorSingle(
                dynamics_params, dynamics_change, dynamics_randomize_every, dyn_sampler_1, dyn_sampler_2,
                raw_control, raw_control_zero_middle, dim_mode, tf_control, sim_freq, sim_steps,
                obs_repr, ep_time, obstacles_num, room_size, init_random_state,
                rew_coeff, sense_noise, verbose, gravity, t2w_std, t2t_std, excite, dynamics_simplification,
                self.multi_agent, self.num_agents
            )
            self.envs.append(e)

        self.resample_goals = resample_goals

        self.scene = None

        self.action_space = self.envs[0].action_space
        self.observation_space = self.envs[0].observation_space

        # reward shaping
        self.rew_coeff = dict(
            pos=1., effort=0.05, action_change=0., crash=1., orient=1., yaw=0., rot=0., attitude=0., spin=0.1, vel=0.,
            quadcol_bin=0., quadsettle=0.
        )
        rew_coeff_orig = copy.deepcopy(self.rew_coeff)

        if rew_coeff is not None:
            assert isinstance(rew_coeff, dict)
            assert set(rew_coeff.keys()).issubset(set(self.rew_coeff.keys()))
            self.rew_coeff.update(rew_coeff)
        for key in self.rew_coeff.keys():
            self.rew_coeff[key] = float(self.rew_coeff[key])

        orig_keys = list(rew_coeff_orig.keys())
        # Checking to make sure we didn't provide some false rew_coeffs (for example by misspelling one of the params)
        assert np.all([key in orig_keys for key in self.rew_coeff.keys()])

        ## Aux variables
        self.pos = np.zeros([self.num_agents, 3]) #Matrix containing all positions
        self.quads_mode = quads_mode

	## Set Goals
        delta = quads_dist_between_goals
        pi = np.pi
        self.goal = []
        for i in range(self.num_agents):
            degree = 2 * pi * i / self.num_agents
            goal_x = delta * np.cos(degree)
            goal_y = delta * np.sin(degree)
            goal = [goal_x, goal_y, 2.0]
            self.goal.append(goal)

        self.goal = np.array(self.goal)
        self.rews_settle = np.zeros(self.num_agents)
        self.rews_settle_raw = np.zeros(self.num_agents)
        self.settle_count = np.zeros(self.num_agents)

    def all_dynamics(self):
        return tuple(e.dynamics for e in self.envs)

    def extend_obs_space(self, obs):
        obs_ext = []
        for i in range(len(obs)):
            observs = obs[i] # obs of current agent
            for j in range(len(obs)):
                if i == j: continue
                obs_neighbor = obs[j][:6] # get xyz_vxyz info of neighboring agents
                obs_neighbor_rel = obs_neighbor - observs[:6] # get relative xyz_vxyz of neighbors to current agent
                observs = np.concatenate((observs, obs_neighbor_rel))
            obs_ext.append(observs)
        return obs_ext

    def reset(self):
        obs, rewards, dones, infos = [], [], [], []

        models = tuple(e.dynamics.model for e in self.envs)

        # TODO: don't create scene object if we're just training and no need to visualize?
        if self.scene is None:
            self.scene = Quadrotor3DSceneMulti(
                models=models,
                w=640, h=480, resizable=True, obstacles=self.envs[0].obstacles, viewpoint=self.envs[0].viewpoint,
            )
        else:
            self.scene.update_models(models)

        for i, e in enumerate(self.envs):
            e.goal = self.goal[i]
            e.rew_coeff = self.rew_coeff

            observation = e.reset()
            obs.append(observation)
        # extend obs to see neighbors

        if self.multi_agent and self.num_agents > 1:
            obs_ext = self.extend_obs_space(obs)
            obs = obs_ext

        self.scene.reset(tuple(e.goal for e in self.envs), self.all_dynamics())
        return obs

    # noinspection PyTypeChecker
    def step(self, actions):
        obs, rewards, dones, infos = [], [], [], []


        for i, a in enumerate(actions):
            self.envs[i].rew_coeff = self.rew_coeff

            observation, reward, done, info = self.envs[i].step(a)
            obs.append(observation)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)

            self.pos[i, :] = self.envs[i].dynamics.pos

        if self.multi_agent and self.num_agents > 1:
            obs_ext = self.extend_obs_space(obs)
            obs = obs_ext


        ## SWARM REWARDS
        # -- BINARY COLLISION REWARD
        self.dist = spatial.distance_matrix(x=self.pos, y=self.pos)
        self.collisions = (self.dist < 2 * self.envs[0].dynamics.arm).astype(np.float32)
        np.fill_diagonal(self.collisions, 0.0) # removing self-collision
        self.rew_collisions_raw = - np.sum(self.collisions, axis=1)
        self.rew_collisions = self.rew_coeff["quadcol_bin"] * self.rew_collisions_raw

        for i in range(self.num_agents):
            rewards[i] += self.rew_collisions[i]
            infos[i]["rewards"]["rew_quadcol"] = self.rew_collisions[i]
            infos[i]["rewards"]["rewraw_quadcol"] = self.rew_collisions_raw[i]

        if self.quads_mode == "circular_config":
            for i, e in enumerate(self.envs):
                dis = np.linalg.norm(self.pos[i] - e.goal)
                if abs(dis) < 0.02:
                    self.settle_count[i] += 1
                else:
                    self.settle_count[i] = 0
                    break

            # drones settled at the goal for 1 sec
            control_step_for_one_sec = int(self.envs[0].control_freq)
            tmp_count = self.settle_count >= control_step_for_one_sec
            if all(tmp_count):
                np.random.shuffle(self.goal)
                for i, env in enumerate(self.envs):
                    env.goal = self.goal[i]
                    # Add settle rewards
                    self.rews_settle_raw[i] = control_step_for_one_sec
                    self.rews_settle[i] = self.rew_coeff["quadsettle"] * self.rews_settle_raw[i]
                    rewards[i] += self.rews_settle[i]
                    infos[i]["rewards"]["rew_quadsettle"] = self.rews_settle[i]
                    infos[i]["rewards"]["rewraw_quadsettle"] = self.rews_settle_raw[i]

                self.rews_settle = np.zeros(self.num_agents)
                self.rews_settle_raw = np.zeros(self.num_agents)
                self.settle_count = np.zeros(self.num_agents)
        elif self.quads_mode == "same_goal":
            tick = self.envs[0].tick
            # teleport every 5 secs
            control_step_for_five_sec = int(5.0 * self.envs[0].control_freq)
            if tick % control_step_for_five_sec == 0 and tick > 0:
                box_size = self.envs[0].box
                x = (random.random() * 2 - 1) * box_size
                y = (random.random() * 2 - 1) * box_size
                z = random.random() * 2 * box_size
                if z < 0.25:
                    z = 0.25

                self.goal = [[x, y, z] for i in range(self.num_agents)]
                self.goal = np.array(self.goal)

                for i, env in enumerate(self.envs):
                    env.goal = self.goal[i]



        ## DONES
        if any(dones):
            obs = self.reset()
            dones = [True] * len(dones)  # terminate the episode for all "sub-envs"

        return obs, rewards, dones, infos

    def render(self, mode='human'):
        goals = tuple(e.goal for e in self.envs)
        return self.scene.render_chase(all_dynamics=self.all_dynamics(), goals=goals, mode=mode)
