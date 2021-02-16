import copy
import numpy as np
from scipy import spatial
import random
import time
import gym

from copy import deepcopy

from gym_art.quadrotor_multi.quad_utils import perform_collision_between_drones, perform_collision_with_obstacle, \
    calculate_collision_matrix, hyperbolic_proximity_penalty
from gym_art.quadrotor_multi.quadrotor_multi_obstacles import MultiObstacles
from gym_art.quadrotor_multi.quadrotor_single import GRAV, QuadrotorSingle
from gym_art.quadrotor_multi.quadrotor_multi_visualization import Quadrotor3DSceneMulti
from gym_art.quadrotor_multi.quad_scenarios import create_scenario
from gym_art.quadrotor_multi.numba_utils import OUNoiseNumba
from gym_art.quadrotor_multi.quad_utils import OUNoise

EPS = 1E-6


class QuadrotorEnvMulti(gym.Env):
    def __init__(self,
                 num_agents,
                 dynamics_params='DefaultQuad', dynamics_change=None,
                 dynamics_randomize_every=None, dyn_sampler_1=None, dyn_sampler_2=None,
                 raw_control=True, raw_control_zero_middle=True, dim_mode='3D', tf_control=False, sim_freq=200.,
                 sim_steps=2, obs_repr='xyz_vxyz_R_omega', ep_time=7, obstacles_num=0, room_length=10, room_width=10,
                 room_height=10, init_random_state=False, rew_coeff=None, sense_noise=None, verbose=False, gravity=GRAV,
                 resample_goals=False, t2w_std=0.005, t2t_std=0.0005, excite=False, dynamics_simplification=False,
                 quads_mode='static_same_goal', quads_formation='circle_horizontal', quads_formation_size=-1.0,
                 swarm_obs='none', quads_use_numba=False, quads_settle=False, quads_settle_range_meters=1.0,
                 quads_vel_reward_out_range=0.8, quads_obstacle_mode='no_obstacles', quads_view_mode='local',
                 quads_obstacle_num=0, quads_obstacle_type='sphere', quads_obstacle_size=0.0, collision_force=True,
                 adaptive_env=False, obstacle_traj='gravity', local_obs=-1, replay_buffer=False):

        super().__init__()

        self.num_agents = num_agents
        self.swarm_obs = swarm_obs
        assert local_obs <= self.num_agents - 1 or local_obs == -1, f'Invalid value ({local_obs}) passed to --local_obs. Should be 0 < n < num_agents - 1, or -1'
        if local_obs == -1:
            self.num_use_neighbor_obs = self.num_agents - 1
        else:
            self.num_use_neighbor_obs = local_obs
        # Set to True means that sample_factory will treat it as a multi-agent vectorized environment even with
        # num_agents=1. More info, please look at sample-factory: envs/quadrotors/wrappers/reward_shaping.py
        self.is_multiagent = True
        self.room_dims = (room_length, room_width, room_height)

        self.envs = []
        self.adaptive_env = adaptive_env
        self.quads_view_mode= quads_view_mode

        for i in range(self.num_agents):
            e = QuadrotorSingle(
                dynamics_params, dynamics_change, dynamics_randomize_every, dyn_sampler_1, dyn_sampler_2,
                raw_control, raw_control_zero_middle, dim_mode, tf_control, sim_freq, sim_steps,
                obs_repr, ep_time, obstacles_num, room_length, room_width, room_height, init_random_state,
                rew_coeff, sense_noise, verbose, gravity, t2w_std, t2t_std, excite, dynamics_simplification,
                quads_use_numba, self.swarm_obs, self.num_agents, quads_settle, quads_settle_range_meters,
                quads_vel_reward_out_range, quads_view_mode, quads_obstacle_mode, quads_obstacle_num,
                self.num_use_neighbor_obs
            )
            self.envs.append(e)

        self.resample_goals = resample_goals

        # we don't actually create a scene object unless we want to render stuff
        self.scene = None

        self.action_space = self.envs[0].action_space
        self.observation_space = self.envs[0].observation_space

        # reward shaping
        self.rew_coeff = dict(
            pos=1., effort=0.05, action_change=0., crash=1., orient=1., yaw=0., rot=0., attitude=0., spin=0.1, vel=0.,
            quadcol_bin=0., quadsettle=0., quadcol_bin_obst=0., quad_spacing_coeff=0.
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

        # Aux variables for observation space of quads
        self.pos = np.zeros([self.num_agents, 3])  # Matrix containing all positions
        self.quads_mode = quads_mode
        if obs_repr == 'xyz_vxyz_R_omega':
            obs_self_size = 18
        else:
            raise NotImplementedError(f'{obs_repr} not supported!')

        if self.swarm_obs == 'pos_vel':
            self.neighbor_obs_size = 6
        elif self.swarm_obs == 'attn':
            self.neighbor_obs_size = 11
        elif self.swarm_obs == 'pos_vel_goals':
            self.neighbor_obs_size = 9
        elif self.swarm_obs == 'none':
            self.neighbor_obs_size = 0
        else:
            raise NotImplementedError(f'Unknown value {self.swarm_obs} passed to --neighbor_obs_type')
        self.clip_neighbor_space_length = self.num_use_neighbor_obs * self.neighbor_obs_size
        self.clip_neighbor_space_min_box = self.observation_space.low[obs_self_size:obs_self_size+self.clip_neighbor_space_length]
        self.clip_neighbor_space_max_box = self.observation_space.high[obs_self_size:obs_self_size+self.clip_neighbor_space_length]

        # Aux variables for rewards
        self.rews_settle = np.zeros(self.num_agents)
        self.rews_settle_raw = np.zeros(self.num_agents)

        # Aux variables for scenarios
        self.scenario = create_scenario(quads_mode=quads_mode, envs=self.envs, num_agents=self.num_agents,
                                        room_dims=self.room_dims, room_dims_callback=self.set_room_dims, rew_coeff=self.rew_coeff,
                                        quads_formation=quads_formation, quads_formation_size=quads_formation_size)
        self.quads_formation_size = quads_formation_size
        self.goal_central = np.array([0., 0., 2.])

        # Set Obstacles
        self.obstacle_max_init_vel = 4.0 * self.envs[0].max_init_vel
        self.obstacle_init_box = 0.5 * self.envs[0].box  # box of env is: 2 meters
        obstacle_bound_box = 4 * self.obstacle_init_box  # obstacle_bound_box: 4 meters
        # obstacle_room: [[-4, -4, 0], [4, 4, 10]]
        # This parameter is used to judge whether to obstacles are out of room, and then, we can reset the obstacles
        self.obstacle_room = np.array(
            [[-obstacle_bound_box, -obstacle_bound_box, 0.], [obstacle_bound_box, obstacle_bound_box, 10.0]])
        self.dt = 1.0 / sim_freq
        self.obstacle_mode = quads_obstacle_mode
        self.obstacle_num = quads_obstacle_num
        self.obstacle_type = quads_obstacle_type
        self.obstacle_size = quads_obstacle_size
        self.set_obstacles = False
        self.obstacle_settle_count = np.zeros(self.num_agents)
        self.obstacle_obs_len = 6  # pos and vel
        self.metric_dist_quads_settle_with_obstacle = self.get_obst_metric()

        self.obstacles = MultiObstacles(
            mode=self.obstacle_mode, num_obstacles=self.obstacle_num, max_init_vel=self.obstacle_max_init_vel,
            init_box=self.obstacle_init_box, dt=self.dt, quad_size=self.envs[0].dynamics.arm, type=self.obstacle_type,
            size=self.obstacle_size, traj=obstacle_traj
        )

        # set render
        self.simulation_start_time = 0
        self.frames_since_last_render = self.render_skip_frames = 0
        self.render_every_nth_frame = 1
        self.render_speed = 1.0  # set to below 1 slowmo, higher than 1 for fast forward (if simulator can keep up)

        # measuring the total number of pairwise collisions per episode
        self.collisions_per_episode = 0

        # some collisions may happen because the quadrotors get initialized on the collision course
        # if we wait a couple of seconds, then we can eliminate all the collisions that happen due to initialization
        # this is the actual metric that we want to minimize
        self.collisions_after_settle = 0
        self.collisions_grace_period_seconds = 1.5

        self.prev_drone_collisions, self.curr_drone_collisions = [], []
        self.all_collisions = {}
        self.apply_collision_force = collision_force

        # set to true whenever we need to reset the OpenGL scene in render()
        self.reset_scene = False
        self.deleted_scene = False

        self.use_replay_buffer = replay_buffer
        self.activate_replay_buffer = False  # only start using the buffer after the drones learn how to fly
        self.saved_in_replay_buffer = False  # since the same collisions happen during replay, we don't want to keep resaving the same event
        self.collision_occurred = False
        self.replay_buffer = None
        if self.use_replay_buffer:
            self.crash_one_episode = 0
        self.crashes_hundred_eps = []

    def set_room_dims(self, dims):
        # dims is a (x, y, z) tuple
        self.room_dims = dims

    def all_dynamics(self):
        return tuple(e.dynamics for e in self.envs)

    def get_obst_metric(self):
        # Distance between every two quadrotors is 4 quads_arm_len
        quad_arm_size = self.envs[0].dynamics.arm
        metric_dist = 4.0 * quad_arm_size * np.sin(np.pi / 2 - np.pi / self.num_agents) / np.sin(
            2 * np.pi / self.num_agents)
        return metric_dist

    def get_obs_neighbor_rel(self, env_id):
        i = env_id
        pos_neighbors = np.stack([self.envs[j].dynamics.pos for j in range(len(self.envs)) if j != i])
        pos_neighbors_rel = pos_neighbors - self.envs[i].dynamics.pos
        dist_to_neighbors = np.linalg.norm(pos_neighbors_rel, axis=1).reshape(-1, 1)
        vel_neighbors = np.stack([self.envs[j].dynamics.vel for j in range(len(self.envs)) if j != i])
        vel_neighbors_rel = vel_neighbors - self.envs[i].dynamics.vel
        neighbor_goals_rel = np.stack([self.envs[j].goal for j in range(len(self.envs)) if j != i]) - self.envs[i].dynamics.pos
        dist_to_neighbor_goals = np.linalg.norm(neighbor_goals_rel, axis=1).reshape(-1, 1)

        if self.swarm_obs == 'pos_vel':
            obs_neighbor_rel = np.concatenate((pos_neighbors_rel, vel_neighbors_rel), axis=1)
        elif self.swarm_obs == 'pos_vel_goals':
            obs_neighbor_rel = np.concatenate((pos_neighbors_rel, vel_neighbors_rel, neighbor_goals_rel), axis=1)
        elif self.swarm_obs == 'attn':
            obs_neighbor_rel = np.concatenate((pos_neighbors_rel, dist_to_neighbors, vel_neighbors_rel, neighbor_goals_rel, dist_to_neighbor_goals), axis=1)
        else:
            raise NotImplementedError

        return obs_neighbor_rel

    def extend_obs_space(self, obs):
        assert self.swarm_obs == 'pos_vel' or self.swarm_obs == 'pos_vel_goals' or self.swarm_obs == 'attn', f'Invalid parameter {self.swarm_obs} passed in --obs_space'
        obs_neighbors = []
        for i in range(len(self.envs)):
            obs_neighbor_rel = self.get_obs_neighbor_rel(env_id=i)
            obs_neighbors.append(obs_neighbor_rel.reshape(-1))
        obs_neighbors = np.stack(obs_neighbors)

        # clip observation space of neighborhoods
        obs_neighbors = np.clip(
            obs_neighbors, a_min=self.clip_neighbor_space_min_box, a_max=self.clip_neighbor_space_max_box,
        )
        obs_ext = np.concatenate((obs, obs_neighbors), axis=1)
        return obs_ext

    def extend_obs_space_n_closest(self, obs):
        obs_neighbors = []
        for i in range(len(obs)):
            obs_neighbor_rel = self.get_obs_neighbor_rel(env_id=i)
            # Get n close neighbors
            rel_pos = np.linalg.norm(obs_neighbor_rel[:, :3], axis=1)
            rel_pos_index = rel_pos.argsort()
            obs_neighbor_rel_n_close = np.array(
                [obs_neighbor_rel[rel_pos_index[i]] for i in range(self.num_use_neighbor_obs)])
            obs_neighbors.append(obs_neighbor_rel_n_close.reshape(-1))
        obs_neighbors = np.stack(obs_neighbors)
        # clip observation space of neighborhoods
        obs_neighbors = np.clip(
            obs_neighbors, a_min=self.clip_neighbor_space_min_box, a_max=self.clip_neighbor_space_max_box,
        )
        obs_ext = np.concatenate((obs, obs_neighbors), axis=1)
        return obs_ext

    def can_drones_fly(self):
        res = abs(np.mean(self.crashes_hundred_eps[:100])) < 1 and len(self.crashes_hundred_eps) >= 1
        self.crashes_hundred_eps = self.crashes_hundred_eps[:100]  # garbage collect unneeded entries
        return res

    def reset_obstacle_mode(self):
        self.obstacle_mode = self.envs[0].obstacle_mode
        self.obstacle_num = self.envs[0].obstacle_num

    def reset_scene_multi(self):
        models = tuple(e.dynamics.model for e in self.envs)
        self.scene = Quadrotor3DSceneMulti(
            models=models,
            w=640, h=480, resizable=True, obstacles=self.obstacles, viewpoint=self.envs[0].viewpoint,
            obstacle_mode=self.obstacle_mode, room_dims=self.room_dims, num_agents=self.num_agents,
            render_speed=self.render_speed, formation_size=self.quads_formation_size,
        )

    def reset_thrust_noise(self):
        for e in self.envs:
            if e.dynamics.use_numba:
                e.dynamics.thrust_noise = OUNoiseNumba(4, sigma=0.2 * e.dynamics.thrust_noise_ratio)
            else:
                e.dynamics.thrust_noise = OUNoise(4, sigma=0.2 * e.dynamics.thrust_noise_ratio)

    def reset(self):
        obs, rewards, dones, infos = [], [], [], []
        self.scenario.reset()
        self.quads_formation_size = self.scenario.formation_size
        self.goal_central = np.mean(self.scenario.goals, axis=0)

        self.reset_obstacle_mode()

        models = tuple(e.dynamics.model for e in self.envs)

        # try to activate replay buffer if enabled
        if self.use_replay_buffer and not self.activate_replay_buffer:
            self.crashes_hundred_eps.append(self.crash_one_episode)
            self.activate_replay_buffer = self.can_drones_fly()

        if self.adaptive_env:
            # TODO: introduce logic to choose the new room dims i.e. based on statistics from last N episodes, etc
            # e.g. self.room_dims = ....
            new_length, new_width, new_height = np.random.randint(1, 31, 3)
            self.room_dims = (new_length, new_width, new_height)

        for i, e in enumerate(self.envs):
            e.goal = self.scenario.goals[i]
            e.rew_coeff = self.rew_coeff
            e.update_env(*self.room_dims)

            observation = e.reset()
            obs.append(observation)

        # extend obs to see neighbors
        if self.swarm_obs != 'none' and self.num_agents > 1:
            if self.num_use_neighbor_obs == (self.num_agents - 1):
                obs_ext = self.extend_obs_space(obs)
            else:
                obs_ext = self.extend_obs_space_n_closest(obs)
            obs = obs_ext

        # Reset Obstacles
        self.set_obstacles = False
        self.obstacle_settle_count = np.zeros(self.num_agents)
        quads_pos = np.array([e.dynamics.pos for e in self.envs])
        quads_vel = np.array([e.dynamics.vel for e in self.envs])
        if self.obstacle_num > 0:
            obs = self.obstacles.reset(obs=obs, quads_pos=quads_pos, quads_vel=quads_vel,
                                       set_obstacles=self.set_obstacles)
        self.all_collisions = {val: [0.0 for _ in range(len(self.envs))] for val in ['drone', 'ground', 'obstacle']}

        self.collisions_per_episode = self.collisions_after_settle = 0

        self.reset_scene = True

        self.crash_one_episode = 0
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

        if self.swarm_obs != 'none' and self.num_agents > 1:
            if self.num_use_neighbor_obs == (self.num_agents - 1):
                obs_ext = self.extend_obs_space(obs)
            else:
                obs_ext = self.extend_obs_space_n_closest(obs)
            obs = obs_ext

        if self.use_replay_buffer and not self.activate_replay_buffer:
            self.crash_one_episode += infos[0]["rewards"]["rew_crash"]

        # Calculating collisions between drones
        drone_col_matrix, self.curr_drone_collisions = calculate_collision_matrix(self.pos, self.envs[0].dynamics.arm)

        unique_collisions = np.setdiff1d(self.curr_drone_collisions, self.prev_drone_collisions)

        if unique_collisions.sum() > 0:
            self.collision_occurred = True

        # collision between 2 drones counts as a single collision
        collisions_curr_tick = len(unique_collisions) // 2
        self.collisions_per_episode += collisions_curr_tick

        if collisions_curr_tick > 0:
            if self.envs[0].tick >= self.collisions_grace_period_seconds * self.envs[0].control_freq:
                self.collisions_after_settle += collisions_curr_tick

        self.prev_drone_collisions = self.curr_drone_collisions

        rew_collisions_raw = np.zeros(self.num_agents)
        if unique_collisions.any():
            rew_collisions_raw[unique_collisions] = -1.0
        rew_collisions = self.rew_coeff["quadcol_bin"] * rew_collisions_raw

        # COLLISION BETWEEN QUAD AND OBSTACLE(S)
        col_obst_quad = self.obstacles.collision_detection(pos_quads=self.pos, set_obstacles=self.set_obstacles)
        rew_col_obst_quad_raw = - np.sum(col_obst_quad, axis=0)
        rew_col_obst_quad = self.rew_coeff["quadcol_bin_obst"] * rew_col_obst_quad_raw

        # Collisions with ground
        ground_collisions = [1.0 if pos[2] < 0.25 else 0.0 for pos in self.pos]

        self.all_collisions = {'drone': np.sum(drone_col_matrix, axis=1), 'ground': ground_collisions,
                               'obstacle': col_obst_quad.sum(axis=0)}

        # Applying random forces for all collisions between drones and obstacles
        if self.apply_collision_force:
            for val in self.curr_drone_collisions:
                perform_collision_between_drones(self.envs[val[0]].dynamics, self.envs[val[1]].dynamics)
            for val in np.argwhere(col_obst_quad > 0.0):
                perform_collision_with_obstacle(self.obstacles.obstacles[val[0]], self.envs[val[1]].dynamics)

        # compute clipped 1/x^2 cost for distance b/w drones
        dists = spatial.distance_matrix(x=self.pos, y=self.pos)
        dt = 1.0 / self.envs[0].control_freq
        spacing_reward = hyperbolic_proximity_penalty(dists, dt)

        for i in range(self.num_agents):
            rewards[i] += rew_collisions[i]
            infos[i]["rewards"]["rew_quadcol"] = rew_collisions[i]
            infos[i]["rewards"]["rewraw_quadcol"] = rew_collisions_raw[i]

            rewards[i] += rew_col_obst_quad[i]
            infos[i]["rewards"]["rew_quadcol_obstacle"] = rew_col_obst_quad[i]
            infos[i]["rewards"]["rewraw_quadcol_obstacle"] = rew_col_obst_quad_raw[i]

            rewards[i] += spacing_reward[i]
            infos[i]["rewards"]["rew_quad_spacing"] = spacing_reward[i]

        # run the scenario passed to self.quads_mode
        infos, rewards = self.scenario.step(infos=infos, rewards=rewards, pos=self.pos)

        # For obstacles
        quads_vel = np.array([e.dynamics.vel for e in self.envs])
        if self.quads_mode == "mix" and self.obstacle_mode == "no_obstacles" and self.obstacle_num > 0:
            obs = self.obstacles.step(obs=obs, quads_pos=self.pos, quads_vel=quads_vel, set_obstacles=False)

        if self.obstacle_mode == 'dynamic' and self.obstacle_num > 0:
            tmp_obs = self.obstacles.step(obs=obs, quads_pos=self.pos, quads_vel=quads_vel,
                                          set_obstacles=self.set_obstacles)

            if self.set_obstacles:
                for obstacle in self.obstacles.obstacles:
                    obstacle_pos = copy.deepcopy(obstacle.pos)
                    obstacle_pos[2] = obstacle.pos[2] - obstacle.size
                    if not np.array_equal(obstacle_pos,
                                          np.clip(obstacle_pos,
                                                  a_min=self.obstacle_room[0],  # [-4, -4, 0]
                                                  a_max=self.obstacle_room[1])):  # [4, 4, 10]
                        self.set_obstacles = False
                        obstacle.reset(set_obstacle=self.set_obstacles)
                        self.obstacle_settle_count = np.zeros(self.num_agents)

            if not self.set_obstacles:
                for i, e in enumerate(self.envs):
                    dis = np.linalg.norm(self.pos[i] - e.goal)
                    if abs(dis) < self.metric_dist_quads_settle_with_obstacle:
                        self.obstacle_settle_count[i] += 1
                    else:
                        self.obstacle_settle_count = np.zeros(self.num_agents)
                        break

                # drones settled at the goal for 1 sec
                control_step_for_one_sec = int(self.envs[0].control_freq)
                tmp_count = self.obstacle_settle_count >= control_step_for_one_sec
                if all(tmp_count):
                    self.set_obstacles = True
                    tmp_obs = self.obstacles.reset(obs=obs, quads_pos=self.pos, quads_vel=quads_vel,
                                                   set_obstacles=self.set_obstacles)

            obs = tmp_obs

        # DONES
        if any(dones):
            for i in range(len(infos)):
                infos[i]['episode_extra_stats'] = {
                    'num_collisions': self.collisions_per_episode,
                    'num_collisions_after_settle': self.collisions_after_settle,
                }

            obs = self.reset()
            dones = [True] * len(dones)  # terminate the episode for all "sub-envs"

        return obs, rewards, dones, infos

    def render(self, mode='human', verbose=False):
        models = tuple(e.dynamics.model for e in self.envs)

        if self.scene is None:
            self.scene = Quadrotor3DSceneMulti(
                models=models,
                w=640, h=480, resizable=True, obstacles=self.obstacles, viewpoint=self.envs[0].viewpoint,
                obstacle_mode=self.obstacle_mode, room_dims=self.room_dims, num_agents=self.num_agents,
                render_speed=self.render_speed, formation_size=self.quads_formation_size,
            )

        if self.reset_scene:
            self.scene.update_models(models)
            self.scene.formation_size = self.quads_formation_size
            self.scene.update_env(self.room_dims)

            self.scene.reset(tuple(e.goal for e in self.envs), self.all_dynamics(), self.obstacles, self.all_collisions)

            self.reset_scene = False

        if self.quads_mode == "mix":
            self.scene.formation_size = self.scenario.scenario.formation_size
        else:
            self.scene.formation_size = self.scenario.formation_size
        self.frames_since_last_render += 1

        if self.render_skip_frames > 0:
            self.render_skip_frames -= 1
            return None

        # this is to handle the 1st step of the simulation that will typically be very slow
        if self.simulation_start_time > 0:
            simulation_time = time.time() - self.simulation_start_time
        else:
            simulation_time = 0

        realtime_control_period = 1 / self.envs[0].control_freq

        render_start = time.time()
        goals = tuple(e.goal for e in self.envs)
        self.scene.render_chase(all_dynamics=self.all_dynamics(), goals=goals, collisions=self.all_collisions,
                                mode=mode, obstacles=self.obstacles)
        # Update the formation size of the scenario
        if self.quads_mode == "mix":
            self.scenario.scenario.update_formation_size(self.scene.formation_size)
        else:
            self.scenario.update_formation_size(self.scene.formation_size)

        render_time = time.time() - render_start

        desired_time_between_frames = realtime_control_period * self.frames_since_last_render / self.render_speed
        time_to_sleep = desired_time_between_frames - simulation_time - render_time

        # wait so we don't simulate/render faster than realtime
        if mode == 'human' and time_to_sleep > 0:
            time.sleep(time_to_sleep)

        if simulation_time + render_time > desired_time_between_frames:
            self.render_every_nth_frame += 1
            if verbose:
                print(f'Last render + simulation time {render_time + simulation_time:.3f}')
                print(f'Rendering does not keep up, rendering every {self.render_every_nth_frame} frames')
        elif simulation_time + render_time < realtime_control_period * (
                self.frames_since_last_render - 1) / self.render_speed:
            self.render_every_nth_frame -= 1
            if verbose:
                print(f'We can increase rendering framerate, rendering every {self.render_every_nth_frame} frames')

        if self.render_every_nth_frame > 4:
            self.render_every_nth_frame = 4
            print(f'Rendering cannot keep up! Rendering every {self.render_every_nth_frame} frames')

        self.render_skip_frames = self.render_every_nth_frame - 1
        self.frames_since_last_render = 0

        self.simulation_start_time = time.time()
