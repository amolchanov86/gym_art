import numpy as np
import random
import bezier
import copy

from gym_art.quadrotor_multi.quad_utils import generate_points, get_circle_radius, get_sphere_radius, get_grid_dim_number

QUADS_MODE_LIST = ['static_same_goal', 'static_diff_goal', 'dynamic_same_goal', 'dynamic_diff_goal', 'circular_config', 'ep_lissajous3D',
                   'ep_rand_bezier', 'swarm_vs_swarm', 'dynamic_formations', 'swap_goals']

QUADS_MODE_DICT = {
            'fix_size': ['static_same_goal', 'dynamic_same_goal', 'ep_lissajous3D', 'ep_rand_bezier'],
            'dynamic_size': ['static_diff_goal', 'dynamic_diff_goal'],
            'swap_goals': ['swarm_vs_swarm', 'swap_goals', 'dynamic_formations', 'circular_config']
}

QUADS_FORMATION_LIST = ['circle_vertical_xz', 'circle_vertical_yz', 'circle_horizontal', 'sphere',
                        'grid_vertical_xz', 'grid_vertical_yz', 'grid_horizontal', 'cube']


def create_scenario(quads_mode, envs, num_agents, room_dims, room_dims_callback, rew_coeff, quads_formation, quads_formation_size):
    cls = eval('Scenario_' + quads_mode)
    scenario = cls(envs, num_agents, room_dims, room_dims_callback, rew_coeff, quads_formation, quads_formation_size)
    return scenario


class QuadrotorScenario:
    def __init__(self, envs, num_agents, room_dims, room_dims_callback, rew_coeff, quads_formation, quads_formation_size):
        self.envs = envs
        self.num_agents = num_agents
        self.room_dims = room_dims
        self.set_room_dims = room_dims_callback  # usage example: self.set_room_dims((10, 10, 10))
        self.rew_coeff = rew_coeff

        self.interp = None

        # Aux variables for goals of quadrotors
        quad_arm = self.envs[0].dynamics.arm
        self.formation = quads_formation
        # Aux for scalibility
        self.update_formation()

        self.formation_size = quads_formation_size
        lowest_dist, highest_dist = 8 * quad_arm, 16 * quad_arm
        self.lowest_formation_size, self.highest_formation_size = self.get_formation_range(mode=None, low=lowest_dist, high=highest_dist)

        self.formation_center = np.array([0.0, 0.0, 2.0])

        # Aux variables for settle, mainly for scenarios:
        # circular configuration && swarm vs swarm
        self.settle_count = np.zeros(self.num_agents)
        self.metric_of_settle = 2.0 * quad_arm

        # Generate goals
        self.goals = None

        # for 3D formations. Specific formations override this
        self.num_agents_per_layer = 8

    def name(self):
        """
        :return: scenario name
        """
        return self.__class__.__name__

    def get_formation_range(self, mode, low, high):
        if mode == 'swarm_vs_swarm':
            n = self.num_agents // 2
        else:
            n = self.num_agents

        if self.formation.startswith("circle"):
            formation_size_low = get_circle_radius(self.num_agents_per_layer, low)
            formation_size_high = get_circle_radius(self.num_agents_per_layer, high)
        elif self.formation.startswith("grid"):
            formation_size_low = low
            formation_size_high = high
        elif self.formation.startswith("sphere"):
            formation_size_low = get_sphere_radius(n, low)
            formation_size_high = get_sphere_radius(n, high)
        elif self.formation.startswith("cube"):
            formation_size_low = low
            formation_size_high = high
        else:
            raise NotImplementedError(f'{self.formation} is not supported!')

        return formation_size_low, formation_size_high

    def get_goal_by_formation(self, pos_0, pos_1, layer_pos=0.):
        if self.formation.endswith("horizontal"):
            goal = np.array([pos_0, pos_1, layer_pos])
        elif self.formation.endswith("vertical_xz"):
            goal = np.array([pos_0, layer_pos, pos_1])
        elif self.formation.endswith("vertical_yz"):
            goal = np.array([layer_pos, pos_0, pos_1])
        else:
            raise NotImplementedError("Unknown formation")

        return goal

    def generate_goals(self, num_agents, formation_center=None):
        if formation_center is None:
            formation_center = np.array([0., 0., 2.])

        if self.formation.startswith("circle"):
            if num_agents <= self.num_agents_per_layer:
                real_num_per_layer = [num_agents]
            else:
                whole_layer_num = num_agents // self.num_agents_per_layer
                real_num_per_layer = [self.num_agents_per_layer for _ in range(whole_layer_num)]
                rest_num = num_agents % self.num_agents_per_layer
                if rest_num > 0:
                    real_num_per_layer.append(rest_num)

            pi = np.pi
            goals = []
            layer_dist = np.random.uniform(low=self.lowest_formation_size, high=self.highest_formation_size)
            for i in range(num_agents):
                cur_layer_num_agents = real_num_per_layer[i // self.num_agents_per_layer]
                degree = 2 * pi * i / cur_layer_num_agents
                pos_0 = self.formation_size * np.cos(degree)
                pos_1 = self.formation_size * np.sin(degree)
                goal = self.get_goal_by_formation(pos_0, pos_1, layer_pos=(i//self.num_agents_per_layer) * layer_dist)
                goals.append(goal)

            goals = np.array(goals)
            goals += formation_center
        elif self.formation == "sphere":
            goals = self.formation_size * np.array(generate_points(num_agents)) + formation_center
        elif self.formation.startswith("grid"):
            if num_agents <= self.num_agents_per_layer:
                real_num_per_layer = [num_agents]
                dim_1, dim_2 = get_grid_dim_number(num_agents)
                dim_size_each_layer = [[dim_1, dim_2]]
            else:
                # whole layer
                whole_layer_num = num_agents // self.num_agents_per_layer
                max_dim_1, max_dim_2 = get_grid_dim_number(self.num_agents_per_layer)
                real_num_per_layer = [self.num_agents_per_layer for _ in range(whole_layer_num)]
                dim_size_each_layer = [[max_dim_1, max_dim_2] for _ in range(whole_layer_num)]

                # deal with the rest of the drones
                rest_num = num_agents % self.num_agents_per_layer
                if rest_num > 0:
                    real_num_per_layer.append(rest_num)
                    dim_1, dim_2 = get_grid_dim_number(rest_num)
                    dim_size_each_layer.append([dim_1, dim_2])

            goals = []
            layer_dist = np.random.uniform(low=self.lowest_formation_size, high=self.highest_formation_size)
            for i in range(num_agents):
                dim_1, dim_2 = dim_size_each_layer[i//self.num_agents_per_layer]
                pos_0 = self.formation_size * (i % dim_2)
                pos_1 = self.formation_size * (int(i / dim_2) % dim_1)
                goal = self.get_goal_by_formation(pos_0, pos_1, layer_pos=(i//self.num_agents_per_layer) * layer_dist)
                goals.append(goal)

            mean_pos = np.mean(goals, axis=0)
            goals = goals - mean_pos + formation_center
        elif self.formation.startswith("cube"):
            dim_size = np.power(num_agents, 1.0 / 3)
            floor_dim_size = int(dim_size)
            goals = []
            for i in range(num_agents):
                pos_0 = self.formation_size * (int(i / floor_dim_size) % floor_dim_size)
                pos_1 = self.formation_size * (i % floor_dim_size)
                goal = np.array([formation_center[2] + self.formation_size * (i // np.square(floor_dim_size)), pos_0, pos_1])
                goals.append(goal)

            mean_pos = np.mean(goals, axis=0)
            goals = goals - mean_pos + formation_center
        else:
            raise NotImplementedError("Unknown formation")

        return goals

    def update_formation_size(self, new_formation_size):
        if new_formation_size != self.formation_size:
            self.formation_size = new_formation_size if new_formation_size > 0.0 else 0.0
            self.goals = self.generate_goals(num_agents=self.num_agents, formation_center=self.formation_center)
            for i, env in enumerate(self.envs):
                env.goal = self.goals[i]

    def step(self, infos, rewards, pos):
        raise NotImplementedError("Implemented in a specific scenario")

    def reset(self):
        self.formation_size = max(0.0, self.formation_size)
        # Generate goals
        self.goals = self.generate_goals(num_agents=self.num_agents, formation_center=self.formation_center)

    def update_formation(self):
        formation_index = np.random.randint(low=0, high=len(QUADS_FORMATION_LIST))
        self.formation = QUADS_FORMATION_LIST[formation_index]

        # Aux for scalibility
        if self.formation.startswith("circle"):
            self.num_agents_per_layer = 8
        elif self.formation.startswith("grid"):
            self.num_agents_per_layer = 50


class Scenario_static_same_goal(QuadrotorScenario):
    def update_formation_size(self, new_formation_size):
        pass

    def step(self, infos, rewards, pos):
        return infos, rewards

    def reset(self):
        self.formation_size = 0.0
        self.goals = self.generate_goals(num_agents=self.num_agents, formation_center=self.formation_center)


class Scenario_static_diff_goal(QuadrotorScenario):
    def step(self, infos, rewards, pos):
        return infos, rewards


class Scenario_dynamic_same_goal(QuadrotorScenario):
    def __init__(self, envs, num_agents, room_dims, room_dims_callback, rew_coeff, quads_formation, quads_formation_size):
        super().__init__(envs, num_agents, room_dims, room_dims_callback, rew_coeff, quads_formation, quads_formation_size)
        # teleport every [4.0, 6.0] secs
        duration_time = 5.0
        self.control_step_for_sec = int(duration_time * self.envs[0].control_freq)

    def update_formation_size(self, new_formation_size):
        pass

    def step(self, infos, rewards, pos):
        tick = self.envs[0].tick
        if tick % self.control_step_for_sec == 0 and tick > 0:
            box_size = self.envs[0].box
            x, y = np.random.uniform(low=-box_size, high=box_size, size=(2,))
            z = np.random.uniform(low=-0.5 * box_size, high=0.5 * box_size) + 2.0
            z = max(0.25, z)
            self.formation_center = np.array([x, y, z])
            self.goals = self.generate_goals(num_agents=self.num_agents, formation_center=self.formation_center)
            for i, env in enumerate(self.envs):
                env.goal = self.goals[i]

        return infos, rewards

    def reset(self):
        # Update duration time
        duration_time = np.random.uniform(low=4.0, high=6.0)
        self.control_step_for_sec = int(duration_time * self.envs[0].control_freq)
        self.formation_size = 0.0
        # Generate goals
        self.goals = self.generate_goals(self.num_agents)


class Scenario_dynamic_diff_goal(QuadrotorScenario):
    def __init__(self, envs, num_agents, room_dims, room_dims_callback, rew_coeff, quads_formation, quads_formation_size):
        super().__init__(envs, num_agents, room_dims, room_dims_callback, rew_coeff, quads_formation, quads_formation_size)
        # teleport every [4.0, 6.0] secs
        duration_time = 5.0
        self.control_step_for_sec = int(duration_time * self.envs[0].control_freq)

    def update_goals(self):
        # reset formation
        self.update_formation()
        self.goals = self.generate_goals(num_agents=self.num_agents, formation_center=self.formation_center)
        np.random.shuffle(self.goals)
        for i, env in enumerate(self.envs):
            env.goal = self.goals[i]

    def step(self, infos, rewards, pos):
        tick = self.envs[0].tick
        if tick % self.control_step_for_sec == 0 and tick > 0:
            box_size = self.envs[0].box
            x, y = np.random.uniform(low=-box_size, high=box_size, size=(2,))
            z = np.random.uniform(low=-0.5 * box_size, high=0.5 * box_size) + 2.0
            z_lower_bound = 0.25
            if self.formation == "sphere" or self.formation.startswith("circle_vertical"):
                z_lower_bound = self.formation_size + 0.25
            elif self.formation.startswith("grid_vertical"):
                real_num_per_layer = np.minimum(self.num_agents, self.num_agents_per_layer)
                dim_1, _ = get_grid_dim_number(real_num_per_layer)
                z_lower_bound = dim_1 * self.formation_size + 0.25

            z = max(z_lower_bound, z)
            self.formation_center = np.array([x, y, z])
            self.update_goals()

        return infos, rewards

    def reset(self):
        # Update duration time
        duration_time = np.random.uniform(low=4.0, high=6.0)
        self.control_step_for_sec = int(duration_time * self.envs[0].control_freq)
        self.formation_size = max(0.0, self.formation_size)
        # Generate goals
        self.goals = self.generate_goals(num_agents=self.num_agents, formation_center=self.formation_center)

class Scenario_ep_lissajous3D(QuadrotorScenario):
    # Based on https://mathcurve.com/courbes3d.gb/lissajous3d/lissajous3d.shtml
    @staticmethod
    def lissajous3D(tick, a=0.03, b=0.01, c=0.01, n=2, m=2, phi=90, psi=90):
        x = a * np.sin(tick)
        y = b * np.sin(n * tick + phi)
        z = c * np.cos(m * tick + psi)
        return x, y, z

    def step(self, infos, rewards, pos):
        control_freq = self.envs[0].control_freq
        tick = self.envs[0].tick / control_freq
        x, y, z = self.lissajous3D(tick)
        goal_x, goal_y, goal_z = self.goals[0]
        x_new, y_new, z_new = x + goal_x, y + goal_y, z + goal_z
        self.goals = np.array([[x_new, y_new, z_new] for i in range(self.num_agents)])

        for i, env in enumerate(self.envs):
            env.goal = self.goals[i]

        return infos, rewards

    def update_formation_size(self, new_formation_size):
        pass

    def reset(self):
        self.formation_size = 0.0
        # Generate goals
        formation_center = np.array([-2.0, 0.0, 2.0])  # prevent drones from crashing into the wall
        self.goals = self.generate_goals(self.num_agents, formation_center)


class Scenario_ep_rand_bezier(QuadrotorScenario):
    def step(self, infos, rewards, pos):
        # randomly sample new goal pos in free space and have the goal move there following a bezier curve
        tick = self.envs[0].tick
        control_freq = self.envs[0].control_freq
        num_secs = 5
        control_steps = int(num_secs * control_freq)
        t = tick % control_steps
        room_dims = np.array(self.room_dims) - self.formation_size
        # min and max distance the goal can spawn away from its current location. 30 = empirical upper bound on
        # velocity that the drones can handle.
        max_dist = min(30, max(room_dims))
        min_dist = max_dist / 2
        if tick % control_steps == 0 or tick == 1:
            # sample a new goal pos that's within the room boundaries and satisfies the distance constraint
            new_goal_found = False
            while not new_goal_found:
                low, high = np.array([-room_dims[0] / 2, -room_dims[1] / 2, 0]), np.array(
                    [room_dims[0] / 2, room_dims[1] / 2, room_dims[2]])
                # need an intermediate point for a deg=2 curve
                new_pos = np.random.uniform(low=-high, high=high, size=(2, 3)).reshape(3, 2)
                # add some velocity randomization = random magnitude * unit direction
                new_pos = new_pos * np.random.randint(min_dist, max_dist + 1) / np.linalg.norm(new_pos, axis=0)
                new_pos = self.goals[0].reshape(3, 1) + new_pos
                lower_bound = np.expand_dims(low, axis=1)
                upper_bound = np.expand_dims(high, axis=1)
                new_goal_found = (new_pos > lower_bound + 0.5).all() and (
                        new_pos < upper_bound - 0.5).all()  # check bounds that are slightly smaller than the room dims
            nodes = np.concatenate((self.goals[0].reshape(3, 1), new_pos), axis=1)
            nodes = np.asfortranarray(nodes)
            pts = np.linspace(0, 1, control_steps)
            curve = bezier.Curve(nodes, degree=2)
            self.interp = curve.evaluate_multi(pts)
            # self.interp = np.clip(self.interp, a_min=np.array([0,0,0.2]).reshape(3,1), a_max=high.reshape(3,1)) # want goal clipping to be slightly above the floor
        if tick % control_steps != 0 and tick > 1:
            self.goals = np.array([self.interp[:, t] for _ in range(self.num_agents)])

            for i, env in enumerate(self.envs):
                env.goal = self.goals[i]

        return infos, rewards

    def update_formation_size(self, new_formation_size):
        pass

    def reset(self):
        self.formation_size = 0.0
        # Generate goals
        self.goals = self.generate_goals(self.num_agents)

class Scenario_swap_goals(QuadrotorScenario):
    def __init__(self, envs, num_agents, room_dims, room_dims_callback, rew_coeff, quads_formation, quads_formation_size):
        super().__init__(envs, num_agents, room_dims, room_dims_callback, rew_coeff, quads_formation, quads_formation_size)
        # teleport every [4.0, 6.0] secs
        duration_time = 5.0
        self.control_step_for_sec = int(duration_time * self.envs[0].control_freq)

    def update_goals(self):
        np.random.shuffle(self.goals)
        for env, goal in zip(self.envs, self.goals):
            env.goal = goal

    def step(self, infos, rewards, pos):
        tick = self.envs[0].tick
        # Switch every [4, 6] seconds
        if tick % self.control_step_for_sec == 0 and tick > 0:
            self.update_goals()

        return infos, rewards

    def reset(self):
        # Update duration time
        duration_time = np.random.uniform(low=4.0, high=6.0)
        self.control_step_for_sec = int(duration_time * self.envs[0].control_freq)
        self.formation_size = max(0.0, self.formation_size)
        # Generate goals
        self.goals = self.generate_goals(num_agents=self.num_agents, formation_center=self.formation_center)

class Scenario_circular_config(QuadrotorScenario):
    def update_goals(self):
        np.random.shuffle(self.goals)
        for env, goal in zip(self.envs, self.goals):
            env.goal = goal

    def step(self, infos, rewards, pos):
        for i, e in enumerate(self.envs):
            dist = np.linalg.norm(pos[i] - e.goal)
            if abs(dist) < self.metric_of_settle:
                self.settle_count[i] += 1
            else:
                self.settle_count = np.zeros(self.num_agents)
                break

        # drones settled at the goal for 1 sec
        control_step_for_one_sec = int(self.envs[0].control_freq)
        tmp_count = self.settle_count >= control_step_for_one_sec
        if all(tmp_count):
            self.update_goals()
            rews_settle_raw = control_step_for_one_sec
            rews_settle = self.rew_coeff["quadsettle"] * rews_settle_raw
            for i, env in enumerate(self.envs):
                env.goal = self.goals[i]
                # Add settle rewards
                rewards[i] += rews_settle
                infos[i]["rewards"]["rew_quadsettle"] = rews_settle
                infos[i]["rewards"]["rewraw_quadsettle"] = rews_settle_raw

            self.settle_count = np.zeros(self.num_agents)

        return infos, rewards


class Scenario_dynamic_formations(QuadrotorScenario):
    def __init__(self, envs, num_agents, room_dims, room_dims_callback, rew_coeff, quads_formation, quads_formation_size):
        super().__init__(envs, num_agents, room_dims, room_dims_callback, rew_coeff, quads_formation, quads_formation_size)
        # if increase_formation_size is True, increase the formation size
        # else, decrease the formation size
        self.increase_formation_size = True
        # low = 0.001: change 0.2m/s, high = 0.01: change 2.0m/s
        self.control_speed = np.random.uniform(low=1.0, high=10.0)

    # change formation sizes on the fly
    def update_goals(self):
        self.goals = self.generate_goals(self.num_agents, self.formation_center)
        for env, goal in zip(self.envs, self.goals):
            env.goal = goal

    def step(self, infos, rewards, pos):
        if self.formation_size <= -self.highest_formation_size:
            self.increase_formation_size = True
            self.control_speed = np.random.uniform(low=2.0, high=6.0)
        elif self.formation_size >= self.highest_formation_size:
            self.increase_formation_size = False
            self.control_speed = np.random.uniform(low=2.0, high=6.0)

        if self.increase_formation_size:
            self.formation_size += 0.001 * self.control_speed
        else:
            self.formation_size -= 0.001 * self.control_speed

        self.update_goals()
        return infos, rewards

    def reset(self):
        self.formation_size = np.random.uniform(low=self.lowest_formation_size, high=self.highest_formation_size)
        self.increase_formation_size = True if np.random.uniform(low=0.0, high=1.0) < 0.5 else False
        self.control_speed = np.random.uniform(low=1.0, high=10.0)
        # Generate goals
        self.goals = self.generate_goals(num_agents=self.num_agents, formation_center=self.formation_center)

    def update_formation_size(self, new_formation_size):
        if new_formation_size != self.formation_size:
            self.formation_size = new_formation_size if new_formation_size > 0.0 else 0.0
            self.update_goals()


class Scenario_swarm_vs_swarm(QuadrotorScenario):
    def __init__(self, envs, num_agents, room_dims, room_dims_callback, rew_coeff, quads_formation, quads_formation_size):
        super().__init__(envs, num_agents, room_dims, room_dims_callback, rew_coeff, quads_formation, quads_formation_size)
        # teleport every [4.0, 6.0] secs
        duration_time = 5.0
        self.control_step_for_sec = int(duration_time * self.envs[0].control_freq)
        quad_arm = self.envs[0].dynamics.arm
        lowest_dist, highest_dist = 8 * quad_arm, 16 * quad_arm
        self.lowest_formation_size, self.highest_formation_size = \
            self.get_formation_range(mode='swarm_vs_swarm', low=lowest_dist, high=highest_dist)

    def formation_centers(self):
        if self.formation_center is None:
            self.formation_center = np.array([0., 0., 2.])

        # self.envs[0].box = 2.0
        box_size = self.envs[0].box
        dist_low_bound = self.lowest_formation_size
        # Get the 1st goal center
        x, y = np.random.uniform(low=-box_size, high=box_size, size=(2,)) + self.formation_center[:2]
        z = np.random.uniform(low=-0.5 * box_size, high=0.5 * box_size) + self.formation_center[2]
        goal_center_1 = np.array([x, y, z])

        # Get the 2nd goal center
        goal_center_distance = np.random.uniform(low=box_size/4, high=box_size)

        phi = np.random.uniform(low=-np.pi, high=np.pi)
        theta = np.random.uniform(low=-0.5 * np.pi, high=0.5 * np.pi)
        goal_center_2 = goal_center_1 + goal_center_distance * np.array(
            [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
        diff_x, diff_y, diff_z = goal_center_2 - goal_center_1
        if self.formation.endswith("horizontal"):
            if abs(diff_z) < dist_low_bound:
                goal_center_2[2] = np.sign(diff_z) * dist_low_bound + goal_center_1[2]
        elif self.formation.endswith("vertical_xz"):
            if abs(diff_y) < dist_low_bound:
                goal_center_2[1] = np.sign(diff_y) * dist_low_bound + goal_center_1[1]
        elif self.formation.endswith("vertical_xz"):
            if abs(diff_x) < dist_low_bound:
                goal_center_2[0] = np.sign(diff_x) * dist_low_bound + goal_center_1[0]

        return goal_center_1, goal_center_2

    def create_formations(self, goal_center_1, goal_center_2):
        self.goals_1 = self.generate_goals(self.num_agents // 2, goal_center_1)
        self.goals_2 = self.generate_goals(self.num_agents - self.num_agents // 2, goal_center_2)
        self.goals = np.concatenate([self.goals_1, self.goals_2])

    def update_goals(self):
        tmp_goal_center_1 = copy.deepcopy(self.goal_center_1)
        tmp_goal_center_2 = copy.deepcopy(self.goal_center_2)
        self.goal_center_1 = tmp_goal_center_2
        self.goal_center_2 = tmp_goal_center_1

        self.update_formation()
        self.create_formations(self.goal_center_1, self.goal_center_2)
        # Shuffle goals
        np.random.shuffle(self.goals_1)
        np.random.shuffle(self.goals_2)
        self.goals = np.concatenate([self.goals_1, self.goals_2])
        for i, env in enumerate(self.envs):
            env.goal = self.goals[i]

    def step(self, infos, rewards, pos):
        tick = self.envs[0].tick
        # Switch every [4, 6] seconds
        if tick % self.control_step_for_sec == 0 and tick > 0:
            self.update_goals()
        return infos, rewards

    def reset(self):
        # Reset the formation size and the goals of swarms
        self.goal_center_1, self.goal_center_2 = self.formation_centers()
        self.create_formations(self.goal_center_1, self.goal_center_2)
        duration_time = np.random.uniform(low=4.0, high=6.0)
        self.control_step_for_sec = int(duration_time * self.envs[0].control_freq)

    def update_formation_size(self, new_formation_size):
        if new_formation_size != self.formation_size:
            self.formation_size = new_formation_size if new_formation_size > 0.0 else 0.0
            self.create_formations(self.goal_center_1, self.goal_center_2)
            for i, env in enumerate(self.envs):
                env.goal = self.goals[i]


class Scenario_tunnel(QuadrotorScenario):

    def update_goals(self, formation_center):
        self.goals = self.generate_goals(self.num_agents, formation_center)

    def step(self, infos, rewards, pos):
        # hack to make drones and goals be on opposite sides of the tunnel
        t = self.envs[0].tick
        if t == 1:
            for env in self.envs:
                if abs(env.goal[0]) > abs(env.goal[1]):
                    env.goal[0] = -env.goal[0]
                else:
                    env.goal[1] = -env.goal[1]
        return infos, rewards

    def reset(self):
        # tunnel could be in the x or y direction
        p = np.random.uniform(0, 1)
        if p <= 0.5:
            self.update_room_dims((10, 2, 2))
            formation_center = np.array([-4, 0, 1])
        else:
            self.update_room_dims((2, 10, 2))
            formation_center = np.array([0, -4, 1])
        self.update_goals(formation_center)


class Scenario_mix(QuadrotorScenario):
    def __init__(self, envs, num_agents, room_dims, room_dims_callback, rew_coeff, quads_formation, quads_formation_size):
        super().__init__(envs, num_agents, room_dims, room_dims_callback, rew_coeff, quads_formation, quads_formation_size)
        quad_arm_size = self.envs[0].dynamics.arm # 4.6 centimeters
        str_no_obstacles = "no_obstacles"
        str_dynamic_obstacles = "dynamic"
        self.obstacle_number = self.envs[0].obstacle_num
        self.room_dims_callback = room_dims_callback

        if self.envs[0].obstacle_mode == "no_obstacles":
            str_dynamic_obstacles = "no_obstacles"
            self.obstacle_number = 0

        # key: quads_mode
        # value: 0. formation, 1: [formation_low_size, formation_high_size], 2: episode_time, 3: obstacle_mode
        self.quads_formation_and_size_dict = {
            "fix_size": {
                "static_same_goal": [["circle_horizontal"], [0.0, 0.0], 8.0, str_dynamic_obstacles],
                "dynamic_same_goal": [["circle_horizontal"], [0.0, 0.0], 12.0, str_no_obstacles],
                "ep_lissajous3D": [["circle_horizontal"], [0.0, 0.0], 12.0, str_no_obstacles],
                "ep_rand_bezier": [["circle_horizontal"], [0.0, 0.0], 12.0, str_no_obstacles],
            },
            "dynamic_size": {
                "static_diff_goal": [QUADS_FORMATION_LIST, [8 * quad_arm_size, 16 * quad_arm_size], 8.0, str_dynamic_obstacles],  # [36, 72] centimeters
                "dynamic_diff_goal": [QUADS_FORMATION_LIST, [8 * quad_arm_size, 16 * quad_arm_size], 12.0, str_no_obstacles],  # [36, 72] centimeters
            },
            "swap_goals":{
                "swarm_vs_swarm": [QUADS_FORMATION_LIST, [8 * quad_arm_size, 16 * quad_arm_size], 16.0, str_no_obstacles],
                "swap_goals": [QUADS_FORMATION_LIST, [8 * quad_arm_size, 16 * quad_arm_size], 16.0, str_no_obstacles],
                "dynamic_formations": [QUADS_FORMATION_LIST, [0.0, 20 * quad_arm_size], 16.0, str_dynamic_obstacles],
                "circular_config": [QUADS_FORMATION_LIST, [8 * quad_arm_size, 16 * quad_arm_size], 16.0, str_no_obstacles],
            }
        }

        # actual scenario being used
        self.scenario = None

    def name(self):
        """
        :return: the name of the actual scenario used in this episode
        """
        return self.scenario.__class__.__name__

    def step(self, infos, rewards, pos):
        infos, rewards = self.scenario.step(infos=infos, rewards=rewards, pos=pos)
        return infos, rewards

    def reset(self):
        # reset mode
        mode_dict_prob = np.random.uniform(low=0, high=1)
        if mode_dict_prob <= 0.2:
            mode_dict = QUADS_MODE_DICT["fix_size"]
        elif 0.2 < mode_dict_prob <= 0.3:
            mode_dict = QUADS_MODE_DICT["dynamic_size"]
        else:
            mode_dict = QUADS_MODE_DICT["swap_goals"]

        mode_index = np.random.randint(low=0, high=len(mode_dict))
        mode = mode_dict[mode_index]

        if mode in self.quads_formation_and_size_dict["fix_size"]:
            quads_dict = self.quads_formation_and_size_dict["fix_size"]
            # reset formation
            self.formation = QUADS_FORMATION_LIST[0]
            # reset formation size
            self.formation_size = quads_dict[mode][1][0]
            formation_size_low, formation_size_high = 0.0, 0.0
        elif mode in self.quads_formation_and_size_dict["dynamic_size"] or mode in self.quads_formation_and_size_dict["swap_goals"]:
            if mode in self.quads_formation_and_size_dict["dynamic_size"]:
                quads_dict = self.quads_formation_and_size_dict["dynamic_size"]
            else:
                quads_dict = self.quads_formation_and_size_dict["swap_goals"]
            # reset formation
            formation_index = np.random.randint(low=0, high=len(quads_dict[mode][0]))
            self.formation = QUADS_FORMATION_LIST[formation_index]
            # Aux for scalibility
            if self.formation.startswith("circle"):
                self.num_agents_per_layer = 8
            elif self.formation.startswith("grid"):
                self.num_agents_per_layer = 50

            # reset formation size
            lowest_dist, highest_dist = quads_dict[mode][1]
            formation_size_low, formation_size_high = self.get_formation_range(mode=mode, low=lowest_dist, high=highest_dist)
            self.formation_size = np.random.uniform(low=formation_size_low, high=formation_size_high)
        else:
            raise NotImplementedError(f'{mode} is not supported!')

        # init the scenario
        self.scenario = create_scenario(quads_mode=mode, envs=self.envs, num_agents=self.num_agents,
                                        room_dims=self.room_dims, room_dims_callback=self.room_dims_callback, rew_coeff=self.rew_coeff,
                                        quads_formation=self.formation, quads_formation_size=self.formation_size)

        self.scenario.lowest_formation_size = formation_size_low
        self.scenario.highest_formation_size = formation_size_high

        self.scenario.reset()
        self.goals = self.scenario.goals
        for env in self.envs:
            # reset episode time
            ep_time = quads_dict[mode][2]
            env.reset_ep_len(ep_time=ep_time)
            # reset obstacle mode and number
            obstacle_mode = quads_dict[mode][3]
            env.reset_obstacle_mode(obstacle_mode=obstacle_mode, obstacle_num=self.obstacle_number)
