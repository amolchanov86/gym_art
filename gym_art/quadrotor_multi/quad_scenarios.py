import numpy as np
import random
import bezier
import copy

from gym_art.quadrotor_multi.quad_utils import generate_points

QUADS_MODE_LIST = ['static_same_goal', 'static_diff_goal', 'dynamic_same_goal', 'dynamic_diff_goal',
                   'circular_config', 'ep_lissajous3D', 'ep_rand_bezier', 'swarm_vs_swarm', 'dynamic_formations']

QUADS_FORMATION_LIST = ['circle_xz_vertical', 'circle_yz_vertical', 'circle_horizontal', 'sphere',
                        'grid_xz_vertical', 'grid_yz_vertical', 'grid_horizontal']


def create_scenario(quads_mode, envs, num_agents, room_dims, rew_coeff, quads_formation, quads_formation_size):
    cls = eval('Scenario_' + quads_mode)
    scenario = cls(envs, num_agents, room_dims, rew_coeff, quads_formation, quads_formation_size)
    return scenario


class QuadrotorScenario:
    def __init__(self, envs, num_agents, room_dims, rew_coeff, quads_formation, quads_formation_size):
        self.envs = envs
        self.num_agents = num_agents
        self.room_dims = room_dims
        self.rew_coeff = rew_coeff

        self.interp = None

        # Aux variables for goals of quadrotors
        self.formation = quads_formation
        self._formation_size = quads_formation_size
        self._lowest_formation_size, self._highest_formation_size = self.init_formation_sizes()
        self.formation_center = np.array([0.0, 0.0, 2.0])

        # Aux variables for settle, mainly for scenarios:
        # circular configuration && swarm vs swarm
        self.settle_count = np.zeros(self.num_agents)
        self.metric_of_settle = self.envs[0].dynamics.arm

        # Generate goals
        self.goals = None

    @property
    def lowest_formation_size(self):
        return self._lowest_formation_size

    @lowest_formation_size.setter
    def lowest_formation_size(self, lfs):
        self._lowest_formation_size = lfs

    @property
    def highest_formation_size(self):
        return self._highest_formation_size

    @highest_formation_size.setter
    def highest_formation_size(self, hfs):
        self._highest_formation_size = hfs

    def init_formation_sizes(self, split=False):
        if split:  # divide num agents by 2 for swarm_vs_swarm type scenarios
            num_agents = self.num_agents // 2
        else:
            num_agents = self.num_agents
        # The highest formation size means the formation size that we set, which can control
        # the distance between the goals of any two quadrotors should be large than 12.0 * quads_arm_size
        quad_arm_size = self.envs[0].dynamics.arm  # arm length: 4.6 centimeters
        highest_formation_size = 12.0 * quad_arm_size * np.sin(np.pi / 2 - np.pi / num_agents) / np.sin(
            2 * np.pi / num_agents)

        # The lowest formation size means the formation size that we set, which can control
        # the distance between the goals of any two quadrotors should be large than 4.0 * quads_arm_size
        quad_arm_size = self.envs[0].dynamics.arm  # arm length: 4.6 centimeters
        lowest_formation_size = 4.0 * quad_arm_size * np.sin(np.pi / 2 - np.pi / num_agents) / np.sin(
            2 * np.pi / num_agents)
        return lowest_formation_size, highest_formation_size

    @property
    def formation_size(self):
        return self._formation_size

    @formation_size.setter
    def formation_size(self, fs):
        self._formation_size = fs

    def get_goal_by_formation(self, pos_0, pos_1):
        if self.formation.endswith("horizontal"):
            goal = np.array([pos_0, pos_1, 0.0])
        elif self.formation.endswith("xz_vertical"):
            goal = np.array([pos_0, 0.0, pos_1])
        elif self.formation.endswith("yz_vertical"):
            goal = np.array([0.0, pos_0, pos_1])
        else:
            raise NotImplementedError("Unknown formation")

        return goal

    def generate_goals(self, num_agents, formation_center=None):
        if formation_center is None:
            formation_center = np.array([0., 0., 2.])

        if self.formation.startswith("circle"):
            pi = np.pi
            goals = []
            for i in range(num_agents):
                degree = 2 * pi * i / num_agents
                pos_0 = self.formation_size * np.cos(degree)
                pos_1 = self.formation_size * np.sin(degree)
                goal = self.get_goal_by_formation(pos_0, pos_1)
                goals.append(goal)

            goals = np.array(goals)
            goals += formation_center
        elif self.formation == "sphere":
            goals = self.formation_size * np.array(generate_points(num_agents)) + formation_center
        elif self.formation.startswith("grid"):
            sqrt_goal_num = np.sqrt(num_agents)
            grid_number = int(np.ceil(sqrt_goal_num))

            goals = []
            for i in range(num_agents):
                pos_0 = self.formation_size * int(i / grid_number)
                pos_1 = self.formation_size * (i % grid_number)
                goal = self.get_goal_by_formation(pos_0, pos_1)
                goals.append(goal)

            goals = np.array(goals)
            mean_pos = np.mean(goals, axis=0)
            goals -= mean_pos
            goals += formation_center
            goals = np.array(goals)
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


class QuadrotorScenario_Dynamic_Goal(QuadrotorScenario):
    def step(self, infos, rewards, pos):
        tick = self.envs[0].tick
        # teleport every 5 secs
        control_step_for_five_sec = int(5.0 * self.envs[0].control_freq)
        if tick % control_step_for_five_sec == 0 and tick > 0:
            box_size = self.envs[0].box
            x, y = np.random.uniform(low=-box_size, high=box_size, size=(2,))
            z = np.random.uniform(low=-0.5 * box_size, high=0.5 * box_size) + 2.0
            z_lower_bound = 0.25
            if self.formation == "sphere":
                z_lower_bound = self.formation_size + 0.25
            elif self.formation == "grid_horizontal":
                z_lower_bound = np.ceil(np.sqrt(self.num_agents)) * self.formation_size + 0.25

            z = max(z_lower_bound, z)
            self.formation_center = np.array([x, y, z])
            self.goals = self.generate_goals(num_agents=self.num_agents, formation_center=self.formation_center)
            for i, env in enumerate(self.envs):
                env.goal = self.goals[i]

        return infos, rewards


# Inherent from QuadrotorScenario_Dynamic_Goal
class Scenario_dynamic_same_goal(QuadrotorScenario_Dynamic_Goal):
    # TODO: Maybe try increasing the difficuly by changing the pos of formation_center
    def future_func(self):
        pass

    def update_formation_size(self, new_formation_size):
        pass

    def reset(self):
        self.formation_size = 0.0
        # Generate goals
        self.goals = self.generate_goals(self.num_agents)


# Inherent from QuadrotorScenario_Dynamic_Goal
class Scenario_dynamic_diff_goal(QuadrotorScenario_Dynamic_Goal):
    # TODO: Maybe try increasing the difficuly by changing the pos of formation_center
    def future_func(self):
        pass


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
    def __init__(self, envs, num_agents, room_dims, rew_coeff, quads_formation, quads_formation_size):
        super().__init__(envs, num_agents, room_dims, rew_coeff, quads_formation, quads_formation_size)
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
        if self.formation_size <= self.lowest_formation_size:
            self.increase_formation_size = True
            self.control_speed = np.random.uniform(low=1.0, high=10.0)
        elif self.formation_size >= self.highest_formation_size:
            self.increase_formation_size = False
            self.control_speed = np.random.uniform(low=1.0, high=10.0)

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
    def formation_centers(self):
        if self.formation_center is None:
            self.formation_center = np.array([0., 0., 2.])

        # self.envs[0].box = 2.0
        box_size = self.envs[0].box
        dist_low_bound = 4 * self.envs[0].dynamics.arm
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
        elif self.formation.endswith("xz_vertical"):
            if abs(diff_y) < dist_low_bound:
                goal_center_2[1] = np.sign(diff_y) * dist_low_bound + goal_center_1[1]
        elif self.formation.endswith("yz_vertical"):
            if abs(diff_x) < dist_low_bound:
                goal_center_2[0] = np.sign(diff_x) * dist_low_bound + goal_center_1[0]

        return goal_center_1, goal_center_2

    def create_formations(self, goal_center_1, goal_center_2):
        self.goals_1 = self.generate_goals(self.num_agents // 2, goal_center_1)
        self.goals_2 = self.generate_goals(self.num_agents - self.num_agents // 2, goal_center_2)
        self.goals = np.concatenate([self.goals_1, self.goals_2])

    def update_goals(self):
        # Switch goals
        tmp_goals_1 = copy.deepcopy(self.goals_1)
        tmp_goals_2 = copy.deepcopy(self.goals_2)
        self.goals_1 = tmp_goals_2
        self.goals_2 = tmp_goals_1
        self.goals = np.concatenate([self.goals_1, self.goals_2])
        for i, env in enumerate(self.envs):
            env.goal = self.goals[i]

    def step(self, infos, rewards, pos):
        tick = self.envs[0].tick
        control_step_for_eight_sec = int(8 * self.envs[0].control_freq)
        # Switch every 8th second
        if tick % control_step_for_eight_sec == 0 and tick > 0:
            self.update_goals()
        return infos, rewards

    def reset(self):
        # Reset the formation size and the goals of swarms
        self.goal_center_1, self.goal_center_2 = self.formation_centers()
        self.create_formations(self.goal_center_1, self.goal_center_2)

    def update_formation_size(self, new_formation_size):
        if new_formation_size != self.formation_size:
            self.formation_size = new_formation_size if new_formation_size > 0.0 else 0.0
            self.create_formations(self.goal_center_1, self.goal_center_2)
            for i, env in enumerate(self.envs):
                env.goal = self.goals[i]


class Scenario_mix(QuadrotorScenario):
    def __init__(self, envs, num_agents, room_dims, rew_coeff, quads_formation, quads_formation_size):
        super().__init__(envs, num_agents, room_dims, rew_coeff, quads_formation, quads_formation_size)
        quad_arm_size = self.envs[0].dynamics.arm # 4.6 centimeters
        self.swarm_lowest_formation_size, self.swarm_highest_formation_size = self.init_formation_sizes(split=True)
        str_no_obstacles = "no_obstacles"
        str_dynamic_obstacles = "dynamic"
        self.obstacle_number = self.envs[0].obstacle_num

        if self.envs[0].obstacle_mode == "no_obstacles":
            str_dynamic_obstacles = "no_obstacles"
            self.obstacle_number = 0

        # key: quads_mode
        # value: 0. formation, 1: [formation_low_size, formation_high_size], 2: episode_time, 3: obstacle_mode
        self.quads_formation_and_size_dict = {
            "static_same_goal": [["circle_horizontal"], [0.0, 0.0], 16.0, str_dynamic_obstacles],
            "dynamic_same_goal": [["circle_horizontal"], [0.0, 0.0], 16.0, str_no_obstacles],
            "static_diff_goal": [QUADS_FORMATION_LIST, [5 * quad_arm_size, 10 * quad_arm_size], 16.0, str_dynamic_obstacles], # [23, 46] centimeters
            "dynamic_diff_goal": [QUADS_FORMATION_LIST, [5 * quad_arm_size, 10 * quad_arm_size], 16.0, str_no_obstacles], # [23, 46] centimeters
            "ep_lissajous3D": [["circle_horizontal"], [0.0, 0.0], 16.0, str_no_obstacles],
            "ep_rand_bezier": [["circle_horizontal"], [0.0, 0.0], 16.0, str_no_obstacles],
            "circular_config": [QUADS_FORMATION_LIST, [self._lowest_formation_size, self._highest_formation_size], 16.0, str_no_obstacles],
            "swarm_vs_swarm": [QUADS_FORMATION_LIST, [self.swarm_lowest_formation_size, self.swarm_highest_formation_size], 16.0, str_no_obstacles],
            "dynamic_formations": [QUADS_FORMATION_LIST, [4 * quad_arm_size, 30 * quad_arm_size], 16.0, str_dynamic_obstacles]
        }

    def step(self, infos, rewards, pos):
        infos, rewards = self.scenario.step(infos=infos, rewards=rewards, pos=pos)
        return infos, rewards

    def reset(self):
        # reset mode
        mode_index = round(np.random.uniform(low=0, high=len(QUADS_MODE_LIST)-1))
        mode = QUADS_MODE_LIST[mode_index]

        # reset formation
        formation_index = round(np.random.uniform(low=0, high=len(self.quads_formation_and_size_dict[mode][0])-1))
        self.formation = QUADS_FORMATION_LIST[formation_index]
        # reset formation size
        formation_size_low, formation_size_high = self.quads_formation_and_size_dict[mode][1]
        self.formation_size = np.random.uniform(low=formation_size_low, high=formation_size_high)

        # init the scenario
        self.scenario = create_scenario(quads_mode=mode, envs=self.envs, num_agents=self.num_agents,
                                        room_dims=self.room_dims, rew_coeff=self.rew_coeff,
                                        quads_formation=self.formation, quads_formation_size=self.formation_size)

        self.scenario.lowest_formation_size = formation_size_low
        self.scenario.highest_formation_size = formation_size_high

        self.scenario.reset()
        self.goals = self.scenario.goals
        for env in self.envs:
            # reset episode time
            env.reset_ep_len(self.quads_formation_and_size_dict[mode][2])
            # reset obstacle mode and number
            obstacle_mode = self.quads_formation_and_size_dict[mode][3]
            env.reset_obstacle_mode(obstacle_mode=obstacle_mode, obstacle_num=self.obstacle_number)
