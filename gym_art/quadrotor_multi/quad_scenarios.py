import numpy as np
import random
import bezier

from gym_art.quadrotor_multi.quad_utils import generate_points


def create_scenario(quads_mode, envs, pos, settle_count, num_agents, room_dims,
                    rews_settle_raw, rews_settle, rew_coeff, quads_formation, quads_formation_size):
    cls = eval('Scenario_' + quads_mode)
    scenario = cls(envs, pos, settle_count, num_agents, room_dims, rews_settle_raw, rews_settle, rew_coeff, quads_formation, quads_formation_size)
    return scenario


class QuadrotorScenario:
    def __init__(self, envs, pos, settle_count, num_agents, room_dims,
                 rews_settle_raw, rews_settle, rew_coeff, quads_formation, quads_formation_size):
        self.envs = envs
        self.pos = pos
        self.settle_count = settle_count
        self.num_agents = num_agents
        self.room_dims = room_dims
        self.rews_settle_raw = rews_settle_raw
        self.rews_settle = rews_settle
        self.rew_coeff = rew_coeff

        self.interp = None

        self.cfg_quads_formation = quads_formation
        self.cfg_quads_formation_size = quads_formation_size

        self.formation = self.formation_size = None

        self.goals = None

    def get_formation(self):
        return "circle"

    def get_formation_size(self):
        return 0.0

    def setup_formation(self):
        if self.cfg_quads_formation == "default":
            self.formation = self.get_formation()
        else:
            self.formation = self.cfg_quads_formation

        if self.cfg_quads_formation_size < 0:
            self.formation_size = self.get_formation_size()
        else:
            self.formation_size = self.cfg_quads_formation_size

    def init_goals(self):
        self.setup_formation()

        pi = np.pi

        goals = []

        if self.formation == "circle":
            for i in range(self.num_agents):
                degree = 2 * pi * i / self.num_agents
                goal_x = self.formation_size * np.cos(degree)
                goal_y = self.formation_size * np.sin(degree)
                goal = [goal_x, goal_y, 2.0]
                goals.append(goal)

            goals = np.array(goals)
        elif self.formation == "sphere":
            goals = self.formation_size * np.array(generate_points(self.num_agents))
            goals[:, 2] += 2.0
        else:
            raise NotImplementedError("Unknown formation")

        self.goals = goals

    def step(self, infos, rewards):
        raise NotImplementedError("Implemented in a specific scenario")


class Scenario_circular_config(QuadrotorScenario):
    def get_formation_size(self):
        return 0.5

    def step(self, infos, rewards):
        for i, e in enumerate(self.envs):
            dist = np.linalg.norm(self.pos[i] - e.goal)
            if abs(dist) < 0.05:
                self.settle_count[i] += 1
            else:
                self.settle_count = np.zeros(self.num_agents)

        # drones settled at the goal for 1 sec
        control_step_for_one_sec = int(self.envs[0].control_freq)
        tmp_count = self.settle_count >= control_step_for_one_sec
        if all(tmp_count):
            np.random.shuffle(self.goals)
            for i, env in enumerate(self.envs):
                env.goal = self.goals[i]
                # Add settle rewards
                self.rews_settle_raw[i] = control_step_for_one_sec
                self.rews_settle[i] = self.rew_coeff["quadsettle"] * self.rews_settle_raw[i]
                rewards[i] += self.rews_settle[i]
                infos[i]["rewards"]["rew_quadsettle"] = self.rews_settle[i]
                infos[i]["rewards"]["rewraw_quadsettle"] = self.rews_settle_raw[i]

            self.rews_settle = np.zeros(self.num_agents)
            self.rews_settle_raw = np.zeros(self.num_agents)
            self.settle_count = np.zeros(self.num_agents)


class Scenario_static_goal(QuadrotorScenario):
    def get_formation(self):
        return 'circle'

    def get_formation_size(self):
        return 0.0

    def step(self, infos, rewards):
        pass


class Scenario_dynamic_goal(QuadrotorScenario):
    def get_formation(self):
        return 'circle'

    def get_formation_size(self):
        return 0.0

    def step(self, infos, rewards):
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

            goal = [[x, y, z] for i in range(self.num_agents)]
            self.goals = np.array(goal)

            for i, env in enumerate(self.envs):
                env.goal = self.goals[i]


class Scenario_ep_lissajous3D(QuadrotorScenario):
    def get_formation(self):
        return 'circle'

    def get_formation_size(self):
        return 0.0

    # Based on https://mathcurve.com/courbes3d.gb/lissajous3d/lissajous3d.shtml
    @staticmethod
    def lissajous3D(tick, a=0.03, b=0.01, c=0.01, n=2, m=2, phi=90, psi=90):
        x = a * np.sin(tick)
        y = b * np.sin(n * tick + phi)
        z = c * np.cos(m * tick + psi)
        return x, y, z

    def step(self, infos, rewards):
        control_freq = self.envs[0].control_freq
        tick = self.envs[0].tick / control_freq
        x, y, z = self.lissajous3D(tick)
        goal_x, goal_y, goal_z = self.goals[0][0], self.goals[0][1], self.goals[0][2]
        x_new, y_new, z_new = x + goal_x, y + goal_y, z + goal_z
        goal = [[x_new, y_new, z_new] for i in range(self.num_agents)]
        goal = np.array(goal)

        for i, env in enumerate(self.envs):
            env.goal = goal[i]


class Scenario_ep_rand_bezier(QuadrotorScenario):
    def get_formation(self):
        return 'circle'

    def get_formation_size(self):
        return 0.0

    def step(self, infos, rewards):
        # randomly sample new goal pos in free space and have the goal move there following a bezier curve
        tick = self.envs[0].tick
        control_freq = self.envs[0].control_freq
        num_secs = 5
        control_steps = int(num_secs * control_freq)
        t = tick % control_steps
        # min and max distance the goal can spawn away from its current location. 30 = empirical upper bound on
        # velocity that the drones can handle.
        max_dist = min(30, max(self.room_dims))
        min_dist = max_dist / 2
        if tick % control_steps == 0 or tick == 1:
            # sample a new goal pos that's within the room boundaries and satisfies the distance constraint
            new_goal_found = False
            while not new_goal_found:
                low, high = np.array([-self.room_dims[0] / 2, -self.room_dims[1] / 2, 0]), np.array(
                    [self.room_dims[0] / 2, self.room_dims[1] / 2, self.room_dims[2]])
                new_pos = np.random.uniform(low=-high, high=high, size=(2, 3)).reshape(3,
                                                                                       2)  # need an intermediate point for  a deg=2 curve
                new_pos = new_pos * np.random.randint(min_dist, max_dist + 1) / np.linalg.norm(new_pos, axis=0)  # add some velocity randomization = random magnitude * unit direction
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
            self.goals = [self.interp[:, t] for _ in range(self.num_agents)]
            self.goals = np.array(self.goals)

            for i, env in enumerate(self.envs):
                env.goal = self.goals[i]


class Scenario_swarm_vs_swarm(QuadrotorScenario):
    def get_formation(self):
        return 'grid_vertical'

    def get_formation_size(self):
        return 1.0

    def step(self, infos, rewards):
        tick = self.envs[0].tick
        control_step_for_five_sec = int(5.0 * self.envs[0].control_freq)
        # Switch every 5th second
        if tick % control_step_for_five_sec == 0 and tick > 0:
            goal_1 = np.array([0.0, 0.0, 2.0])
            goal_2 = np.array([1.5, 1.5, 2.0])
            mid = self.num_agents // 2
            # Reverse every 10th second
            if tick % (control_step_for_five_sec * 2) == 0:
                for env in self.envs[:mid]:
                    env.goal = goal_1
                for env in self.envs[mid:]:
                    env.goal = goal_2
            else:
                for env in self.envs[:mid]:
                    env.goal = goal_2
                for env in self.envs[mid:]:
                    env.goal = goal_1
