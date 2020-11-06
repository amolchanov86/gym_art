import numpy as np

from gym_art.quadrotor_multi.quadrotor_single_obstacle import Single_Obstacle

EPS = 1e-6


class MultiObstacles():
    def __init__(self, mode='no_obstacles', num_obstacles=0, max_init_vel=1., init_box=2.0, mean_goals=2.0,
                 goal_central=np.array([0., 0., 2.0]), dt=0.005, quad_size=0.04, type='sphere', size=0.0):
        self.max_init_vel = max_init_vel
        self.init_box = init_box
        self.mode = mode
        self.num_obstacles = num_obstacles
        self.mean_goals = mean_goals
        self.goal_central = goal_central
        self.dt = dt
        self.size = size
        self.quad_size = quad_size
        self.type = type
        self.obstacles = []
        for i in range(num_obstacles):
            obstacle = Single_Obstacle(max_init_vel=self.max_init_vel, init_box=self.init_box,
                                       mean_goals=self.mean_goals, goal_central=self.goal_central,
                                       mode=self.mode, type=self.type, size=self.size, quad_size=self.quad_size,
                                       dt=self.dt
                                       )
            self.obstacles.append(obstacle)

    def reset(self, obs=None, quads_pos=None, quads_vel=None, set_obstacles=False):
        for obstacle in self.obstacles:
            obstacle.reset(set_obstacle=set_obstacles)

            # Add rel_pos and rel_vel to obs
            rel_pos = obstacle.pos - quads_pos
            rel_vel = obstacle.vel - quads_vel
            obs = np.concatenate((obs, rel_pos, rel_vel), axis=1)  # TODO: Improve, same as extend_obs function

        return obs

    def step(self, obs=None, quads_pos=None, quads_vel=None, set_obstacles=False):
        # Generate force, mimic force between electron, F = k*q1*q2 / r^2,
        # Here, F = r^2, k = 1, q1 = q2 = 1
        for obstacle in self.obstacles:
            rel_pos_obstacle_agents, rel_vel_obstacle_agents = obstacle.step(quads_pos=quads_pos, quads_vel=quads_vel,
                                                                             set_obstacles=set_obstacles)

            obs = np.concatenate((obs, rel_pos_obstacle_agents, rel_vel_obstacle_agents), axis=1)

        return obs

    def collision_detection(self, pos_quads=None, saft_dist=0.05):
        collision_arr = np.zeros((len(self.obstacles), len(pos_quads)))
        for i, obstacle in enumerate(self.obstacles):
            col_arr = obstacle.collision_detection(pos_quads=pos_quads)
            collision_arr[i] = col_arr

        return collision_arr
