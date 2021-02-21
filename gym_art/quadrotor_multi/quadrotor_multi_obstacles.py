import numpy as np
from scipy import spatial

from gym_art.quadrotor_multi.quadrotor_single_obstacle import SingleObstacle
from gym_art.quadrotor_multi.quad_obstacle_utils import OBSTACLES_SHAPE_LIST

EPS = 1e-6


class MultiObstacles:
    def __init__(self, mode='no_obstacles', num_obstacles=0, max_init_vel=1., init_box=2.0,
                 dt=0.005, quad_size=0.046, shape='sphere', size=0.0, traj='gravity'):
        self.num_obstacles = num_obstacles
        self.obstacles = []
        self.shape = shape
        self.shape_list = OBSTACLES_SHAPE_LIST

        for _ in range(num_obstacles):
            obstacle = SingleObstacle(max_init_vel=max_init_vel, init_box=init_box, mode=mode, shape=shape, size=size,
                                      quad_size=quad_size, dt=dt, traj=traj)
            self.obstacles.append(obstacle)

    def reset(self, obs=None, quads_pos=None, quads_vel=None, set_obstacles=None, formation_size=0.0, goal_central=np.array([0., 0., 2.])):
        if self.num_obstacles <= 0:
            return obs
        if set_obstacles is None:
            raise ValueError('set_obstacles is None')

        if self.shape == 'random':
            shape_list = self.get_shape_list()
        else:
            shape_list = [self.shape for _ in range(self.num_obstacles)]
            shape_list = np.array(shape_list)

        for i, obstacle in enumerate(self.obstacles):
            obst_obs = obstacle.reset(set_obstacle=set_obstacles[i], formation_size=formation_size,
                                      goal_central=goal_central, shape=shape_list[i], quads_pos=quads_pos,
                                      quads_vel=quads_vel)

            obs = np.concatenate((obs, obst_obs), axis=1)

        return obs

    def step(self, obs=None, quads_pos=None, quads_vel=None, set_obstacles=None):
        if set_obstacles is None:
            raise ValueError('set_obstacles is None')

        for i, obstacle in enumerate(self.obstacles):
            obst_obs = obstacle.step(quads_pos=quads_pos, quads_vel=quads_vel, set_obstacle=set_obstacles[i])
            obs = np.concatenate((obs, obst_obs), axis=1)

        return obs

    def collision_detection(self, pos_quads=None, set_obstacles=None):
        if set_obstacles is None:
            raise ValueError('set_obstacles is None')

        # Shape: (num_agents, num_obstacles)
        collision_matrix = np.zeros((len(pos_quads), self.num_obstacles))

        for i, obstacle in enumerate(self.obstacles):
            if set_obstacles[i]:
                col_arr = obstacle.collision_detection(pos_quads=pos_quads)
                collision_matrix[:, i] = col_arr

        # check which drone collide with obstacle(s)
        all_collisions = []
        col_w1 = np.where(collision_matrix >= 1)
        for i, val in enumerate(col_w1[0]):
            all_collisions.append((col_w1[0][i], col_w1[1][i]))

        obst_positions = np.stack([self.obstacles[i].pos for i in range(self.num_obstacles)])
        distance_matrix = spatial.distance_matrix(x=pos_quads, y=obst_positions)

        return collision_matrix, all_collisions, distance_matrix

    def get_shape_list(self):
        all_shapes = np.array(self.shape_list)
        shape_id_list = np.random.randint(low=0, high=len(all_shapes), size=self.num_obstacles)
        shape_list = all_shapes[shape_id_list]
        return shape_list
