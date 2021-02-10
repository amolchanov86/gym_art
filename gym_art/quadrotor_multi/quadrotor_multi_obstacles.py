import numpy as np

from gym_art.quadrotor_multi.quadrotor_single_obstacle import SingleObstacle

EPS = 1e-6


class MultiObstacles:
    def __init__(self, mode='no_obstacles', num_obstacles=0, max_init_vel=1., init_box=2.0,
                 dt=0.005, quad_size=0.046, type='sphere', size=0.0, traj='gravity'):
        self.num_obstacles = num_obstacles
        self.obstacles = []
        for _ in range(num_obstacles):
            obstacle = SingleObstacle(max_init_vel=max_init_vel, init_box=init_box, mode=mode, type=type, size=size,
                                      quad_size=quad_size, dt=dt, traj=traj)
            self.obstacles.append(obstacle)

    def reset(self, obs=None, quads_pos=None, quads_vel=None, set_obstacles=False, formation_size=0.0, goal_central=np.array([0., 0., 2.])):
        if self.num_obstacles <= 0:
            return obs

        for obstacle in self.obstacles:
            obstacle.reset(set_obstacle=set_obstacles, formation_size=formation_size, goal_central=goal_central)

            # Add rel_pos and rel_vel to obs
            rel_pos = obstacle.pos - quads_pos
            rel_vel = obstacle.vel - quads_vel
            obs = np.concatenate((obs, rel_pos, rel_vel), axis=1)

        return obs

    def step(self, obs=None, quads_pos=None, quads_vel=None, set_obstacles=False):
        for obstacle in self.obstacles:
            obs = obstacle.step(obs=obs, quads_pos=quads_pos, quads_vel=quads_vel, set_obstacles=set_obstacles)

        return obs

    def collision_detection(self, pos_quads=None, set_obstacles=False):
        collision_arr = np.zeros((len(self.obstacles), len(pos_quads)))
        if set_obstacles is False:
            return collision_arr

        for i, obstacle in enumerate(self.obstacles):
            col_arr = obstacle.collision_detection(pos_quads=pos_quads)
            collision_arr[i] = col_arr

        return collision_arr
