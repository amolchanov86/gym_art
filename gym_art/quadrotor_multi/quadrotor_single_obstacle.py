import numpy as np
from scipy import spatial
EPS = 1e-6
GRAV = 9.81  # default gravitational constant

class SingleObstacle():
    def __init__(self, max_init_vel=1., init_box=2.0, mean_goals=2.0, goal_central=np.array([0., 0., 2.0]), mode='no_obstacles',
                 type='sphere', size=0.0, quad_size=0.04, dt=0.05, traj='gravity', formation_size=0.0):
        self.max_init_vel = max_init_vel
        self.init_box = init_box
        self.mean_goals = mean_goals
        self.goal_central = goal_central
        self.mode = mode
        self.type = type
        self.size = size
        self.quad_size = quad_size
        self.dt = dt
        self.traj = traj
        self.formation_size = formation_size
        self.pos = np.zeros(3)
        self.vel = np.zeros(3)
        self.reset()

    def reset(self, set_obstacle=False):
        if set_obstacle:
            if self.mode == 'static':
                self.static_obstacle()
            elif self.mode == 'dynamic':
                self.dynamic_obstacle()
            else:
                pass
        else:
            self.pos = np.array([20.0, 20.0, 20.0])
            self.vel = np.array([0., 0., 0.])

    def step(self, quads_pos=None, quads_vel=None, set_obstacles=False):
        if self.traj == 'electron':
            return self.step_electron(quads_pos, quads_vel, set_obstacles)
        elif self.traj == 'gravity':
            return self.step_gravity(quads_pos, quads_vel, set_obstacles)
        else:
            raise NotImplementedError()

    def step_electron(self, quads_pos=None, quads_vel=None, set_obstacles=False):
        force_pos = 2 * self.goal_central - self.pos
        rel_force_goal = force_pos - self.goal_central
        force_noise = np.random.uniform(low=-0.5 * rel_force_goal, high=0.5 * rel_force_goal)
        force_pos = force_pos + force_noise
        rel_force_obstacle = force_pos - self.pos
        radius = 2.0 * np.linalg.norm(rel_force_obstacle)
        radius = max(EPS, radius)

        force_direction = rel_force_obstacle / radius
        force = radius * radius * force_direction
        # Calculate acceleration, F = ma, here, m = 1.0
        acc = force
        # Calculate velocity
        if set_obstacles:
            self.vel += self.dt * acc
        # Calculate pos
        if set_obstacles:
            self.pos += self.dt * self.vel

        # The pos and vel of the obstacle give by the agents
        rel_pos_obstacle_agents = self.pos - quads_pos
        rel_vel_obstacle_agents = self.vel - quads_vel
        return rel_pos_obstacle_agents, rel_vel_obstacle_agents

    def step_gravity(self, quads_pos=None, quads_vel=None, set_obstacles=False):
        acc = np.array([0., 0., -GRAV]) # 9.81
        # Calculate velocity
        if set_obstacles:
            self.vel += self.dt * acc
            self.pos += self.dt * self.vel


        # The pos and vel of the obstacle give by the agents
        rel_pos_obstacle_agents = self.pos - quads_pos
        rel_vel_obstacle_agents = self.vel - quads_vel
        return rel_pos_obstacle_agents, rel_vel_obstacle_agents

    def static_obstacle(self):
        pass

    def dynamic_obstacle(self):
        # Init position for an obstacle
        x, y= np.random.uniform(-2 * self.init_box, 2 * self.init_box, size=(2,))
        z = np.random.uniform(-self.init_box, self.init_box)
        judge_x_out_of_formation = abs(x) - (self.formation_size + self.size)
        judge_y_out_of_formation = abs(y) - (self.formation_size + self.size)
        if judge_x_out_of_formation <= 0:
            x += np.sign(x) * np.random.uniform(low=judge_x_out_of_formation + 1.0, high=judge_x_out_of_formation + 1.0 + self.init_box)
        if judge_y_out_of_formation <= 0:
            y += np.sign(y) * np.random.uniform(low=judge_y_out_of_formation + 1.0, high=judge_y_out_of_formation + 1.0 + self.init_box)

        z += self.mean_goals
        z = max(0.5, z)

        self.pos = np.array([x, y, z])

        # Init velocity for an obstacle
        # obstacle_vel = np.random.uniform(low=-self.max_init_vel, high=self.max_init_vel, size=(3,))
        if self.traj == 'gravity':
            target_noise = np.random.uniform(-0.5*self.formation_size, 0.5*self.formation_size, size=(3,))
            target_pos = self.goal_central + target_noise
            dx, dy, dz = target_pos - self.pos
            if dz >= 0:
                vz_noise = np.random.uniform(low=0.0, high=1.0)
                vz = np.sqrt(2 * GRAV * dz) + vz_noise
                delta = np.sqrt(vz * vz - 2 * GRAV * dz)
                t_list = [(vz+delta)/GRAV, (vz-delta)/GRAV]
                t_index  = round(np.random.uniform(low=0, high=1))
                t = t_list[t_index]
            else:
                vz_index = round(np.random.uniform(low=0, high=1))
                if vz_index == 0: # vz < 0
                    vz = np.random.uniform(low=-1.0, high=0.0)
                    delta = np.sqrt(vz * vz - 2 * GRAV * dz)
                    t = (-vz+delta)/GRAV
                else:
                    vz_noise = np.random.uniform(low=0.0, high=1.0)
                    vz = np.sqrt(-2 * GRAV * dz) + vz_noise
                    delta = np.sqrt(vz * vz + 2 * GRAV * dz)
                    t = (vz + delta) / GRAV

            # Calculate vx
            vx = dx / t
            vy = dy / t
            self.vel = np.array([vx, vy, vz])
        else:
            obstacle_vel_direct = self.goal_central - self.pos
            obstacle_vel_direct_noise = np.random.uniform(low=-0.1, high=0.1, size=(3,))
            obstacle_vel_direct += obstacle_vel_direct_noise
            obstacle_vel_magn = np.random.uniform(low=0., high=self.max_init_vel)
            obstacle_vel = obstacle_vel_magn / (np.linalg.norm(obstacle_vel_direct) + EPS) * obstacle_vel_direct
            self.vel = obstacle_vel

    def cube_detection(self, pos_quads=None):
        # https://developer.mozilla.org/en-US/docs/Games/Techniques/3D_collision_detection
        # Sphere vs. AABB
        collision_arr = np.zeros(len(pos_quads))
        for i, pos_quad in enumerate(pos_quads):
            rel_pos = np.maximum(self.pos - 0.5 * self.size, np.minimum(pos_quad, self.pos + 0.5 * self.size))
            distance = np.dot(rel_pos - pos_quad, rel_pos - pos_quad)
            if distance < self.quad_size ** 2:
                collision_arr[i] = 1.0

        return collision_arr

    def sphere_detection(self, pos_quads=None):
        dist = np.linalg.norm(pos_quads - self.pos, axis=1)
        collision_arr = (dist < (self.quad_size + 0.5 * self.size)).astype(np.float32)
        return collision_arr

    def collision_detection(self, pos_quads=None):
        if self.type == 'cube':
            collision_arr = self.cube_detection(pos_quads)
        elif self.type == 'sphere':
            collision_arr = self.sphere_detection(pos_quads)
        else:
            raise NotImplementedError()

        return collision_arr
