import numpy as np

EPS = 1e-6
GRAV = 9.81  # default gravitational constant


class SingleObstacle():
    def __init__(self, max_init_vel=1., init_box=2.0, goal_central=np.array([0., 0., 2.0]), mode='no_obstacles',
                 type='sphere', size=0.0, quad_size=0.04, dt=0.05, traj='gravity', formation_size=0.0):
        self.max_init_vel = max_init_vel
        self.init_box = init_box  # means the size of initial space that the obstacles spawn at
        self.goal_central = goal_central
        self.mode = mode
        self.type = type
        self.size = size
        self.quad_size = quad_size
        self.dt = dt
        self.traj = traj
        self.formation_size = formation_size
        self.pos = np.array([100., 100., -100.])
        self.vel = np.array([0., 0., 0.])
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
            self.pos = np.array([100., 100., -100.])
            self.vel = np.array([0., 0., 0.])

    def static_obstacle(self):
        pass

    def dynamic_obstacle(self):
        # Init position for an obstacle
        x, y = np.random.uniform(-2 * self.init_box, 2 * self.init_box, size=(2,))
        z = np.random.uniform(-self.init_box, self.init_box) + self.goal_central[2]
        z = max(0.5, z)
        # Make the position of obstacles out of the space of goals
        formation_range = self.formation_size + self.size
        rel_x = abs(x) - formation_range
        rel_y = abs(y) - formation_range
        if rel_x <= 0:
            x += np.sign(x) * np.random.uniform(low=formation_range + 0.5 * self.init_box,
                                                high=formation_range + self.init_box)
        if rel_y <= 0:
            y += np.sign(y) * np.random.uniform(low=formation_range + 0.5 * self.init_box,
                                                high=formation_range + self.init_box)
        self.pos = np.array([x, y, z])

        # Init velocity for an obstacle
        # obstacle_vel = np.random.uniform(low=-self.max_init_vel, high=self.max_init_vel, size=(3,))
        if self.traj == 'gravity':
            self.vel = self.get_grav_init_vel()
        elif self.traj == "electron":
            self.vel = self.get_electron_init_vel()

    def get_grav_init_vel(self):
        # Calculate the initial position of the obstacle, which can make it finally fly through the center of the
        # goal formation.
        # There are three situations for the initial positions
        # 1. Below the center of goals (dz > 0). Then, there are two trajectories.
        # 2. Equal or above the center of goals (dz <= 0). Then, there is only one trajectory.
        # More details, look at: https://drive.google.com/file/d/1Vp0TaiQ_4vN9pH-Z3uGR54gNx6jh9thP/view
        target_noise = np.random.uniform(-0.5 * self.formation_size, 0.5 * self.formation_size, size=(3,))
        target_pos = self.goal_central + target_noise
        dx, dy, dz = target_pos - self.pos

        vz_noise = np.random.uniform(low=0.0, high=1.0)
        vz = np.sqrt(2 * GRAV * abs(dz)) + vz_noise
        delta = np.sqrt(vz * vz - 2 * GRAV * dz)
        if dz > 0:
            t_list = [(vz + delta) / GRAV, (vz - delta) / GRAV]
            t_index = round(np.random.uniform(low=0, high=1))
            t = t_list[t_index]
        elif dz < 0:
            # vz_index = 0, vz < 0; vz_index = 1, vz > 0;
            vz_index = round(np.random.uniform(low=0, high=1))
            if vz_index == 0:  # vz < 0
                vz = - vz

            t = (vz + delta) / GRAV
        else:  # dz = 0, vz > 0
            vz = np.random.uniform(low=0.5 * self.max_init_vel, high=self.max_init_vel)
            t = 2 * vz / GRAV

        # Calculate vx
        vx = dx / t
        vy = dy / t
        vel = np.array([vx, vy, vz])
        return vel

    def get_electron_init_vel(self):
        vel_direct = self.goal_central - self.pos
        vel_direct_noise = np.random.uniform(low=-0.1, high=0.1, size=(3,))
        vel_direct += vel_direct_noise
        vel_magn = np.random.uniform(low=0., high=self.max_init_vel)
        vel = vel_magn * vel_direct / (np.linalg.norm(vel_direct) + EPS)
        return vel

    def update_obs(self, obs=None, quads_pos=None, quads_vel=None):
        # The pos and vel of the obstacle give by the agents
        rel_pos = self.pos - quads_pos
        rel_vel = self.vel - quads_vel
        obs = np.concatenate((obs, rel_pos, rel_vel), axis=1)
        return obs

    def step(self, obs=None, quads_pos=None, quads_vel=None, set_obstacles=False):
        if not set_obstacles:
            obs = self.update_obs(obs=obs, quads_pos=quads_pos, quads_vel=quads_vel)
            return obs

        if self.traj == 'electron':
            obs = self.step_electron(obs=obs, quads_pos=quads_pos, quads_vel=quads_vel)
            return obs
        elif self.traj == 'gravity':
            obs = self.step_gravity(obs=obs, quads_pos=quads_pos, quads_vel=quads_vel)
            return obs
        else:
            raise NotImplementedError()

    def step_electron(self, obs=None, quads_pos=None, quads_vel=None):
        # Generate force, mimic force between electron, F = k*q1*q2 / r^2,
        # Here, F = r^2, k = 1, q1 = q2 = 1
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
        # Calculate position and velocity
        self.vel += self.dt * acc
        self.pos += self.dt * self.vel

        obs = self.update_obs(obs=obs, quads_pos=quads_pos, quads_vel=quads_vel)
        return obs

    def step_gravity(self, obs=None, quads_pos=None, quads_vel=None):
        acc = np.array([0., 0., -GRAV])  # 9.81
        # Calculate velocity
        self.vel += self.dt * acc
        self.pos += self.dt * self.vel

        obs = self.update_obs(obs=obs, quads_pos=quads_pos, quads_vel=quads_vel)
        return obs

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
