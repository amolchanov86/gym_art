from gym_art.quadrotor.quadrotor_modular import *

# Gym environment for quadrotor seeking a given goal
# with obstacles and vision + IMU observations
class QuadrotorVisionEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self):
        np.seterr(under='ignore')
        self.dynamics = default_dynamics()
        #self.controller = ShiftedMotorControl(self.dynamics)
        self.controller = OmegaThrustControl(self.dynamics)
        self.action_space = self.controller.action_space(self.dynamics)
        self.scene = None
        self.crashed = False
        self.oracle = NonlinearPositionController(self.dynamics)

        seq_len = 4
        img_w, img_h = 64, 64
        img_space = spaces.Box(-1, 1, (img_h, img_w, seq_len))
        imu_space = spaces.Box(-100, 100, (6, seq_len))
        # vector from us to goal projected onto world plane and rotated into
        # our "looking forward" coordinates, and clamped to a maximal length
        dir_space = spaces.Box(-4, 4, (2, seq_len))
        self.observation_space = spaces.Tuple([img_space, imu_space, dir_space])
        self.img_buf = np.zeros((img_w, img_h, seq_len))
        self.imu_buf = np.zeros((6, seq_len))
        self.dir_buf = np.zeros((2, seq_len))

        # TODO get this from a wrapper
        self.ep_len = 500
        self.tick = 0
        self.dt = 1.0 / 50.0

        self._seed()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        if not self.crashed:
            #self.controller.step(self.dynamics, action, self.dt)
            # print("oracle step")
            self.oracle.step(self.dynamics, self.goal, self.dt)
            self.crashed = self.scene.update_state(self.dynamics)
        reward, rew_info = goal_seeking_reward(self.dynamics, self.goal, action, self.dt, self.crashed)
        self.tick += 1
        done = (self.tick > self.ep_len)

        rgb = self.scene.render_obs()
        # for debugging:
        #rgb = np.flip(rgb, axis=0)
        #plt.imshow(rgb)
        #plt.show()

        grey = (2.0 / 255.0) * np.mean(rgb, axis=2) - 1.0
        #self.img_buf = np.roll(self.img_buf, -1, axis=2)
        self.img_buf[:,:,:-1] = self.img_buf[:,:,1:]
        self.img_buf[:,:,-1] = grey

        imu = np.concatenate([self.dynamics.omega, self.dynamics.accelerometer])
        #self.imu_buf = np.roll(self.imu_buf, -1, axis=1)
        self.imu_buf[:,:-1] = self.imu_buf[:,1:]
        self.imu_buf[:,-1] = imu

        # heading measurement - simplified, #95489c has a more nuanced version
        our_gps = self.dynamics.pos[:2]
        goal_gps = self.goal[:2]
        dir = clamp_norm(goal_gps - our_gps, 4.0)
        #self.dir_buf = np.roll(self.dir_buf, -1, axis=1)
        self.dir_buf[:,:-1] = self.dir_buf[:,1:]
        self.dir_buf[:,-1] = dir

        return (self.img_buf, self.imu_buf, self.dir_buf), reward, done, {'rewards': rew_info}

    def _reset(self):
        if self.scene is None:
            self.scene = Quadrotor3DScene(self.np_random, self.dynamics.arm,
                640, 480, resizable=True)

        self.goal = self.scene.map.sample_goal(self.np_random)
        pos = self.scene.map.sample_start(self.np_random)
        vel = omega = npa(0, 0, 0)
        # for debugging collisions w/ no policy:
        #vel = self.np_random.uniform(-20, 20, size=3)
        #vel[2] = 0

        # make us point towards the goal
        xb = to_xyhat(self.goal - pos)
        zb = npa(0, 0, 1)
        yb = cross(zb, xb)
        rotation = np.column_stack([xb, yb, zb])
        self.dynamics.set_state(pos, vel, rotation, omega)
        self.crashed = False

        self.scene.reset(self.goal, self.dynamics)
        collided = self.scene.update_state(self.dynamics)
        assert not collided

        # fill the buffers with copies of initial state
        w, h, seq_len = self.img_buf.shape
        rgb = self.scene.render_obs()
        grey = (2.0 / 255.0) * np.mean(rgb, axis=2) - 1.0
        self.img_buf = np.tile(grey[:,:,None], (1,1,seq_len))
        imu = np.concatenate([self.dynamics.omega, self.dynamics.accelerometer])
        self.imu_buf = np.tile(imu[:,None], (1,seq_len))

        self.tick = 0
        return (self.img_buf, self.imu_buf)

    def _render(self, mode='human', close=False):
        self.scene.render_chase()