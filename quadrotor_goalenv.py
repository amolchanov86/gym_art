from quadrotor_modular import *


def Rdiff(P, Q):
    """
    Rotation angle from matrix P to matrix Q
    :param P:
    :param Q:
    :return: float: rotation angle
    """
    R = np.matmul(P, Q.transpose())
    return np.arccos((np.trace(R) - 1.0) / 2.0)


def randrot():
    rotz = np.random.uniform(-np.pi, np.pi)
    return r3d.rotz(rotz)[:3, :3]


# Gym environment for quadrotor seeking the origin
# with no obstacles and full state observations
class QuadrotorGoalEnv(gym.GoalEnv):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }


    def __init__(self, raw_control=True):
        np.seterr(under='ignore')
        self.dynamics = default_dynamics()
        #self.controller = ShiftedMotorControl(self.dynamics)
        # self.controller = OmegaThrustControl(self.dynamics) ## The last one used
        #self.controller = VelocityYawControl(self.dynamics)
        self.scene = None
        self.oracle = NonlinearPositionController(self.dynamics)

        if raw_control:
            self.controller = RawControl(self.dynamics)
        else:
            # Mellinger controller
            self.controller = NonlinearPositionController(self.dynamics)

        self.action_space = self.controller.action_space(self.dynamics)
        self.action_last = self.action_default()


        # size of the box from which initial position will be randomly sampled
        # if box_scale > 1.0 then it will also growevery episode
        self.box = 2.0
        self.box_scale = 1.0 #scale the initialbox by this factor eache episode
        self.room_box = np.array([[-10, -10, 0], [10, 10, 10]])
        self.wall_offset = 0.3 #how much offset from the walls to have for initilization
        self.init_box = np.array([self.room_box[0] + self.wall_offset, self.room_box[1] - self.wall_offset])
        self.hover_eps = 0.1 #the box within which quad should be penalized for not adjusting its orientation and velocities

        # eps-radius of the goal
        self.goal_dist_eps = np.array([0.015,  # xyz
                                       0.015,  # Vxyz
                                       0.15,   # rotation angle tolerance (rad)
                                       0.15])  # Wxyz [rad/s]

        # pos, vel, rot, rot vel
        obs_dim = 3 + 3 + 9 + 3
        # TODO tighter bounds on some variables
        obs_high =  np.ones(obs_dim)
        obs_low  = -np.ones(obs_dim)
        # xyz room constraints
        obs_high[0:3] = self.room_box[1]
        obs_low[0:3]  = self.room_box[0]

        # rotation mtx guaranteed to be orthogonal
        obs_high[6:-3] = 1
        obs_low[6:-3] = -1

        self.observation_space = spaces.Dict(dict(
            desired_goal = spaces.Box(obs_low, obs_high, shape=obs_high.shape, dtype='float32'),
            achieved_goal= spaces.Box(obs_low, obs_high, shape=obs_high.shape, dtype='float32'),
            observation  = spaces.Box(obs_low, obs_high, shape=obs_high.shape, dtype='float32'),
        ))

        # TODO get this from a wrapper
        self.ep_len = 100
        self.tick = 0
        self.dt = 1.0 / 50.0
        self.crashed = False
        self.time_remain = self.ep_len

        self._seed()
        self._reset()

        if self.spec is None:
            self.spec = gym_reg.EnvSpec(id='Quadrotor-v0', max_episode_steps=self.ep_len)

    def action_default(self):
        return np.zeros([4,])


    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def _step(self, action):
        # print('actions: ', action)
        if not self.crashed:
            self.controller.step(self.dynamics, action, goal=self.goal[0:3], dt=self.dt)
            # self.oracle.step(self.dynamics, self.goal, goal=self.goal[0:3], dt=self.dt)
            self.crashed = self.scene.update_state(self.dynamics)
            self.crashed = self.crashed or not np.array_equal(self.dynamics.pos,
                                                          np.clip(self.dynamics.pos,
                                                                  a_min=self.room_box[0],
                                                                  a_max=self.room_box[1]))
        self.action_last = action.copy()
        self.time_remain = self.ep_len - self.tick
        rew_info = {}
        reward = self.compute_reward(achieved_goal=self.dynamics.state_vector(),
                                     desired_goal=self.goal,
                                     info=rew_info)

        self.tick += 1
        done = self.tick > self.ep_len or self.crashed
        sv = self.dynamics.state_vector()

        obs = {
            'achieved_goal': sv.copy(),
            'desired_goal': self.goal.copy(),
            'observation': sv.copy()
        }

        info = {
            'is_success': self._is_success(obs['achieved_goal'], self.goal),
            'rewards': rew_info,
        }

        return obs, reward, done, info


    def _reset(self):
        self.time_remain = self.ep_len
        if self.scene is None:
            self.scene = Quadrotor3DScene(None, self.dynamics.arm,
                640, 480, resizable=True, obstacles=False)

        # Goal and start point initilization
        self.goal = self._sample_goal()
        xyz_init, vel_init, rot_init, rot_vel_init = self._sample_init_state()
        self.dynamics.set_state(xyz_init, vel_init, rot_init, rot_vel_init)

        # Scene initilization
        self.scene.reset(self.goal[0:3], self.dynamics)
        self.scene.update_state(self.dynamics)

        self.crashed = False
        self.tick = 0
        # if self.ep_len < 1000:
        #     self.ep_len += 0.01 # len 1000 after 100k episodes

        state = self.dynamics.state_vector()
        self.action_last = self.action_default()

        obs = {
            'achieved_goal': state.copy(),
            'desired_goal': self.goal.copy(),
            'observation': state.copy()
        }

        return obs


    def _render(self, mode='human', close=False):
        self.scene.render_chase()


    def obs_components(self, obs):
        xyz = obs[0:3]
        vel = obs[3:6]
        rot_mx = obs[6:15]
        rot_vel = obs[15:18]
        return xyz, vel, rot_mx, rot_vel


    def compute_reward(self, achieved_goal, desired_goal, info):

        xyz, vel, rot_mx, rot_vel = self.obs_components(achieved_goal)
        goal_xyz, goal_vel, goal_rot_mx, goal_rot_vel = self.obs_components(desired_goal)

        if not self.crashed:
            #####################
            ## Loss position
            # log to create a sharp peak at the goal
            dist = np.linalg.norm(goal_xyz - xyz)
            loss_pos = np.log(dist + 0.1) + 0.1 * dist
            # loss_pos = dist

            # dynamics_pos = dynamics.pos
            # print('dynamics.pos', dynamics.pos)

            #Goal Proximity Coefficient (to have smooth influence of the axilliary distances)
            gpc = np.clip(-(1.0 / self.hover_eps) * dist + 1.0, a_min=0, a_max=1.0)
            #####################
            ## Loss velocity when within eps distance to the goal
            vel_dist = np.linalg.norm(goal_vel - vel)
            loss_vel_eps = gpc * 0.2 * vel_dist + (1.0 - gpc) * self.dt

            #####################
            ## Loss orientation when within eps distance to the goal
            rot_dist = np.fabs(Rdiff(goal_rot_mx, rot_mx))
            loss_rot_eps = gpc * 0.1 * rot_dist + (1.0 - gpc) * self.dt

            #####################
            ## Loss angular velocity when within eps distance to the goal
            rot_vel_dist = np.linalg.norm(goal_rot_vel - rot_vel)
            loss_rot_vel_eps = gpc * 0.1 * rot_vel_dist + (1.0 - gpc) * self.dt

            #####################
            ## penalize altitude above this threshold
            # max_alt = 6.0
            # loss_alt = np.exp(2 * (achieved_goal[2] - max_alt))

            #####################
            ## penalize amount of control effort
            # loss_effort = 0.001 * np.linalg.norm(self.action_last)

            #####################
            ## loss velocity
            # dx = desired_goal[0:3] - achieved_goal[0:3]
            # dx = dx / (np.linalg.norm(dx) + EPS)
            # vel_direct = achieved_goal[3:6] / (np.linalg.norm(achieved_goal[3:6]) + EPS)
            # vel_proj = np.dot(dx, vel_direct)
            # loss_vel_proj = -self.dt * 0.5 * (vel_proj - 1.0)

            #####################
            # Crashing
            loss_crash = 0
        else:
            loss_pos = 0
            loss_vel_eps = self.dt
            loss_rot_eps = self.dt
            loss_rot_vel_eps = self.dt

            # loss_alt = 0
            # loss_effort = 0
            # loss_vel_proj = 0
            loss_crash = self.dt * self.time_remain * 100

        reward = -self.dt * np.sum([loss_pos, loss_vel_eps, loss_rot_eps, loss_rot_vel_eps, loss_crash])

        rew_info = {'rew_crash': -loss_crash,
                    'rew_pos': -loss_pos,
                    'rew_vel_eps': -loss_vel_eps,
                    'rew_rot_esp': -loss_rot_eps,
                    'rew_rot_vel_eps': -loss_rot_vel_eps}

        # print('reward: ', reward, ' pos:', dynamics.pos, ' action', action)
        # print('pos', dynamics.pos)
        if np.isnan(reward) or not np.isfinite(reward):
            for key, value in locals().items():
                print('%s: %s \n' % (key, str(value)))
            raise ValueError('QuadEnv: reward is Nan')

        # assert reward == env.compute_reward(ob['achieved_goal'], ob['goal'], info)
        return reward

    def distances(self, obs1, obs2):
        """
        Spits out vector of distances by components (xyz, vel, rot, rot_vel)
        :param obs1:
        :param obs2:
        :return: vector of distances
        """
        xyz1, vel1, rot_mx1, rot_vel1 = self.obs_components(obs1)
        xyz2, vel2, rot_mx2, rot_vel2 = self.obs_components(obs2)

        dist_xyz = np.linalg.norm(xyz1 - xyz2)
        dist_vel = np.linalg.norm(vel1 - vel2)
        dist_rot_vel = np.linalg.norm(rot_vel1 - rot_vel2)
        dist_rot = Rdiff(rot_mx1, rot_mx2)

        return np.array([dist_xyz, dist_vel, dist_rot, dist_rot_vel])


    def _is_success(self, achieved_goal, desired_goal):
        """Indicates whether or not the achieved goal successfully achieved the desired goal.
        """
        return np.all(self.goal_dist_eps > self.distances(achieved_goal, desired_goal))


    def _sample_goal(self):
        """
        Samples a new goal and returns it.
        """
        xyz = self.np_random.uniform(-self.init_box, self.init_box)
        vel = np.array([0., 0., 0.])
        rot = np.zeros([9,])
        rot_vel = np.array([0., 0., 0.])

        return np.concatenate([xyz, vel, rot, rot_vel])


    def _sample_init_state(self):

        xyz = self.np_random.uniform(-self.init_box, self.init_box)
        vel = np.array([0., 0., 0.])
        rot_vel = np.array([0., 0., 0.])

        # Increase box size as a form of curriculum
        if self.box < 10:
            # from 0.5 to 10 after 100k episodes
            nextbox = self.box * self.box_scale
            if int(4 * nextbox) > int(4 * self.box):
                print("box:", nextbox)
            self.box = nextbox

        # make sure we're sort of pointing towards goal
        rotation = randrot()
        while np.dot(rotation[:, 0], to_xyhat(-xyz)) < 0.5:
            rotation = randrot()

        return xyz, vel, rotation, rot_vel




def test_rollout():
    #############################
    # Init plottting
    fig = plt.figure(1)
    # ax = plt.subplot(111)
    plt.show(block=False)

    render = True
    plot_step = 50
    time_limit = 25
    render_each = 2
    rollouts_num = 10
    plot_obs = False

    env = QuadrotorGoalEnv(raw_control=False)

    env.max_episode_steps = time_limit
    print('Reseting env ...')

    try:
        print('Observation space:', env.observation_space.low, env.observation_space.high)
        print('Action space:', env.action_space.low, env.action_space.high)
    except:
        print('Observation space:', env.observation_space.spaces[0].low, env.observation_space[0].spaces[0].high)
        print('Action space:', env.action_space[0].spaces[0].low, env.action_space[0].spaces[0].high)
    input('Press any key to continue ...')

    action = [0.5, 0.5, 0.5, 0.5]
    rollouts_id = 0

    while rollouts_id < rollouts_num:
        rollouts_id += 1
        s = env.reset()
        ## Diagnostics
        observations = []
        velocities = []

        t = 0
        while True:
            if render and (t % render_each == 0): env.render()
            s, r, done, info = env.step(action)
            observations.append(s['observation'])
            print('Step: ', t, ' Obs:', s)

            if t % plot_step == 0:
                plt.clf()

                if plot_obs:
                    observations_arr = np.array(observations)
                    # print('observations array shape', observations_arr.shape)
                    dimenstions = observations_arr.shape[1]
                    for dim in range(dimenstions):
                        plt.plot(observations_arr[:, dim])
                    plt.legend([str(x) for x in range(observations_arr.shape[1])])

                plt.pause(0.05) #have to pause otherwise does not draw
                plt.draw()
            if done: break
            t += 1
    input("Rollouts are done. Press Enter to continue...")

def main(argv):
    # parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-m',"--mode",
        type=int,
        default=0,
        help="Test mode: "
             "0 - rollout with default controller"
    )
    args = parser.parse_args()

    if args.mode == 0:
        print('Running test rollout ...')
        test_rollout()

if __name__ == '__main__':
    main(sys.argv)