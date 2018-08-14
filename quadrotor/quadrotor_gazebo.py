#!/usr/bin/env python

import numpy as np
#from gym_art.quadrotor.quadrotor_modular import *
from quadrotor_modular import *


def Rdiff(P, Q):
    """
    Rotation angle from matrix P to matrix Q
    :param P:
    :param Q:
    :return: float: rotation angle
    """
    R = np.matmul(P, Q.transpose())
    #We have to clip because in quad env we do not perform orthogonalization every time
    return np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, a_min=-1.0, a_max=1.0))


def randrot():
    rotz = np.random.uniform(-np.pi, np.pi)
    return r3d.rotz(rotz)[:3, :3]


# Gym environment for quadrotor seeking the origin
# with no obstacles and full state observations
class QuadrotorGazeboEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self, raw_control=True, vertical_only=False):
        np.seterr(under='ignore')
        self.dynamics = default_dynamics()
        self.scene = None
        self.vertical_only = vertical_only
        if self.vertical_only:
            self.viewpoint = 'side'
        else:
            self.viewpoint = 'chase'

        if raw_control:
            if vertical_only:
                self.controller = VerticalControl(self.dynamics)
            else:
                self.controller = RawControl(self.dynamics)
        else:
            # Mellinger controller
            self.controller = NonlinearPositionController(self.dynamics)

        self.action_space = self.controller.action_space(self.dynamics)
        self.action_last = self.action_default()


        # size of the box from which initial position will be randomly sampled
        # if box_scale > 1.0 then it will also growevery episode
        self.box = 2.0
        self.box_scale = 1.0  #scale the initialbox by this factor eache episode
        self.room_size = 3  #height, width, length
        self.room_box = np.array([[-self.room_size, -self.room_size, 0],
                                  [self.room_size, self.room_size, self.room_size]])
        self.wall_offset = 0.3  #how much offset from the walls to have for initilization
        self.init_box = np.array([self.room_box[0] + self.wall_offset, self.room_box[1] - self.wall_offset])
        self.hover_eps = 0.1  #the box within which quad should be penalized for not adjusting its orientation and velocities

        # eps-radius of the goal
        self.goal_diameter = 0.2

        self.goal_dist_eps = np.array([self.goal_diameter,
                                       0.2]) #Vxyz

        # self.goal_dist_eps = np.array([self.goal_diameter,  # xyz
        #                                0.15,  # Vxyz
        #                                np.pi,   # rotation angle tolerance (rad)
        #                                0.15])  # Wxyz [rad/s]

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
            desired_goal = spaces.Box(obs_low, obs_high, dtype='float32'),
            achieved_goal= spaces.Box(obs_low, obs_high, dtype='float32'),
            observation  = spaces.Box(obs_low, obs_high, dtype='float32'),
        ))

        # TODO get this from a wrapper
        self.ep_len = 200
        self.tick = 0
        self.dt = 1.0 / 50.0
        self.crashed = False
        self.time_remain = self.ep_len

        self._seed()
        self.reset()

        if self.spec is None:
            self.spec = gym_reg.EnvSpec(id='QuadrotorGoalEnv-v1', max_episode_steps=self.ep_len)

        # self._max_episode_seconds = self.ep_len
        self._max_episode_steps = self.ep_len
        self._elapsed_steps = 0
        # self._episode_started_at = None

    def action_default(self):
        return np.zeros([4,])


    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def obs2goal(self, obs):
        xyz, vel_xyz, rot, rot_vel = self.obs_components(obs)
        return np.concatenate([xyz, vel_xyz])

    def step(self, action):
        # print('actions: ', action)
        if not self.crashed:
            self.controller.step(dynamics=self.dynamics, action=action, goal=self.goal[0:3], dt=self.dt)
            # self.oracle.step(self.dynamics, self.goal, goal=self.goal[0:3], dt=self.dt)
            self.crashed = self.scene.update_state(self.dynamics)
            self.crashed = self.crashed or not np.array_equal(self.dynamics.pos,
                                                          np.clip(self.dynamics.pos,
                                                                  a_min=self.room_box[0],
                                                                  a_max=self.room_box[1]))
        self.action_last = action.copy()
        self.time_remain = self.ep_len - self.tick
        # info MUST contain all current state variables
        # since info will be passed when the goals will be recomputed
        info = {}
        info['crashed'] = self.crashed
        info['time_remain'] = self.time_remain
        reward = self.compute_reward(achieved_goal=self.dynamics.state_vector(),
                                     desired_goal=self.goal,
                                     info=info)

        self.tick += 1
        self._elapsed_steps = self.tick
        done = self.tick > self.ep_len #or self.crashed
        sv = self.dynamics.state_vector()

        obs = {
            'achieved_goal': self.obs2goal(sv.copy()),
            'desired_goal': self.goal.copy(),
            'observation': sv.copy()
        }

        info['is_success'] =  self._is_success(obs['achieved_goal'], self.goal)

        return obs, reward, done, info


    def reset(self):
        self.time_remain = self.ep_len
        if self.scene is None:
            self.scene = Quadrotor3DScene(None, self.dynamics.arm,
                640, 480, resizable=True, obstacles=False,
                                          goal_diameter=self.goal_diameter,
                                          viewpoint=self.viewpoint)

        # Goal and start point initilization
        self.goal = self._sample_goal()
        xyz_init, vel_init, rot_init, rot_vel_init = self._sample_init_state()
        self.dynamics.set_state(xyz_init, vel_init, rot_init, rot_vel_init)

        # Scene initilization
        self.scene.reset(self.goal[0:3], self.dynamics)
        self.scene.update_state(self.dynamics)

        self.crashed = False
        self.tick = 0
        self._elapsed_steps = 0
        # if self.ep_len < 1000:
        #     self.ep_len += 0.01 # len 1000 after 100k episodes

        state = self.dynamics.state_vector()
        self.action_last = self.action_default()

        obs = {
            'achieved_goal': self.obs2goal(state.copy()),
            'desired_goal': self.goal.copy(),
            'observation': state.copy()
        }

        return obs


    def render(self, mode='human', close=False):
        self.scene.render_chase()

    # ... allows indexing single and multi dimensional arrays
    def obs_components(self, obs):
        return obs[..., 0:3], obs[..., 3:6], obs[..., 6:15], obs[..., 15:18]


    # ... allows indexing single and multi dimensional arrays
    def goal_components(self, obs):
        return obs[..., 0:3], obs[..., 3:6]

    def compute_reward(self, achieved_goal, desired_goal, info):
        """
        This function must be vectorizable, i.e. it must be capable of processing batches
        :param achieved_goal:
        :param desired_goal:
        :param info:
        :return:
        """

        xyz, vel = self.goal_components(achieved_goal)
        goal_xyz, goal_vel = self.goal_components(desired_goal)


        #####################
        ## Loss position
        # log to create a sharp peak at the goal
        dist = np.linalg.norm(goal_xyz - xyz, axis=-1, keepdims=True).flatten()
        loss_pos = np.log(dist + 0.1) + 0.1 * dist
        # loss_pos = dist

        # dynamics_pos = dynamics.pos
        # print('dynamics.pos', dynamics.pos)

        #Goal Proximity Coefficient (to have smooth influence of the axilliary distances)
        gpc = np.clip(-(1.0 / self.hover_eps) * dist + 1.0, a_min=0, a_max=1.0)
        #####################
        ## Loss velocity when within eps distance to the goal
        vel_dist = np.linalg.norm(goal_vel - vel, axis=-1, keepdims=True).flatten()
        loss_vel_eps = gpc * 0.2 * vel_dist + (1.0 - gpc) * self.dt

        #####################
        ## Loss orientation when within eps distance to the goal
        # rot_dist = np.fabs(Rdiff(goal_rot_mx.reshape([3,3]), rot_mx.reshape([3,3])))
        # loss_rot_eps = gpc * 0.1 * rot_dist + (1.0 - gpc) * self.dt

        #####################
        ## Loss angular velocity when within eps distance to the goal
        # rot_vel_dist = np.linalg.norm(goal_rot_vel - rot_vel, axis=-1, keepdims=True).flatten()
        # loss_rot_vel_eps = gpc * 0.1 * rot_vel_dist + (1.0 - gpc) * self.dt

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
        loss_crash = np.zeros_like(dist)

        #####################
        # Corrections for crashed states with vectorization
        if not isinstance(info['crashed'], np.ndarray):
            #have to make shape (1,1) otherwise indexing will be impossible
            #because in vector form the first dimension is batch
            crashed_bool = np.array([info['crashed']]).astype(bool)
            time_remain = np.array([info['time_remain']])
        else:
            crashed_bool = info['crashed'].astype(bool).squeeze()
            time_remain = info['time_remain'].squeeze()

        loss_pos[crashed_bool] = 0
        loss_vel_eps[crashed_bool] = self.dt
        # loss_rot_vel_eps[crashed_bool] = self.dt
        loss_crash[crashed_bool] = self.dt * time_remain[crashed_bool] * 100

        reward_mx = np.stack([loss_pos, loss_vel_eps, loss_crash], axis=-1)
        reward = -self.dt * np.sum(reward_mx, axis=-1).squeeze()

        rew_info = {'rew_crash': -loss_crash,
                    'rew_pos': -loss_pos,
                    'rew_vel_eps': -loss_vel_eps}

        # print('reward: ', reward, ' pos:', dynamics.pos, ' action', action)
        # print('pos', dynamics.pos)
        if np.any(np.isnan(reward)) or not np.all(np.isfinite(reward)):
            for key, value in locals().items():
                print('%s: %s \n' % (key, str(value)))
            raise ValueError('QuadEnv: reward is Nan')

        # assert reward == env.compute_reward(ob['achieved_goal'], ob['goal'], info)
        return reward


    def _compute_reward_nonvector(self, achieved_goal, desired_goal, info):
        """
        Non vectorized reward
        :param achieved_goal:
        :param desired_goal:
        :param info:
        :return:
        """

        xyz, vel, rot_mx, rot_vel = self.obs_components(achieved_goal)
        goal_xyz, goal_vel, goal_rot_mx, goal_rot_vel = self.obs_components(desired_goal)

        if not info['crashed']:
            #####################
            ## Loss position
            # log to create a sharp peak at the goal
            dist = np.linalg.norm(goal_xyz - xyz, axis=-1)
            loss_pos = np.log(dist + 0.1) + 0.1 * dist
            # loss_pos = dist

            # dynamics_pos = dynamics.pos
            # print('dynamics.pos', dynamics.pos)

            #Goal Proximity Coefficient (to have smooth influence of the axilliary distances)
            gpc = np.clip(-(1.0 / self.hover_eps) * dist + 1.0, a_min=0, a_max=1.0)
            #####################
            ## Loss velocity when within eps distance to the goal
            vel_dist = np.linalg.norm(goal_vel - vel, axis=-1)
            loss_vel_eps = gpc * 0.2 * vel_dist + (1.0 - gpc) * self.dt

            #####################
            ## Loss orientation when within eps distance to the goal
            # rot_dist = np.fabs(Rdiff(goal_rot_mx.reshape([3,3]), rot_mx.reshape([3,3])))
            # loss_rot_eps = gpc * 0.1 * rot_dist + (1.0 - gpc) * self.dt

            #####################
            ## Loss angular velocity when within eps distance to the goal
            # rot_vel_dist = np.linalg.norm(goal_rot_vel - rot_vel, axis=-1)
            # loss_rot_vel_eps = gpc * 0.1 * rot_vel_dist + (1.0 - gpc) * self.dt

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
            # loss_rot_eps = self.dt
            # loss_rot_vel_eps = self.dt

            # loss_alt = 0
            # loss_effort = 0
            # loss_vel_proj = 0
            loss_crash = self.dt * info['time_remain'] * 100

        reward = -self.dt * np.sum([loss_pos, loss_vel_eps, loss_crash], axis=-1)

        rew_info = {'rew_crash': -loss_crash,
                    'rew_pos': -loss_pos,
                    'rew_vel_eps': -loss_vel_eps}

        # print('reward: ', reward, ' pos:', dynamics.pos, ' action', action)
        # print('pos', dynamics.pos)
        if np.any(np.isnan(reward)) or not np.all(np.isfinite(reward)):
            for key, value in locals().items():
                print('%s: %s \n' % (key, str(value)))
            raise ValueError('QuadEnv: reward is Nan')

        # assert reward == env.compute_reward(ob['achieved_goal'], ob['goal'], info)
        return reward

    def distances(self, goal1, goal2):
        """
        Spits out vector of distances by components (xyz, vel, rot, rot_vel)
        :param goal1:
        :param goal2:
        :return: vector of distances
        """
        xyz1, vel1 = self.goal_components(goal1)
        xyz2, vel2 = self.goal_components(goal2)

        dist_xyz = np.linalg.norm(xyz1 - xyz2)
        dist_vel = np.linalg.norm(vel1 - vel2)
        # print('dist_rot: ', dist_rot)

        return np.array([dist_xyz, dist_vel])


    def distances_obs(self, obs1, obs2):
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
        dist_rot = np.fabs(Rdiff(rot_mx1.reshape([3,3]), rot_mx2.reshape([3,3])))
        # print('dist_rot: ', dist_rot)

        return np.array([dist_xyz, dist_vel, dist_rot, dist_rot_vel])


    def _is_success(self, achieved_goal, desired_goal):
        """Indicates whether or not the achieved goal successfully achieved the desired goal.
        """
        return np.all(self.goal_dist_eps > self.distances(achieved_goal, desired_goal))

    def _sample_goal(self):
        """
        Samples a new goal and returns it.
        """
        if self.vertical_only:
            xyz = np.array([0., 0., 0.])
            xyz[2] = self.np_random.uniform(self.init_box[0][2], self.init_box[1][2])
        else:
            xyz = np.random.uniform(low=self.init_box[0], high=self.init_box[1])
        vel = np.array([0., 0., 0.])

        return np.concatenate([xyz, vel])


    def _sample_goal_full(self):
        """
        Samples a new goal and returns it.
        """
        if self.vertical_only:
            xyz = np.array([0., 0., 0.])
            xyz[2] = self.np_random.uniform(self.init_box[0][2], self.init_box[1][2])
        else:
            xyz = np.random.uniform(low=self.init_box[0], high=self.init_box[1])
        vel = np.array([0., 0., 0.])
        rot = np.eye(3).flatten()
        rot_vel = np.array([0., 0., 0.])

        return np.concatenate([xyz, vel, rot, rot_vel])


    def _sample_init_state(self):
        if self.vertical_only:
            xyz = self.goal.copy()[0:3]
            xyz[2] = self.np_random.uniform(self.init_box[0][2], self.init_box[1][2])
            vel = np.array([0., 0., 0.])
            rot_vel = np.array([0., 0., 0.])
            rotation = np.eye(3)
        else:
            xyz = self.np_random.uniform(self.init_box[0], self.init_box[1])
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


#topics = [
 #"imu", 
 #"motor_speed",
#"motor_position",
#"motor_force",
#"magnetic_field",
#"gps",
#"rc",
#"status",
#"filtered_sensor_data",
#"air_speed",
#"ground_speed",
# "command/motor_speed", # COMMAND_ACTUATORS
#"command/rate_thrust",
#"command/roll_pitch_yawrate_thrust",
#"command/attitude_thrust",
#"command/trajectory",
#"command/pose",
#"command/gps_waypoint",
#"pose",
#"pose_with_covariance",
#"transform",
#"odometry",
#"position",
#"wrench",
#"wind_speed",
#"external_force",
#"ground_truth/pose",
#"ground_truth/twist",
#"command/trajectory"
#]


# string model_name
# geometry_msgs/Pose pose
#   geometry_msgs/Point position
#     float64 x
#     float64 y
#     float64 z
#   geometry_msgs/Quaternion orientation
#     float64 x
#     float64 y
#     float64 z
#     float64 w
# geometry_msgs/Twist twist
#   geometry_msgs/Vector3 linear
#     float64 x
#     float64 y
#     float64 z
#   geometry_msgs/Vector3 angular
#     float64 x
#     float64 y
#     float64 z
# string reference_frame


def test_gazeobo(thrust_val):
    """
    THe simple test for gazebo with hummingbird
    First, launch
    roslaunch rotors_gazebo humminbird_raw_control.launch
    """
    import rospy
    import rospy.rostime
    from nav_msgs.msg import Odometry
    from mav_msgs.msg import Actuators
    from trajectory_msgs.msg import MultiDOFJointTrajectoryPoint
    from gazebo_msgs.srv import SetModelState
    from gazebo_msgs.msg import ModelState
    
    quadrotor = "hummingbird"
    # That is for us to command a new trajectory
    trajectory_topic = "command_trajectory"
    # Topic to get feedback from the quadrotor
    odometry_topic = "odometry_sensor1/odometry"
    # Topic to send commands to quadrotor
    actuators_topic = "command/motor_speed"
    # Resettting quadrotor
    reset_topic = "/gazebo/set_model_state"


    #Initializing the node
    rospy.init_node('quadrotor_env', anonymous=True)
    
    def odometry_callback(msg):
        # print("Odometry received", msg)
        pass
    
    def traj_callback(msg):
        # print("Trajectory received", msg)
        pass

    def reset(reset_service, pos=[0,0,0], orientation=[0,0,0,1], pos_vel=[0,0,0], angle_vel=[0,0,0]):
        print("Sending reset request ...")
        req = ModelState()
        req.model_name = "hummingbird"

        req.pose.position.x = pos[0]
        req.pose.position.y = pos[1]
        req.pose.position.z = pos[2]

        req.pose.orientation.x = orientation[0]
        req.pose.orientation.y = orientation[1]
        req.pose.orientation.z = orientation[2]
        req.pose.orientation.w = orientation[3]

        req.twist.linear.x = pos_vel[0]
        req.twist.linear.y = pos_vel[1]
        req.twist.linear.z = pos_vel[2]

        req.twist.angular.x = angle_vel[0]
        req.twist.angular.y = angle_vel[1]
        req.twist.angular.z = angle_vel[2]

        print('Sending request: ', req)

        try:
            resp = reset_service(req)
            print('RESET response: ', resp)
            return resp
        except rospy.ServiceException as e:
            print('Reset failed: ', str(e))
        

    # Setting subscribers and publishers    
    rospy.Subscriber(quadrotor + "/" + odometry_topic, Odometry, odometry_callback)
    rospy.Subscriber(quadrotor + "/" + trajectory_topic, MultiDOFJointTrajectoryPoint, traj_callback)
    action_publisher = rospy.Publisher(quadrotor + "/" + actuators_topic, Actuators, queue_size=1)
    
    # Waiting for reset service to appear
    rospy.wait_for_service(reset_topic)
    reset_service = rospy.ServiceProxy(reset_topic, SetModelState)

    # Resetting
    reset(reset_service)

    # Looping
    while True:
        actuator_msg = Actuators()
        actuator_msg.angular_velocities = thrust_val*np.array([1, 1, 1, 1])
        action_publisher.publish(actuator_msg)
        rospy.sleep(0.1)
        


    # Just prevent exiting
    # rospy.spin()
   


def test_rollout():
    import transforms3d as t3d
    import seaborn as sns
    sns.set_style('darkgrid')

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

    env = QuadrotorGoalEnv(raw_control=False, vertical_only=True)

    env.max_episode_steps = time_limit
    print('Reseting env ...')

    try:
        print('Observation space:', env.observation_space.low, env.observation_space.high)
        print('Action space:', env.action_space.low, env.action_space.high)
    except:
        print('Observation space:', env.observation_space.spaces['observation'].low, env.observation_space.spaces['observation'].high)
        print('Action space:', env.action_space.low, env.action_space.high)
    # input('Press any key to start rollouts ...')

    action = [0.5, 0.5, 0.5, 0.5]
    rollouts_id = 0
    ep_lengths = []

    distances_arr = []
    angles_arr = []
    distances_legend = ['xyz', 'vel']
    angles_legend = ['roll', 'pitch', 'yaw', 'roll_des', 'pitch_des', 'yaw_des']

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
            print('Step: ', t, ' Obs:', s['observation'], 'Goal: ', s['desired_goal'], 'Reached:', info['is_success'])
            print('Reward:', r)

            if t % plot_step == 0:
                plt.figure(1)
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
            if done:
                distances_arr.append(env.distances(s['achieved_goal'], s['desired_goal']))
                angles_arr.append(t3d.euler.mat2euler(s['observation'][6:15].reshape([3,3]), 'sxyz') +
                                  t3d.euler.mat2euler(np.eye(3), 'sxyz'))
                ep_lengths.append(t)
                break
            t += 1

    print('Average ep length:', np.mean(ep_lengths))
    print('remaining distances at the end of the episode: ', distances_arr)
    plt.figure(2)
    distances_arr = np.array(distances_arr)
    for i, dim in enumerate(distances_legend):
        plt.plot(distances_arr[:, i])
    plt.legend(distances_legend)

    plt.figure(3)
    angles_arr = np.array(angles_arr)
    print('remaining angles at the end of the episode: ', angles_arr)
    for i, dim in enumerate(angles_legend):
        plt.plot(angles_arr[:, i])
    plt.legend(angles_legend)

    plt.pause(0.05)
    plt.show(block=False)

    input("Rollouts are done. Press Enter to continue...")

def main(argv):
    # parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-m',"--mode",
        type=int,
        default=0,
        help="Test mode: "
             "0 - gazebo simple test"
             "1 - rollout test with mellenger controller"
    )
    parser.add_argument(
        "-a", "--action",
        type=int,
        default=450,
        help="Thrust value. Max: 838. Hower: ~450"
        )
    args = parser.parse_args()

    if args.mode == 1:
        print('Running test rollout ...')
        test_rollout()
    
    if args.mode == 0:
        print('Running simple gazebo test ...')
        test_gazeobo(args.action)

if __name__ == '__main__':
    main(sys.argv)