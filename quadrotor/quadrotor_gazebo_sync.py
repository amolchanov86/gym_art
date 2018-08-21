#!/usr/bin/env python

"""
OpenAI gym wrapper for the Gazebo Rotors.Gazebo Hummingbird quadrotor
Notes:
- make sure that you run in unpased mode, since the scripts does not ask to unpause


Important files:
- hummingbird physics params: 
https://github.com/TaoChenOSU/rotors_simulator/blob/gazebo_step_control/rotors_description/urdf/hummingbird.xacro
- Lee position controller:
https://github.com/TaoChenOSU/rotors_simulator/blob/gazebo_step_control/rotors_control/src/library/lee_position_controller.cpp


"""

import numpy as np
#from gym_art.quadrotor.quadrotor_modular import *
from gym_art.quadrotor.quadrotor_modular import *
import logging
import time
import sys, os
import traceback

import rospy
import rospy.rostime
from nav_msgs.msg import Odometry
from mav_msgs.msg import Actuators
from trajectory_msgs.msg import MultiDOFJointTrajectoryPoint
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState
from rotors_step_simulation_plugin.srv import RequestToTakeNSteps

import transforms3d as t3d
# from transforms3d.euler import mat2euler

current_time_ms = lambda: int(round(time.time() * 1000))

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
    rotz = np.random.uniform(-nbuddyp.pi, np.pi)
    return r3d.rotz(rotz)[:3, :3]

def quat2R(qw, qx, qy, qz):
    
    R = \
    [[1.0 - 2*qy**2 - 2*qz**2,         2*qx*qy - 2*qz*qw,         2*qx*qz + 2*qy*qw],
     [      2*qx*qy + 2*qz*qw,   1.0 - 2*qx**2 - 2*qz**2,         2*qy*qz - 2*qx*qw],
     [      2*qx*qz - 2*qy*qw,         2*qy*qz + 2*qx*qw,   1.0 - 2*qx**2 - 2*qy**2]]
    return np.array(R)

def qwxyz2R(quat):
    return quat2R(qw=quat[0], qx=quat[1], qy=quat[2], qz=quat[3])

def R2quat(rot):
    # print('R2quat: ', rot, type(rot))
    R = rot.reshape([3,3])
    w = np.sqrt(1.0 + R[0,0] + R[1,1] + R[2,2]) / 2.0;
    w4 = (4.0 * w);
    x = (R[2,1] - R[1,2]) / w4
    y = (R[0,2] - R[2,0]) / w4
    z = (R[1,0] - R[0,1]) / w4
    return np.array([w,x,y,z])

def default_hummingbird_dynamics(sim_steps=1, room_box=None):
    ## For hummingbird parameters see:
    # https://github.com/TaoChenOSU/rotors_simulator/blob/gazebo_step_control/rotors_description/urdf/hummingbird.xacro

    # similar to AscTec Hummingbird
    # TODO: dictionary of dynamics of real quadrotors
    mass = 0.68 # 0.009 * 4 #body + rotors
    arm_length = 0.34 / 2.0
    # inertia = mass * npa(0.01, 0.01, 0.02)
    inertia = npa(0.007, 0.007, 0.012)
    thrust_to_weight = 2.0
    thrust2motorspeed=8.54858e-06
    return QuadrotorGazeboSyncDynamics(mass, arm_length, inertia,
        thrust_to_weight=thrust_to_weight, dynamics_steps_num=sim_steps, 
        thrust2motorspeed=thrust2motorspeed, room_box=room_box)



# simple simulation of quadrotor dynamics.
class QuadrotorGazeboSyncDynamics(object):
    def __init__(self, mass, arm_length, inertia, 
        thrust_to_weight=2.0, torque_to_thrust=0.05, 
        dynamics_steps_num=1, thrust2motorspeed=8.54858e-06, 
        room_box=None, obs_repr="state_xyz_vxyz_euler_omega"):
        assert np.isscalar(mass)
        assert np.isscalar(arm_length)
        assert inertia.shape == (3,)
        # This hack allows parametrize calling dynamics multiple times
        # without expensive for-loops
        self.dynamics_steps_num = dynamics_steps_num

        ## Obs representations:
        # state_xyz_vxyz_rot_omega
        # state_xyz_vxyz_euler_omega
        # state_xyz_vxyz_quat_omega
        self.obs_repr = obs_repr
        self.state_vector = getattr(self, obs_repr)

        if room_box is None:
            self.room_box = np.array([[-3., -3., 0.], [3., 3., 3.]])
        else:
            self.room_box = np.array(room_box).copy()

        self.get_observation_space()

        # Measuring odometry frequency
        self.odo_start_time = current_time_ms() / 1000.
        self.odo_freq = 0.

        # Quadrotor physical parameters
        self.mass = mass
        self.arm = arm_length
        self.inertia = inertia
        self.thrust_to_weight = thrust_to_weight
        self.thrust = GRAV * mass * thrust_to_weight / 4.0
        self.torque = torque_to_thrust * self.thrust
        self.thrust2motorspeed = thrust2motorspeed

        ## X configuration (Jame's configuration and craziflie)
        # scl = arm_length / norm([1.,1.,0.])
        # self.prop_pos = scl * np.array([
        #     [1.,  1., -1., -1.],
        #     [1., -1., -1.,  1.],
        #     [0.,  0.,  0.,  0.]]).T # row-wise easier with np

        ## + configurations (Hummingbird in Rotors)
        scl = arm_length
        self.prop_pos = scl * np.array([
            [1.,  0., -1.,  0.],
            [0.,  1.,  0.,  -1.],
            [0.,  0.,  0.,  0.]]).T # row-wise easier with np

        # unit: meters^2 ??? maybe wrong
        self.prop_crossproducts = np.cross(self.prop_pos, [0., 0., 1.])
        # 1 for props turning CCW, -1 for CW # the comment is probably
        self.prop_ccw = np.array([1., -1., 1., -1.])
        # self.prop_ccw = np.array([-1., 1., -1., 1.])
        self.max_angular_val = 838. #rad/s``

        self.def_state()

        ##########################################
        ### ROS & Gazebo stuff

        self.quadrotor = "hummingbird"
        self.reset_topic = "/gazebo/set_model_state"
        self.actuators_topic = "/gazebo_step/take_n_steps"
        
        ## Original rotors topics (left here for convenience)
        # That is for us to command a new trajectory
        #self.trajectory_topic = self.quadrotor + "/" + "command_trajectory"
        # Topic to get feedback from the quadrotor
        #self.odometry_topic = self.quadrotor + "/" + "odometry_sensor1/odometry"
        # Topic to send commands to quadrotor
        #self.actuators_topic = self.quadrotor + "/" + "command/motor_speed"
        # Resettting quadrotor

        self.init_ros()
            
        # Reseting gazebo (defaults)
        print("DYN: Resetting the env ...")
        self.set_state(pos=[0., 0., self.arm + 0.2], vel=[0.,0.,0.], rot=np.eye(3), omega=[0.,0.,0.])  
        
        # Auxiliary stuff
        self.time_last = current_time_ms()
        print("DYN: Ready to run simulation ...")

    def get_observation_space(self):
        if self.obs_repr == "state_xyz_vxyz_rot_omega":
            ## Creating observation space
            # pos, vel, rot, rot vel
            obs_dim = 3 + 3 + 9 + 3
            # TODO tighter bounds on some variables
            obs_high =  np.ones(obs_dim)
            obs_low  = -np.ones(obs_dim)
            # xyz room constraints
            obs_high[0:3] = self.room_box[1]
            obs_low[0:3]  = self.room_box[0]


        elif self.obs_repr == "state_xyz_vxyz_euler_omega":
             ## Creating observation space
            # pos, vel, rot, rot vel
            obs_dim = 3 + 3 + 3 + 3
            # TODO tighter bounds on some variables
            obs_high =  np.ones(obs_dim)
            obs_low  = -np.ones(obs_dim)
            # xyz room constraints
            obs_high[0:3] = self.room_box[1]
            obs_low[0:3]  = self.room_box[0]

            # Euler angles
            obs_high[6:9] = np.pi*obs_high[6:9] 
            obs_low[6:9]  = np.pi*obs_low[6:9]           

        elif self.obs_repr == "state_xyz_vxyz_quat_omega":
             ## Creating observation space
            # pos, vel, rot, rot vel
            obs_dim = 3 + 3 + 4 + 3
            # TODO tighter bounds on some variables
            obs_high =  np.ones(obs_dim)
            obs_low  = -np.ones(obs_dim)
            # xyz room constraints
            obs_high[0:3] = self.room_box[1]
            obs_low[0:3]  = self.room_box[0]


        self.observation_space = spaces.Box(obs_low, obs_high)
        return self.observation_space
        ## This stuff for GoalEnv (later marked simply as GoalEnv)
        # try:
        #     self.observation_space = spaces.Dict(dict(
        #         desired_goal = spaces.Box(obs_low, obs_high, dtype='float32'),
        #         achieved_goal= spaces.Box(obs_low, obs_high, dtype='float32'),
        #         observation  = spaces.Box(obs_low, obs_high, dtype='float32'),
        #     ))
        # except Exception as e:
        #     self.observation_space = spaces.Box(-obs_high, obs_high)
    

        
    def def_state(self):
        self.pos = np.array([0.,0.,0.])
        self.vel = np.array([0.,0.,0.])
        self.rot = np.eye(3)
        self.omega = np.array([0.,0.,0.])
        self.quat = R2quat(self.rot)
        self.euler = t3d.euler.mat2euler(self.rot)


    def init_ros(self):
        print("DYN: initilization of the ROS node ...") 
        rospy.init_node('quadrotor_env', anonymous=True)


        # Setting subscribers and publishers
        print("DYN: setting publishers and subscribers ...")    

        # Waiting for reset service to appear
        print("DYN: waiting for service:", self.reset_topic)
        rospy.wait_for_service(self.reset_topic)
        self.reset_service = rospy.ServiceProxy(self.reset_topic, SetModelState)
        print("DYN: successfully initialized service:", self.reset_topic)

        print("DYN: waiting for service:", self.actuators_topic)
        rospy.wait_for_service(self.actuators_topic)
        self.actuator_service = rospy.ServiceProxy(self.actuators_topic, RequestToTakeNSteps)       
        print("DYN: successfully initialized service:", self.actuators_topic)

        ## Original subsribers to topics (left here for convenience)
        # self.action_publisher = rospy.Publisher(self.actuators_topic, Actuators, queue_size=1)
        # rospy.Subscriber(self.odometry_topic, Odometry, self.odometry_callback, queue_size=1)
        # rospy.Subscriber(self.quadrotor + "/" + self.trajectory_topic, MultiDOFJointTrajectoryPoint, self.traj_callback)
        
        time.sleep(0.2)
        print("DYN: All subscribers are initialized ...")


    def step(self, thrust_cmds, dt):
        ## Service message content
        # int8 step_size
        # float32 motor_0_speed
        # float32 motor_1_speed
        # float32 motor_2_speed
        # float32 motor_3_speed
        # ---
        # nav_msgs/Odometry new_state

        ## Approximate torque control (converting torque to angular velocity for quad input)
        motor_speeds = list(np.clip(np.sqrt((thrust_cmds * self.thrust) / self.thrust2motorspeed), 
            a_min=0., a_max=self.max_angular_val).astype(np.float32))


        # step_start_time = time.time()
        try:
            resp = self.actuator_service(step_size=self.dynamics_steps_num, 
                                        motor_speeds=motor_speeds)
        except rospy.ServiceException as e:
            print('ERROR: DYN: Action failed: ', str(e))
            traceback.print_exc(file=sys.stdout)
        # print("ENV: action req step time: ", time.time() - step_start_time)

        self.odometry_callback(msg=resp.new_state)
        self.check_collisions()

        return self.state_vector()


    def check_collisions(self):
        self.crashed = self.pos[2] <= (self.arm + 0.05) #offset to account for rotor itself
        return self.crashed


    def repackOdometry(self, msg):
        # print("DYN: repack: msg: ", msg)
        xyz = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z])
        quat = np.array([msg.pose.pose.orientation.w, msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z])
        vel_xyz = np.array([msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z])
        vel_angular = np.array([msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z])
        return (xyz, quat, vel_xyz, vel_angular)

    def odometry_callback(self, msg):
        # print("Odometry received", msg)
        self.pos, self.quat, self.vel_body, self.omega = self.repackOdometry(msg)
        self.rot = quat2R(qw=self.quat[0], qx=self.quat[1], qy=self.quat[2], qz=self.quat[3])
        # print("DYN: odom callback: state", self.state_vector())
        # Gazebo publishes in the body frame (both vel and omega)
        # converting to the world frame for the controller (policy)
        self.vel = np.matmul(self.rot, self.vel_body)
        self.euler = t3d.euler.mat2euler(self.rot)

        time_cur = current_time_ms() / 1000.
        odo_dt = time_cur - self.odo_start_time
        self.odo_freq = 1./ (odo_dt + 1e-6)
        self.odo_start_time = time_cur
        # print('odo freq: ', self.odo_freq)

        ## The way to convert omega to the world frame:        
        # self.omega = np.matmul(self.rot.T, self.omega_glob)

        ## Debug printing
        # print('Odometry:')
        # print('xyz:',self.pos)
        # print('quat:', self.quat)
        # print('vel_xyz:', self.vel)
        # print('vel_angular:', self.omega)

    def set_state(self, pos, vel, rot, omega):
        # print('DYN: set_state: pos, rot, vel, omega', pos, rot, vel, omega)
        req = ModelState()
        # print("DYN: ModelState: ", req)
        req.model_name = "hummingbird"

        quat = R2quat(rot)

        req.pose.position.x = pos[0]
        req.pose.position.y = pos[1]
        req.pose.position.z = pos[2]

        req.pose.orientation.x = quat[1]
        req.pose.orientation.y = quat[2]
        req.pose.orientation.z = quat[3]
        req.pose.orientation.w = quat[0]

        req.twist.linear.x = vel[0]
        req.twist.linear.y = vel[1]
        req.twist.linear.z = vel[2]

        req.twist.angular.x = omega[0]
        req.twist.angular.y = omega[1]
        req.twist.angular.z = omega[2]

        # print('DYN: Sending RESET request: ', req)

        try:
            resp = self.reset_service(req)
            print('DYN: RESET response: ', resp)
            return resp
        except rospy.ServiceException as e:
            print('ERROR: DYN: Reset failed: ', str(e))
            traceback.print_exc(file=sys.stdout)

        self.pos = np.array(pos)
        self.rot = np.array(rot)
        self.vel = np.array(vel)
        self.omega = np.array(omega)
        self.quat = R2quat(self.rot)
        self.euler = t3d.euler.mat2euler(self.rot)

        self.check_collisions()
        if self.crashed:
            print("WARNING: DYN: Quadrotor was initialized in crashed state!")

        return self.state_vector()

    def state_xyz_vxyz_rot_omega(self):
        return np.concatenate([
            self.pos, self.vel, self.rot.flatten(), self.omega])

    def state_xyz_vxyz_quat_omega(self):
        return np.concatenate([
            self.pos, self.vel, self.quat, self.omega])

    def state_xyz_vxyz_euler_omega(self):
        return np.concatenate([
            self.pos, self.vel, self.euler, self.omega])

    def action_space(self):
        low = np.zeros(4)
        high = np.ones(4)
        return spaces.Box(low, high)

    def step_old(self, thrust_cmds, dt):
        """
        This is how you can publish in the old version
        """
        # Publish the action
        actuator_msg = Actuators()

        ## Direct angular velocity control        
        # angular_velocities = (self.max_angular_val*np.array(thrust_cmds)).astype(dtype=np.int)
        
        ## Approximate torque control (converting torque to angular velocity for quad input)
        angular_velocities = np.clip(np.sqrt((thrust_cmds * self.thrust) / 8.54858e-06), 
            a_min=0., a_max=self.max_angular_val)

        # print('Rotor commands: ', angular_velocities)
        actuator_msg.angular_velocities = angular_velocities
        self.action_publisher.publish(actuator_msg)
    
    def traj_callback(self, msg):
        print("Trajectory received:\n", msg)


# Gym environment for quadrotor seeking the origin
# with no obstacles and full state observations
# (this functionality does not fully work yet, step() and reset() functions should be fixed)
# try:
#     # For latest gym we could use GoalEnv 
#     gym_env_parent = gym.GoalEnv
# except:
#     # For older gym it is just Env
#     gym_env_parent = gym.Env



"""
Synchronized version of the quadrotor env.
Requires special plugin for making sync steps:
https://github.com/TaoChenOSU/rotors_step_simulation_plugin
"""
class QuadrotorGazeboSyncEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self, raw_control=True, vertical_only=False, goal=[0.,0.,2.]):
        np.seterr(under='ignore')

        self.room_box = 5. * np.array([[-1., -1., 0], [1., 1., 1.]])
        self.sim_steps = 1
        self.dynamics = default_hummingbird_dynamics(sim_steps=self.sim_steps, room_box=self.room_box)
        self.vertical_only = vertical_only

        if goal is not None:
            self.goal_static = np.array(goal)
        else:
            self.goal_static = None

        if raw_control:
            if vertical_only:
                self.controller = VerticalControl(self.dynamics)
            else:
                self.controller = RawControl(self.dynamics)
        else:
            # Mellinger controller
            self.controller = NonlinearPositionController(self.dynamics, tf_control=False)     

            # Awesome gains    
            self.controller.kp_p, self.controller.kd_p = 3.5, 2.9 # 2.0, 2.0 # 
            self.controller.kp_a, self.controller.kd_a = 50., 11. # 50.0, 12.# 

            # Lowgains (if you see oscilations - try to use these)
            # self.controller.kp_p, self.controller.kd_p = 2.0, 2.0 # 
            # self.controller.kp_a, self.controller.kd_a = 50.0, 12.# 

        
        self.rotors_num = 4
        self.action_space = self.controller.action_space(self.dynamics)
        self.action_last = self.action_default()


        # size of the box from which initial position will be randomly sampled
        # if box_scale > 1.0 then it will also growevery episode
        self.box = 2.0
        self.box_scale = 1.0  #scale the initialbox by this factor eache episode
        self.wall_offset = 0.3  #how much offset from the walls to have for initilization
        self.init_box = np.array([self.room_box[0] + self.wall_offset, self.room_box[1] - self.wall_offset])
        self.hover_eps = 0.1  #the box within which quad should be penalized for not adjusting its orientation and velocities

        # eps-radius of the goal
        self.goal_diameter = 0.2

        self.goal_dist_eps = np.array([self.goal_diameter,
                                       0.2]) #Vxyz

        self.observation_space = self.dynamics.observation_space


        # TODO get this from a wrapper
        self.ep_time = 4.0 #In seconds - this is just preference.  
        
        self.control_freq = 100. / self.sim_steps #0.01 integration step
        # self.rosrate = rospy.Rate(self.control_freq)
        self.dt = 1.0 / self.control_freq
        self.ep_len = int(self.ep_time * self.control_freq)
        self.tick = 0

        self.crashed = False
        self.time_remain = self.ep_len

        self._seed()
        self.reset()

        if self.spec is None:
            self.spec = gym_reg.EnvSpec(id='QuadrotorGazeboEnv-v1', max_episode_steps=self.ep_len)

        self._elapsed_steps = 0
        self.add_goal_marker()

        ################################################################
        ## Activate logger when necessary
        # for handler in logging.root.handlers[:]:
        #     logging.root.removeHandler(handler)
        # logging.basicConfig(filename='/tmp/quadrotor_gazebo.log', level=logging.DEBUG)


    def add_goal_marker(self):
        self.add_marker(self.goal, self.goal_dist_eps[0])


    @staticmethod
    def add_marker(pos, size):
        print("DYN: adding the marker")
        os.system("gz marker -x") #removes all markers
        add_marker_cmd =  \
        "gz marker -m \'action: ADD_MODIFY, type: SPHERE, id: 0, pose: {position: {x:%.3f, y:%.3f, z:%.3f}}, scale: {x:%.2f, y:%.2f, z:%.2f}'" % \
        (pos[0], pos[1], pos[2], size, size, size)
        os.system(add_marker_cmd)


    def action_default(self):
        return np.zeros([self.rotors_num,])


    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def obs2goal(self, obs):
        xyz, vel_xyz, rot, rot_vel = self.obs_components(obs)
        return np.concatenate([xyz, vel_xyz])

    def step(self, action):
        # print('actions: ', action)
        # if not self.crashed:
        # step_start_time = time.time()
        self.controller.step(dynamics=self.dynamics, action=action, goal=self.goal[0:3], dt=self.dt)
        # print("ENV: contr step time: ", time.time() - step_start_time)
        # self.oracle.step(self.dynamics, self.goal, goal=self.goal[0:3], dt=self.dt)
        # self.rosrate.sleep() #Allows maintain frequency relative to simulation

        self.crashed = self.dynamics.crashed
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

        ## GoalEnv
        # obs = {
        #     'achieved_goal': self.obs2goal(sv.copy()),
        #     'desired_goal': self.goal.copy(),
        #     'observation': sv.copy()
        # }
        obs = sv.copy()

        ## GoalEnv
        # info['is_success'] =  self._is_success(obs['achieved_goal'], self.goal)
        info['is_success'] =  self._is_success(obs, self.goal)

        return obs, reward, done, info


    def reset(self):
        # print('ENV: resetting env ...')
        self.time_remain = self.ep_len

        # Goal and start point initilization
        self.goal = self._sample_goal()
        # self.add_marker(self.goal, self.goal_dist_eps[0])
        # print('new goal: ', self.goal)
        xyz_init, vel_init, rot_init, rot_vel_init = self._sample_init_state()
        print('ENV: reset: pos, vel, rot, omega: ', xyz_init, vel_init, rot_init, rot_vel_init, rot_init.shape)
        self.dynamics.set_state(pos=xyz_init, vel=vel_init, rot=rot_init, omega=rot_vel_init)

        # Scene initilization
        # self.scene.reset(self.goal[0:3], self.dynamics)

        self.crashed = False
        self.tick = 0
        self._elapsed_steps = 0
        # if self.ep_len < 1000:
        #     self.ep_len += 0.01 # len 1000 after 100k episodes

        state = self.dynamics.state_vector()
        self.action_last = self.action_default()

        # obs = {
        #     'achieved_goal': self.obs2goal(state.copy()),
        #     'desired_goal': self.goal.copy(),
        #     'observation': state.copy()
        # }
        obs = state.copy()

        return obs

    def _render(self, mode='human', close=False):
        # self.scene.render_chase()
        pass

    def render(self, mode='human', close=False):
        # self.scene.render_chase()
        pass

    # ... allows indexing single and multi dimensional arrays
    def obs_components(self, obs):
        return obs[..., 0:3], obs[..., 3:6], obs[..., 6:15], obs[..., 15:18]


    # ... allows indexing single and multi dimensional arrays
    def goal_components(self, obs):
        return obs[..., 0:3], obs[..., 3:6]

    def compute_reward_vectorized(self, achieved_goal, desired_goal, info):
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


    def compute_reward(self, achieved_goal, desired_goal, info):
        xyz, vel, rot_mx, rot_vel = self.obs_components(achieved_goal)
        goal_xyz, goal_vel, goal_rot_mx, goal_rot_vel = self.obs_components(desired_goal)
        
        ###############################################
        ## Distance based reward
        dist = np.linalg.norm(goal_xyz - xyz, axis=-1)
        loss_pos = np.log(dist + 0.1) + 0.1 * dist
        # loss_vert_pos = np.abs(goal_xyz[2] - xyz[2])
        # loss_hor_pos = 0.25 * np.linalg.norm(goal_xyz[0:2] - xyz[0:2])
        # loss_pos = loss_vert_pos + loss_hor_pos

        ###############################################
        # gpc = np.clip(-(1.0 / self.hover_eps) * dist + 1.0, a_min=0, a_max=1.0)
        #Goal Proximity Coefficient (to have smooth influence of the axilliary distances)
        
        ###############################################
        ## Loss orientation
        loss_orient = -self.dynamics.rot[2,2] #Projection of the z-body axis to z-world axis
        

        ###############################################
        ## Loss velocity when within eps distance to the goal
        # vel_dist = np.linalg.norm(goal_vel - vel, axis=-1)
        # loss_vel_eps = gpc * 0.2 * vel_dist + (1.0 - gpc) * self.dt

        ###############################################
        ## Crashing
        loss_crash =  float(self.crashed)


        #####################
        ## loss velocity direction
        # dx = desired_goal[0:3] - achieved_goal[0:3]
        # dx = dx / (np.linalg.norm(dx) + EPS)
        # vel_direct = achieved_goal[3:6] / (np.linalg.norm(achieved_goal[3:6]) + EPS)
        # vel_proj = np.dot(dx, vel_direct)
        # loss_vel_proj = - 0.5 * (vel_proj - 1.0)

        ###############################################
        ## Total
        # reward = -self.dt * np.sum([loss_pos, loss_vel_eps, loss_crash, loss_vel_proj], axis=-1)
        reward = -self.dt * np.sum([loss_pos, loss_crash, loss_orient], axis=-1)
        # print('reward: ', reward, ' pos:', dynamics.pos, ' action', action)

        ## Reporting
        # rew_info = {'rew_crash': -loss_crash,
        #             'rew_pos': -loss_pos,
        #             'rew_vel_eps': -loss_vel_eps,
        #             'rew_vel_proj': -loss_vel_proj}
        rew_info = {'rew_crash': -loss_crash,
                    'rew_pos': -loss_pos,
                    "rew_orient": -loss_orient}

        info["rewards"] = rew_info
        if np.any(np.isnan(reward)) or not np.all(np.isfinite(reward)):
            for key, value in locals().items():
                print('%s: %s \n' % (key, str(value)))
            raise ValueError('QuadEnv: reward is Nan')

        # assert reward == env.compute_reward(ob['achieved_goal'], ob['goal'], info)
        return reward


    def compute_reward_full(self, achieved_goal, desired_goal, info):
        xyz, vel, rot_mx, rot_vel = self.obs_components(achieved_goal)
        goal_xyz, goal_vel, goal_rot_mx, goal_rot_vel = self.obs_components(desired_goal)
        
        ###############################################
        ## Distance based reward
        dist = np.linalg.norm(goal_xyz - xyz, axis=-1)
        # loss_pos = np.log(dist + 0.1) + 0.1 * dist
        loss_pos = dist

        ###############################################
        #Goal Proximity Coefficient (to have smooth influence of the axilliary distances)
        gpc = np.clip(-(1.0 / self.hover_eps) * dist + 1.0, a_min=0, a_max=1.0)
        
        ###############################################
        ## Loss velocity when within eps distance to the goal
        vel_dist = np.linalg.norm(goal_vel - vel, axis=-1)
        loss_vel_eps = gpc * 0.2 * vel_dist + (1.0 - gpc) * self.dt

        ###############################################
        ## Crashing
        loss_crash = 10 * float(self.crashed)


        #####################
        ## loss velocity direction
        dx = desired_goal[0:3] - achieved_goal[0:3]
        dx = dx / (np.linalg.norm(dx) + EPS)
        vel_direct = achieved_goal[3:6] / (np.linalg.norm(achieved_goal[3:6]) + EPS)
        vel_proj = np.dot(dx, vel_direct)
        loss_vel_proj = - 0.5 * (vel_proj - 1.0)

        ###############################################
        ## Total
        reward = -self.dt * np.sum([loss_pos, loss_vel_eps, loss_crash, loss_vel_proj], axis=-1)
        # print('reward: ', reward, ' pos:', dynamics.pos, ' action', action)

        ## Reporting
        rew_info = {'rew_crash': -loss_crash,
                    'rew_pos': -loss_pos,
                    'rew_vel_eps': -loss_vel_eps,
                    'rew_vel_proj': -loss_vel_proj}

        info["rewards"] = rew_info
        if np.any(np.isnan(reward)) or not np.all(np.isfinite(reward)):
            for key, value in locals().items():
                print('%s: %s \n' % (key, str(value)))
            raise ValueError('QuadEnv: reward is Nan')

        # assert reward == env.compute_reward(ob['achieved_goal'], ob['goal'], info)
        return reward



    def compute_reward_james(self, achieved_goal, desired_goal, info):
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
            loss_pos = np.log(dist + 0.1) + 0.1 * dist ## Reward from James
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
            loss_crash = self.dt * 100

        reward = -self.dt * np.sum([loss_pos, loss_vel_eps, loss_crash], axis=-1)

        rew_info = {'rew_crash': -loss_crash,
                    'rew_pos': -loss_pos,
                    'rew_vel_eps': -loss_vel_eps}

        info["rewards"] = rew_info

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
        if self.goal_static is not None:
            xyz = self.goal_static.copy()
        else:
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


    def randrot(self):
        rotz = np.random.uniform(-np.pi, np.pi)
        return r3d.rotz(rotz)[:3,:3]


    def _sample_init_state(self):
        ## This is to set some random independently of the type of init (just debugging)
        # xyz = self.np_random.uniform(3*np.array([-1, -1, -0.1]), 3*np.array([1, 1, 0.1]) )  + self.goal_static
        # vel = np.array([0., 0., 0.])
        # return xyz, vel, np.array([[0, -1, 0],[1, 0, 0],[0, 0 ,1]]), vel.copy()
        
        # rot = self.randrot()
        
        ## This is to set desired rotation to controller. By default it is just along x axis
        # self.controller.rot_des = rot.copy()

        # xyz = self.goal[0:3]
        # vel = self.goal[3:6]
        # rot = np.eye(3)
        # rot_vel = np.array([0,0,0])
        # return xyz, vel, rot, rot_vel

        if self.vertical_only:
            xyz = self.goal.copy()[0:3]
            #Only vertical coord is random
            xyz[2] = self.np_random.uniform(self.init_box[0][2], self.init_box[1][2])
            vel = np.array([0., 0., 0.])
            rot_vel = np.array([0., 0., 0.])
            rot = np.eye(3)
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
            rot = self.randrot()
            while np.dot(rot[:, 0], to_xyhat(-xyz)) < 0.5:
                rot = self.randrot()

        # print('ENV: _sample_init_state: rot', rot, type(rotation))
        return xyz, vel, rot, rot_vel


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
   

def test_rollout():
    import transforms3d as t3d
    # import seaborn as sns
    # sns.set_style('darkgrid')

    #############################
    # Init plottting
    plot_obs = False
    if plot_obs:
        fig = plt.figure(1)
        # ax = plt.subplot(111)
        plt.show(block=False)

    render = True
    plot_step = 100
   
    render_each = 2
    rollouts_num = 10


    env = QuadrotorGazeboSyncEnv(raw_control=False, vertical_only=False, goal=[0.,0.,2.])

    time_limit = env.spec.max_episode_steps
    print('Reseting env ...')

    try:
        print('Observation space:', env.observation_space.low, env.observation_space.high)
        print('Action space:', env.action_space.low, env.action_space.high)
    except:
        print('Observation space:', env.observation_space.spaces['observation'].low, env.observation_space.spaces['observation'].high)
        print('Action space:', env.action_space.low, env.action_space.high)
    # input('Press any key to start rollouts ...')

    #Since raw control by default creates bias 0.5 it should have some gravity compensation effect 
    action = np.array([0.0, 0.0, 0.0, 0.0]) 
    rollouts_id = 0
    ep_lengths = []

    distances_arr = []
    angles_arr = []
    distances_legend = ['xyz', 'vel']
    angles_legend = ['roll', 'pitch', 'yaw', 'roll_des', 'pitch_des', 'yaw_des']


    exp_start_time = time.time()
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
            observations.append(s)

            ## For GoalEnv
            # observations.append(s['observation'])
            # print('Step: ', t, ' Obs:', s['observation'], 'Goal: ', s['desired_goal'], 'Reached:', info['is_success'])
            # print('Reward:', r)

            if t % plot_step  == 0 and plot_obs:
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
                break
            #     distances_arr.append(env.distances(s['achieved_goal'], s['desired_goal']))

            #     angles_arr.append(t3d.euler.mat2euler(s['observation'][6:15].reshape([3,3]), 'sxyz') +
            #                       t3d.euler.mat2euler(np.eye(3), 'sxyz'))
                
            #     ep_lengths.append(t)
            #     break
            t += 1

    # print('Average ep length:', np.mean(ep_lengths))
    # print('remaining distances at the end of the episode: ', distances_arr)
    # plt.figure(2)
    # distances_arr = np.array(distances_arr)
    # for i, dim in enumerate(distances_legend):
    #     plt.plot(distances_arr[:, i])
    # plt.legend(distances_legend)

    # plt.figure(3)
    # angles_arr = np.array(angles_arr)
    # print('remaining angles at the end of the episode: ', angles_arr)
    # for i, dim in enumerate(angles_legend):
    #     plt.plot(angles_arr[:, i])
    # plt.legend(angles_legend)

    # plt.pause(0.05)

    print("Exp time: ", time.time() - exp_start_time)
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
             "0 - rollout test with mellenger controller"
    )
    parser.add_argument(
        "-a", "--action",
        type=int,
        default=450,
        help="Thrust value. Max: 838. Hower: ~450"
        )
    args = parser.parse_args()

    if args.mode == 0:
        print('Running test rollout ...')
        test_rollout()

if __name__ == '__main__':
    main(sys.argv)