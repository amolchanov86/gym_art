#!/usr/bin/env python

import numpy as np
import time
current_time_ms = lambda: int(round(time.time() * 1000))

import rospy
import rospy.rostime
from nav_msgs.msg import Odometry
from mav_msgs.msg import Actuators
from trajectory_msgs.msg import MultiDOFJointTrajectoryPoint
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState



def quat2R(qw, qx, qy, qz):
    
    R = \
    [[1.0 - 2*qy**2 - 2*qz**2,         2*qx*qy - 2*qz*qw,         2*qx*qz + 2*qy*qw],
     [      2*qx*qy + 2*qz*qw,   1.0 - 2*qx**2 - 2*qz**2,         2*qy*qz - 2*qx*qw],
     [      2*qx*qz - 2*qy*qw,         2*qy*qz + 2*qx*qw,   1.0 - 2*qx**2 - 2*qy**2]]
    return np.array(R)




class TestGazebo(object):
    def __init__(self, thrust_val, freq=10.0):
        super(TestGazebo, self).__init__()

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
        # Sync publisher (send syncing messages)
        sync_topic = "/world_control"


        self.start_time = time.time()
        self.odo_msg_count = 0

        #Initializing the node
        rospy.init_node('quadrotor_env', anonymous=True)
            

        # Setting subscribers and publishers    
        rospy.Subscriber(quadrotor + "/" + odometry_topic, Odometry, self.odometry_callback)
        # rospy.Subscriber(quadrotor + "/" + trajectory_topic, MultiDOFJointTrajectoryPoint, self.traj_callback)
        action_publisher = rospy.Publisher(quadrotor + "/" + actuators_topic, Actuators, queue_size=1)
        
        # Waiting for reset service to appear
        rospy.wait_for_service(reset_topic)
        reset_service = rospy.ServiceProxy(reset_topic, SetModelState)

        # Resetting
        self.reset(reset_service)

        # Looping
        reset_count = freq
        count = 0

        # rospy.sleep(10)

        while True:        
            count += 1
            # try:
            #     print("waiting for the message on: ", quadrotor + "/" + odometry_topic) 
            #     odom_msg = rospy.wait_for_message(quadrotor + "/" + odometry_topic, Odometry, timeout=1)
            #     odometry_callback(odom_msg)
            # except Exception as e:
            #     print(str(e))
            #     reset(reset_service)

            actuator_msg = Actuators()
            actuator_msg.angular_velocities = thrust_val*np.array([1., 1., 1., 1.])
            action_publisher.publish(actuator_msg)
            if count % reset_count == 0:
                self.reset(reset_service)

            rospy.sleep(1.0/freq)

    def repackOdometry(self, msg):
        xyz = [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z]
        quat = [msg.pose.pose.orientation.w, msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z]
        vel_xyz = [msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z]
        vel_angular = [msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z]      
        return (xyz, quat, vel_xyz, vel_angular)

    
    def traj_callback(self, msg):
        # print("Trajectory received", msg)
        pass

    def reset(self, reset_service, pos=[0,0,0], orientation=[0,0,0,1], pos_vel=[0,0,0], angle_vel=[0,0,0]):
        # print("##############################################################")
        # print("Sending reset request ...")
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

        # print('Sending request: ', req)

        try:
            resp = reset_service(req)
            # print('RESET response: ', resp)
            return resp
        except rospy.ServiceException as e:
            print('Reset failed: ', str(e))
            raise e


    def odometry_callback(self, msg):
        # print("Odometry received", msg)

        self.odo_msg_count +=1
        time_cur = time.time()
        print('Freq: ', self.odo_msg_count / (float) (time_cur - self.start_time))

        xyz, quat, vel_xyz, vel_angular = self.repackOdometry(msg)
        # print('Odometry:')
        # print('xyz:',xyz)
        # print('quat:', quat)
        # print('vel_xyz:', vel_xyz)
        # print('vel_angular:', vel_angular)
        # print('R:', quat2R(qw=quat[0], qx=quat[1], qy=quat[2], qz=quat[3]))

TestGazebo(thrust_val=900., freq=10)
