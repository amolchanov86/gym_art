#!/usr/bin/env python

import numpy as np
import time

import rospy
import rospy.rostime
from nav_msgs.msg import Odometry
from mav_msgs.msg import Actuators
from trajectory_msgs.msg import MultiDOFJointTrajectoryPoint
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState


class TestGazebo(object):
    def __init__(self):
        """
        Quadrotor position resetter
        """
        
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
        

        action_publisher = rospy.Publisher(quadrotor + "/" + actuators_topic, Actuators, queue_size=1)
        # Waiting for reset service to appear
        rospy.wait_for_service(reset_topic)
        reset_service = rospy.ServiceProxy(reset_topic, SetModelState)


        # Resetting
        self.reset(reset_service)

        for i in range(10):
            actuator_msg = Actuators()
            actuator_msg.angular_velocities = 0.*np.array([1., 1., 1., 1.])
            action_publisher.publish(actuator_msg)

            rospy.sleep(.1)

        # print('Motors are reset: ', actuator_msg)
        print("Motors are reset")


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
            print('RESET done: ', resp)
            return resp
        except rospy.ServiceException as e:
            print('Reset failed: ', str(e))
            raise e

test = TestGazebo()
