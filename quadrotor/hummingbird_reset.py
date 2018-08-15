#!/usr/bin/env python
import numpy as np
import rospy
import rospy.rostime
from nav_msgs.msg import Odometry
from mav_msgs.msg import Actuators
from trajectory_msgs.msg import MultiDOFJointTrajectoryPoint
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState


def R2quat(rot):
	# print('R2quat: ', rot, type(rot))
	R = rot.reshape([3,3])
	w = np.sqrt(1.0 + R[0,0] + R[1,1] + R[2,2]) / 2.0
	w4 = (4.0 * w)
	x = (R[2,1] - R[1,2]) / w4
	y = (R[0,2] - R[2,0]) / w4
	z = (R[1,0] - R[0,1]) / w4
	return np.array([w,x,y,z])


def set_state(pos, vel, rot, omega):
	reset_topic = "/gazebo/set_model_state"
	reset_service = rospy.ServiceProxy(reset_topic, SetModelState)

	print('ENV: set_state: pos, rot, vel, omega', pos, rot, vel, omega, type(rot), rot.shape)
	req = ModelState()
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

	print('DYN: Sending RESET request: ', req)

	try:
	    resp = reset_service(req)
	    print('DYN: RESET response: ', resp)
	    return resp
	except rospy.ServiceException as e:
		print('ERROR: DYN: Reset failed: ', str(e))


set_state(pos=[0., 0., 0.], vel=[0.,0.,0.], rot=np.eye(3), omega=[0.,0.,0.])