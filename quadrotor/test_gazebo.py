#!/usr/bin/env python


def test_gazeobo(thrust_val, freq=10.0):
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


    #Initializing the node
    rospy.init_node('quadrotor_env', anonymous=True)
    

    def repackOdometry(msg):
        xyz = [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z]
        quat = [msg.pose.pose.orientation.w, msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z]
        vel_xyz = [msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z]
        vel_angular = [msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z]      
        return (xyz, quat, vel_xyz, vel_angular)

    def odometry_callback(msg):
        # print("Odometry received", msg)
        xyz, quat, vel_xyz, vel_angular = repackOdometry(msg)
        print('Odometry:')
        print('xyz:',xyz)
        print('quat:', quat)
        print('vel_xyz:', vel_xyz)
        print('vel_angular:', vel_angular)
        print('R:', quat2R(qw=quat[0], qx=quat[1], qy=quat[2], qz=quat[3]))
    
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
    reset_service = rospy.ServiceProxy(reset_topic, SetModelState, persistent=True)

    # Resetting
    reset(reset_service)

    # Looping
    while True:
        actuator_msg = Actuators()
        actuator_msg.angular_velocities = thrust_val*np.array([1, 1, 1, 1])
        action_publisher.publish(actuator_msg)
        rospy.sleep(1.0/freq)