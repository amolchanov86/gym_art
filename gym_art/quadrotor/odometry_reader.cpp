#include <ros/ros.h>
#include <iostream>
#include <fstream>
#include <Eigen/Eigen>

#include <mav_msgs/default_topics.h>
#include "std_msgs/String.h"


#include <nav_msgs/Odometry.h>
#include <ros/ros.h>

#include <mav_msgs/conversions.h>
#include <mav_msgs/eigen_mav_msgs.h>

#include "rotors_control/common.h"

#define MAX_RPM 838

void OdometryCallback(const nav_msgs::OdometryConstPtr& odometry_msg) {
    ROS_INFO_ONCE("NNHoveringController got first odometry message.");

    EigenOdometry odometry;
    eigenOdometryFromMsg(odometry_msg, &odometry);
    
    // position error 
    Eigen::Vector3d position_error;
    position_error = odometry.position - command_trajectory_.position_W;
    position_x_e = position_error[0];
    position_y_e = position_error[1];
    position_z_e = position_error[2];

    ROS_INFO("CONTROLLER_INPUT_STATES: %f %f %f", odometry.position[0], odometry.position[1], odometry.position[2]);
    ROS_INFO("CONTROLLER_ERROR: %f %f %f", position_x_e, position_y_e, position_z_e);

    // orientation
    rotational_matrix = odometry.orientation.toRotationMatrix();

    // linear velocity error
    // Transform velocity to world frame 
    Eigen::Vector3d velocity_W = rotational_matrix * odometry.velocity;

    linear_vx_e = velocity_W[0];
    linear_vy_e = velocity_W[1];
    linear_vz_e = velocity_W[2];    

    // assuming angular velocity is always (0, 0, 0)
    angular_vx = odometry_msg->twist.twist.angular.x;
    angular_vy = odometry_msg->twist.twist.angular.y;
    angular_vz = odometry_msg->twist.twist.angular.z;

    ofstream myfile ("/dev/shm/odometry.txt");
    if (myfile.is_open())
    {
      myfile << odometry.position[0] << odometry.position[1] << odometry.position[2] 
             << odometry.velocity[0] << odometry.velocity[1] << odometry.velocity[2]
             << odometry.orientation[0] << odometry.orientation[1] << odometry.orientation[2] << odometry.orientation[3]
             << angular_vx << angular_vy << angular_vz;
      myfile.close();
    }

}


int main(int argc, char **argv)
{

  ros::init(argc, argv, "listener");

  ros::NodeHandle n;

  ros::Subscriber sub = n.subscribe("/hummingbird/odometry_sensor1/odometry", 1, OdometryCallback);

  ros::spin();

  return 0;
}