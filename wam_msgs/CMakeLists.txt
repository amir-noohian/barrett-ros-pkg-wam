cmake_minimum_required(VERSION 2.8.3)
project(wam_msgs)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS message_generation genmsg roscpp std_msgs sensor_msgs)

#######################################
## Declare ROS messages and services ##
#######################################

## Generate messages in the 'msg' folder
add_message_files(DIRECTORY msg FILES
  FILES
  RTCartPos.msg
  RTVelocity.msg
  RTCartVel.msg
  RTJointPos.msg
  RTJointVel.msg
  RTOrtn.msg
  RTOrtnPos.msg
  RTOrtnVel.msg
  RTToolInfo.msg
  RTPose.msg
  MatrixMN.msg
  RTCartForce.msg
  WAMJointState.msg
  Gravity.msg
  CartForce.msg
)

## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
	std_msgs
	sensor_msgs
)

###################################
## catkin specific configuration ##
###################################
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
  CATKIN_DEPENDS message_runtime sensor_msgs std_msgs
)
