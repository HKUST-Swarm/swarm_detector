cmake_minimum_required(VERSION 2.8.3)
project(swarm_detector)

add_compile_options(-std=c++11 -O3 -g)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  cv_bridge
  geometry_msgs
  roscpp
  rospy
  std_msgs
  swarm_msgs
  camera_model
)

catkin_package(
 INCLUDE_DIRS include
#  LIBRARIES swarm_detector
#  CATKIN_DEPENDS cv_bridge cv_bridge geometry_msgs roscpp rospy std_msgs swarm_msgs
#  DEPENDS system_lib
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

add_library(${PROJECT_NAME}
  src/swarm_detector.cpp
  src/darknet_detector.cpp
)

target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES} 
${OpenCV_LIBS}
camera_model
)