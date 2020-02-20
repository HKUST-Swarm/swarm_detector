cmake_minimum_required(VERSION 2.8.3)
project(swarm_detector)

add_compile_options(-std=c++11 -O3 -g)

set(CMAKE_BUILD_TYPE "Debug")


find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  geometry_msgs
  roscpp
  rospy
  std_msgs
  swarm_msgs
  camera_model
)


SET("OpenCV_DIR"  "/usr/local/share/OpenCV/")
find_package(OpenCV 3.4 REQUIRED)

# set(ENABLE_BACKWARD true)
set(DARKNET_ROOT darknet)

if(ENABLE_BACKWARD)
    add_definitions(-D USE_BACKWARD)
endif()

catkin_package(
 INCLUDE_DIRS include
#  LIBRARIES swarm_detector
#  CATKIN_DEPENDS cv_bridge cv_bridge geometry_msgs roscpp rospy std_msgs swarm_msgs
#  DEPENDS system_lib
)
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${DARKNET_ROOT}/include
)

link_directories(${DARKNET_ROOT})

add_library(${PROJECT_NAME}
  src/swarm_detector.cpp
  src/darknet_detector.cpp
)
target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES} 
${OpenCV_LIBS}
camera_model
darknet
)

if (ENABLE_BACKWARD) 
target_link_libraries(${PROJECT_NAME}
dw
)
endif()

add_executable(test
src/testDetector.cpp
src/darknet_detector.cpp
)

target_link_libraries(test ${catkin_LIBRARIES} 
${OpenCV_LIBS}
darknet
)
