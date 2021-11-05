cmake_minimum_required(VERSION 2.8.3)
project(swarm_detector)

add_compile_options(-std=c++14 -O3 -g)

set(CMAKE_BUILD_TYPE "Release")


find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  geometry_msgs
  roscpp
  rospy
  std_msgs
  swarm_msgs
  vins
  camera_models
  swarm_loop
)

SET(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")

SET("OpenCV_DIR"  "/usr/local/share/OpenCV/")
find_package(OpenCV 3.4 REQUIRED)

set(ENABLE_BACKWARD true)
set(ENABLE_DARKNET false)
set(ENABLE_TENSORRT true)

set(DARKNET_ROOT $ENV{HOME}/source/darknet)
set(TRT_YOLO_ROOT $ENV{HOME}/source/yolo-tensorrt)

if(ENABLE_BACKWARD)
  add_definitions(-D USE_BACKWARD)
endif()

if(ENABLE_DARKNET) 
  add_definitions(-D USE_DARKNET)
  link_directories(${DARKNET_ROOT})
endif()

if(ENABLE_TENSORRT)
  add_definitions(-D USE_TENSORRT)
  include_directories(${TRT_YOLO_ROOT}/modules)
  include_directories(${TRT_YOLO_ROOT}/extra)
  link_directories(${TRT_YOLO_ROOT}/build)

  set(Torch_DIR "$ENV{HOME}/source/libtorch/share/cmake/Torch")
  find_package(Torch REQUIRED)

  include_directories("$ENV{HOME}/source/yolo-tensorrt/modules/")
  include_directories("$ENV{HOME}/source/TensorRT-7.1.3.4/include")

  link_directories(${TENSORRT_ROOT}/lib)
  link_directories("$ENV{HOME}/source/yolo-tensorrt/build/")
  find_package(CUDA)
  include_directories(${CUDA_INCLUDE_DIRS} ${TORCH_INCLUDE_DIRS})
  add_definitions("-D USE_TENSORRT")
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
  /usr/include/eigen3
)



add_library(${PROJECT_NAME}
  src/swarm_detector.cpp
  src/darknet_detector.cpp
  src/visual_detection_matcher.cpp
  src/dronepose_network.cpp
)

target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES} 
  ${OpenCV_LIBS}
  camera_models
  munkres
  loop_cnn
)

if(ENABLE_DARKNET) 
target_link_libraries(${PROJECT_NAME}
  darknet
)
endif()

if(ENABLE_TENSORRT)
  target_link_libraries(${PROJECT_NAME} detector)
endif()

if (ENABLE_BACKWARD) 
target_link_libraries(${PROJECT_NAME}
dw
${TORCH_LIBRARIES}
)
endif()

