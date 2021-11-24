#pragma once
#include "swarm_msgs/Pose.h"
#include <camodocal/camera_models/PinholeCamera.h>
#include "opencv2/opencv.hpp"

namespace Swarm {
class StereoBundleAdjustment {
    std::vector<Eigen::Vector3d> landmarks3d;
    std::vector<Eigen::Vector2d> landmarks_unit_1; 
    std::vector<Eigen::Vector2d> landmarks_unit_2;
    std::vector<int> landmarks2d_1_index;
    std::vector<int> landmarks2d_2_index;
    Swarm::Pose camera_pose_1;
    Swarm::Pose camera_pose_2;
    std::vector<float> confs1;
    std::vector<float> confs2;
    double focal_length = 300;
    double pixel_error = 6;

public:
    Swarm::Pose est_drone_pose;
    Swarm::Pose cam_pose_2_est;

    StereoBundleAdjustment(const std::vector<Eigen::Vector3d> & landmarks3d, 
        const std::vector<Eigen::Vector2d> & landmarks_unit_1, 
        const std::vector<Eigen::Vector2d> & landmarks_unit_2, 
        const std::vector<int> & landmarks2d_1_index,
        const std::vector<int> & landmarks2d_2_index,
        const std::vector<float> & confs1,
        const std::vector<float> & confs2,
        const Swarm::Pose & camera1, 
        const Swarm::Pose & camera2);
        
    StereoBundleAdjustment(const std::vector<Eigen::Vector3d> & landmarks3d, 
        const std::vector<Eigen::Vector2d> & landmarks_unit_1, 
        const std::vector<int> & landmarks2d_1_index,
        const std::vector<float> & confs1,
        const Swarm::Pose & camera1);

    std::pair<Swarm::Pose, Matrix6d> solve(const Swarm::Pose & initial, bool est_extrinsic=false); //Return the pose of landmarks coordinates relative to camera
    std::vector<double*> landmarks;
};
}