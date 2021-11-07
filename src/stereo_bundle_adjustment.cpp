#include "swarm_detector/stereo_bundle_adjustment.hpp"
#include "ceres/ceres.h"
#include "swarm_detector/reprojection_error.h"

using namespace ceres;

namespace Swarm {
StereoBundleAdjustment::StereoBundleAdjustment(const std::vector<Eigen::Vector3d> & _landmarks3d, 
        const std::vector<Eigen::Vector2d> & _landmarks_unit_1, 
        const std::vector<Eigen::Vector2d> & _landmarks_unit_2, 
        const std::vector<int> & _landmarks2d_1_index,
        const std::vector<int> & _landmarks2d_2_index,
        const std::vector<float> & _confs1,
        const std::vector<float> & _confs2,
        const Swarm::Pose & _camera1, 
        const Swarm::Pose & _camera2):
    landmarks3d(_landmarks3d),
    landmarks_unit_1(_landmarks_unit_1),
    landmarks_unit_2(_landmarks_unit_2),
    landmarks2d_1_index(_landmarks2d_1_index),
    landmarks2d_2_index(_landmarks2d_2_index),
    confs1(_confs1),
    confs2(_confs2),
    camera_pose_1(_camera1),
    camera_pose_2(_camera2)
{}


StereoBundleAdjustment::StereoBundleAdjustment(const std::vector<Eigen::Vector3d> & _landmarks3d, 
        const std::vector<Eigen::Vector2d> & _landmarks_unit_1, 
        const std::vector<int> & _landmarks2d_1_index,
        const std::vector<float> & _confs1,
        const Swarm::Pose & _camera1):
    landmarks3d(_landmarks3d),
    landmarks_unit_1(_landmarks_unit_1),
    landmarks2d_1_index(_landmarks2d_1_index),
    confs1(_confs1),
    camera_pose_1(_camera1)
{}


Swarm::Pose StereoBundleAdjustment::solve(const Swarm::Pose & initial) {
    Problem problem;
    double pose_drone[7] = {0};
    initial.to_vector(pose_drone);
    for (auto index: landmarks2d_1_index) {
        auto cf = ReprojectionError::Create(landmarks3d[index], landmarks_unit_1[index], 
                camera_pose_1, confs1[index]);
        problem.AddResidualBlock(cf, nullptr, pose_drone);
    }
    
    for (auto index: landmarks2d_2_index) {
        auto cf = ReprojectionError::Create(landmarks3d[index], landmarks_unit_2[index], 
                camera_pose_2, confs2[index]);
        problem.AddResidualBlock(cf, nullptr, pose_drone);
    }

    ceres::LocalParameterization* pose_local_parameterization = new ceres::ProductParameterization (new ceres::IdentityParameterization(3), 
        new ceres::EigenQuaternionParameterization());

    problem.SetParameterization(pose_drone, pose_local_parameterization);
    ceres::Solver::Options options;
    options.max_num_iterations = 100;
    options.linear_solver_type = ceres::DENSE_QR;
    // options.minimizer_type = ceres::LINE_SEARCH;
    options.max_solver_time_in_seconds = 0.1;
    Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    est_drone_pose = Swarm::Pose(pose_drone);
    // std::cout << summary.FullReport() << std::endl;
    // std::cout << "Initial" << initial.tostr() << "Ret" << est_drone_pose.tostr() << std::endl;
    return est_drone_pose;
}

}