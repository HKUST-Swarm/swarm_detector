#pragma once

#include "ceres/rotation.h"
#include "swarm_msgs/Pose.h"

namespace Swarm
{
struct ReprojectionError
{
    ReprojectionError(const Eigen::Vector3d &_landmark_drone, double _observed_x, double _observed_y, const Swarm::Pose &_camera_pose, double _conf) : landmark_drone(_landmark_drone), observed_x(_observed_x), observed_y(_observed_y), conf(_conf)
    {
        Swarm::Pose _cam_pose_inv = _camera_pose.inverse();
        quat_cam_inv = _cam_pose_inv.att();
        T_cam_inv = _cam_pose_inv.pos();
        printf("Landmark %.1f %.1f %.1f observed %.3f %.3f, camera_pose_inv %s conf %.2f\n",
                    _landmark_drone.x(), _landmark_drone.y(), _landmark_drone.z(), _observed_x, _observed_y,
                    _camera_pose.inverse().tostr().c_str(),
                    _conf);
    }

    template <typename T>
    bool operator()(const T *const drone_pose,
                    T *residuals) const
    {
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> p_a(drone_pose);
        Eigen::Map<const Eigen::Quaternion<T>> q_a(drone_pose + 3);
        Eigen::Matrix<T, 3, 1> landmark_body = q_a*landmark_drone.template cast<T>() + p_a;

        Eigen::Matrix<T, 3, 1> p = quat_cam_inv.template cast<T>()*landmark_body + T_cam_inv;

        const T predicted_x = p(0) / p(2);
        const T predicted_y = p(1) / p(2);

        T _conf = (T)conf;
        residuals[0] = (predicted_x - observed_x) * _conf;
        residuals[1] = (predicted_y - observed_y) * _conf;

        return true;
    }

    static ceres::CostFunction *Create(const Eigen::Vector3d &_landmark_drone, const Eigen::Vector2d &observed_unit, const Swarm::Pose &_camera_pose, double conf)
    {
        double observed_x = observed_unit.x();
        double observed_y = observed_unit.y();

        return (new ceres::AutoDiffCostFunction<ReprojectionError, 2, 7>(
            new ReprojectionError(_landmark_drone, observed_x, observed_y, _camera_pose, conf)));
    }

    double observed_x;
    double observed_y;
    double conf;
    Quaterniond quat_cam_inv;
    Vector3d T_cam_inv;
    Vector3d landmark_drone;
};
}