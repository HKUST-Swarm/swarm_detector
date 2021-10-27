#include <eigen3/Eigen/Eigen>
#include "drone_tracker.hpp"
#include "swarm_detector/fisheye_undist.hpp"
#include "km.h"

#define MAX_COST 100000
namespace swarm_detector_pkg {
class VisualDetectionMatcher {

    Swarm::Pose pose_drone;
    std::vector<Swarm::Pose> swarm_est_poses;
    std::vector<int> swarm_est_ids;
    Eigen::Vector3d tic;

    std::vector<Swarm::Pose> pose_cams;

    double drone_scale;
    double min_p;
    double accept_direction_thres;
    double accept_inv_depth_thres;

    bool is_concat_track = false;
    int single_width = false;

    FisheyeUndist * fisheye;

    std::vector<Vector3d> boundbox3d_corners;

    bool show = false;

public:
    VisualDetectionMatcher(Eigen::Vector3d _tic, 
            std::vector<Eigen::Matrix3d> rcams,
            FisheyeUndist* _fisheye,
            double _accept_direction_thres,
            double _accept_inv_depth_thres,
            bool debug_show):
        tic(_tic),
        fisheye(_fisheye),
        accept_direction_thres(_accept_direction_thres),
        accept_inv_depth_thres(_accept_inv_depth_thres),
        show(debug_show)
    {
        for (auto R : rcams) {
            pose_cams.emplace_back(Swarm::Pose(_tic, Quaterniond(R)));
        }

        double w_2 = 0.2;
        double h_2 = 0.2;
        double z_max = 0.1;
        double z_min = -0.1;
        Vector3d Gc_imu(-0.06, 0, 0);

        boundbox3d_corners = std::vector<Vector3d>{
            Vector3d(w_2, h_2, z_max) + Gc_imu,
            Vector3d(w_2, -h_2, z_max) + Gc_imu,
            Vector3d(-w_2, h_2, z_max) + Gc_imu,
            Vector3d(-w_2, -h_2, z_max) + Gc_imu,
            Vector3d(w_2, h_2, z_min) + Gc_imu,
            Vector3d(w_2, -h_2, z_min) + Gc_imu,
            Vector3d(-w_2, h_2, z_min) + Gc_imu,
            Vector3d(-w_2, -h_2, z_min) + Gc_imu,
        };
    } 

    void set_swarm_state(Swarm::Pose _pose_drone, std::map<int, Swarm::Pose> _swarm_positions) {
        // Note: Swarm Position should not contain self drone.
        pose_drone = _pose_drone;
        swarm_est_poses.clear();
        swarm_est_ids.clear();
        for (auto it: _swarm_positions) {
            swarm_est_ids.emplace_back(it.first);
            swarm_est_poses.emplace_back(it.second);
        }
    }

    std::pair<bool, cv::Rect2d> reproject_drone_to_vcam(int direction, Swarm::Pose est, Swarm::Pose cur) const {
        cv::Rect2d reproject_bbox;
        MatrixXd corners2d_body(2, boundbox3d_corners.size());
        auto cam = fisheye->cam_side;
        if (direction == 0) {
            cam = fisheye->cam_top;
        }

        // std::cout << "direction" << direction << "Est Pose\t" << est.tostr() << "\tcam pose\t" << cur.tostr()  << std::endl;
        for (size_t i = 0; i < boundbox3d_corners.size(); i ++) {
            auto corner = boundbox3d_corners[i];
            auto corner3d_body = est * corner;
            corner3d_body = cur.apply_inv_pose_to(corner3d_body);
            // std::cout << "corner3d\t" << (est * corner).transpose() << "body\t" << corner3d_body.transpose() << std::endl;

            if (corner3d_body.z() < 0) {
                return std::make_pair(false, reproject_bbox);
            }

            Vector2d corner2d;
            cam->spaceToPlane(corner3d_body, corner2d);
            corners2d_body.block(0, i, 2, 1) = corner2d;
        }


        auto xs = corners2d_body.block(0, 0, 1, corners2d_body.cols());
        auto ys = corners2d_body.block(1, 0, 1, corners2d_body.cols());

        reproject_bbox.x = xs.minCoeff();
        reproject_bbox.width = xs.maxCoeff() - reproject_bbox.x;
        reproject_bbox.y = ys.minCoeff();
        reproject_bbox.height = ys.maxCoeff() - reproject_bbox.y;

        // std::cout << "corner3d_body\n" << corners2d_body << std::endl;
        // std::cout << "corner3d_body_col_xs\n" << xs << std::endl;
        // std::cout << "corner3d_body_col_ys\n" << ys << std::endl;
        // std::cout << "bbox\n" << reproject_bbox << std::endl;

        if (reproject_bbox.x + reproject_bbox.width < 0 || reproject_bbox.x > cam->imageWidth() || 
                reproject_bbox.y + reproject_bbox.height < 0 || reproject_bbox.y > cam->imageWidth()) {
            return std::make_pair(false, reproject_bbox);
        }

        return std::make_pair(true, reproject_bbox);

    }

    double cost_det_to_est(TrackedDrone det, Swarm::Pose est) const {
        Swarm::Pose pose_cam_local = pose_drone*pose_cams[det.direction];
        auto ret = reproject_drone_to_vcam(det.direction, est, pose_cam_local);
        if (ret.first) {
            return 1-det.overlap(ret.second);
        }

        return MAX_COST;
    }

    double cost_det_to_tracked(TrackedDrone det, TrackedDrone tracked) const {
        if (det.direction != tracked.direction) {
            return MAX_COST;
        }
        return 1 - det.overlap(tracked);
    }

    Eigen::MatrixXf construct_cost_matrix(const std::vector<TrackedDrone> & detected_targets, std::vector<Swarm::Pose> swarm_poses, 
            const std::vector<TrackedDrone> & tracked_drones)  const {
        Eigen::MatrixXf cost(swarm_poses.size() + tracked_drones.size(), detected_targets.size());
        //      det1 det2 ... detn
        // Est1 cost ...........
        // Est2 cost ...........
        // ...
        // Estn cost ...........
        // trk2 cost ...........
        // ....
        // trk1 cost ...........
        // trkn cost ...........
        
        for (size_t j = 0; j < detected_targets.size(); j ++) {
            for (size_t i = 0; i < swarm_est_ids.size(); i ++) {
                cost(i, j) = cost_det_to_est(detected_targets[j], swarm_est_poses[i]);
                ROS_INFO("Cost for det %d est %d: %f", j, i, cost(i, j));

            }

            for (size_t _i = 0; _i < tracked_drones.size(); _i ++) {
                auto i = _i + swarm_est_ids.size();
                // cost(i, j) = ...;
                cost(i, j) = cost_det_to_tracked(detected_targets[j], tracked_drones[i]);
            }
        }

        ROS_INFO("SWARM_DETECT: cost for KM:");
        std::cout << cost <<std::endl;
        return cost;
    }

    void draw_debug(std::vector<cv::Mat> & debug_imgs) const {
        for (size_t i = 0; i < debug_imgs.size(); i ++) {
            if (debug_imgs[i].empty()) {
                continue;
            }
            Swarm::Pose pose_cam_local = pose_drone*pose_cams[i];
            for (size_t j = 0; j < swarm_est_ids.size(); j++) {
                auto ret = reproject_drone_to_vcam(i, swarm_est_poses[j], pose_cam_local);
                if (ret.first) {
                    cv::rectangle(debug_imgs[i], ret.second, cv::Scalar(255, 0, 0), 2);
                }
            }
        }
    }

    std::vector<TrackedDrone> match_targets(std::vector<TrackedDrone> detected_targets, std::vector<TrackedDrone> tracked_drones) const {
        std::vector<TrackedDrone> matched_targets;
        ROS_INFO("[SWARM_DETECT] match_targets %ld detected with %ld tracked and %ld est", 
            detected_targets.size(), tracked_drones.size(), swarm_est_ids.size());
        if (detected_targets.size() > 0 && tracked_drones.size() + swarm_est_poses.size() > 0) {
            auto cost = construct_cost_matrix(detected_targets, swarm_est_poses, tracked_drones);
            KM km(cost);
            auto matched = km.getMatch(1); //This returns the detected targets matched to the object/
            std::cout << "Matched size" << matched.size() << std::endl;
            for (size_t i = 0; i < detected_targets.size(); i ++) {
                auto matched_to = matched[i];
                auto assigned_id = -1;
                if (matched_to < swarm_est_ids.size()) {
                    assigned_id = swarm_est_ids[matched_to];
                } else if (matched_to < swarm_est_ids.size() + tracked_drones.size()) {
                    assigned_id = tracked_drones[matched_to-swarm_est_ids.size()]._id;
                }
                printf("%d->%d (matched_to %d)", i, assigned_id, matched_to);

                if (assigned_id >= 0) {
                    auto target = detected_targets[i];
                    target._id = assigned_id;
                    matched_targets.emplace_back(target);
                }
            }
            std::cout << std::endl;
            return matched_targets;
        } 
        
        return std::vector<TrackedDrone>();
    }
};

}