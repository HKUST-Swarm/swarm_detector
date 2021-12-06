#include <eigen3/Eigen/Eigen>
#include "drone_tracker.hpp"
#include "swarm_detector/fisheye_undist.hpp"
#include "swarm_msgs/swarm_types.hpp"

#define MAX_DRONE_ID 1000

// #define DEBUG_OUTPUT
namespace swarm_detector_pkg {
class VisualDetectionMatcher {

    Swarm::Pose pose_drone;
    std::vector<Swarm::Pose> swarm_est_poses;
    std::vector<int> swarm_est_ids;
    Eigen::Vector3d tic;

    std::vector<Swarm::Pose> pose_cams;

    double drone_scale;
    double min_p;
    double accept_overlap_thres;

    bool is_concat_track = false;
    int single_width = false;

    FisheyeUndist * fisheye;

    std::vector<Vector3d> boundbox3d_corners;

    bool show = false;

    bool enable_anonymous;
    int self_id = 0;

public:
    Vector3d Gc_imu = Vector3d(-0.06, 0, 0.00);

    VisualDetectionMatcher(Eigen::Vector3d _tic, 
            std::vector<Eigen::Matrix3d> rcams,
            FisheyeUndist* _fisheye,
            double _accept_overlap_thres,
            int _self_id,
            bool _enable_anonymous,
            bool debug_show);

    void set_swarm_state(const Swarm::Pose & _pose_drone, const std::map<int, Swarm::Pose> & _swarm_positions);

    std::pair<bool, Eigen::Vector2d> reproject_point_to_vcam(int direction, Eigen::Vector3d corner, Swarm::Pose est, Swarm::Pose cur) const;

    std::pair<bool, cv::Rect2d> reproject_drone_to_vcam(int direction, Swarm::Pose est, Swarm::Pose cur, const std::vector<Vector3d> & corners) const;
    std::pair<bool, cv::Rect2d> reproject_drone_to_vcam(int direction, Swarm::Pose est, Swarm::Pose cur) const;

    double cost_det_to_est(TrackedDrone det, Swarm::Pose est) const;

    double cost_det_to_tracked(TrackedDrone det, TrackedDrone tracked) const;

    Eigen::MatrixXd construct_cost_matrix(const std::vector<TrackedDrone> & detected_targets, std::vector<Swarm::Pose> swarm_poses, 
            const std::vector<TrackedDrone> & tracked_drones)  const;

    void draw_debug(std::vector<cv::Mat> & debug_imgs) const;

    std::vector<TrackedDrone> match_targets(std::vector<TrackedDrone> detected_targets, std::vector<TrackedDrone> tracked_drones) const;
};

}