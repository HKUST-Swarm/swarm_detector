#include "ros/ros.h"
#include <nodelet/nodelet.h>
#include "sensor_msgs/Image.h"
#include "sensor_msgs/CompressedImage.h"
#include <pluginlib/class_list_macros.h>
#include <eigen3/Eigen/Eigen>
#include <queue>
#include <tuple>
#include <nav_msgs/Odometry.h>
#include <swarm_msgs/swarm_fused.h>
#include <swarm_msgs/Pose.h>
#include <sensor_msgs/Imu.h>
#include <opencv2/opencv.hpp>
#include <vins/FlattenImages.h>
#include "visual_detection_matcher.hpp"
#include <mutex>

class BaseDetector;
class DroneTracker;
struct TrackedDrone;

namespace Swarm {
    class DronePoseNetwork;
}

typedef std::tuple<ros::Time, Eigen::Quaterniond, Eigen::Vector3d> EigenPoseStamped;

namespace swarm_detector_pkg
{

class FisheyeUndist;
class SwarmDetector : public nodelet::Nodelet
{
public:
    SwarmDetector(): lookUpTable(1, 256, CV_8U)
    {
    }

private:
    FisheyeUndist *fisheye = nullptr;
    FisheyeUndist *fisheye_down = nullptr;
    BaseDetector *detector = nullptr;
    Swarm::DronePoseNetwork* dronepose_network = nullptr;
    virtual void onInit();
    ros::Subscriber fisheye_img_sub;
    ros::Subscriber fisheye_img_comp_sub;
    ros::Subscriber vins_imgs_sub;
    ros::Subscriber swarm_fused_sub;
    ros::Publisher swarm_detected_pub;
    ros::Publisher image_show_pub;
    ros::Publisher node_detected_pub;
    ros::Subscriber odom_sub;
    void init_camera_extrinsics() {
        std::vector<Eigen::Quaterniond> Rvcams;
        Rvcams.push_back(Eigen::Quaterniond::Identity());                                             //0 top (up half)
        Rvcams.push_back(Eigen::Quaterniond(Eigen::AngleAxisd(-M_PI / 2, Eigen::Vector3d(1, 0, 0)))); //1 left
        Rvcams.push_back(Rvcams.back() * Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d(0, 1, 0)));      //2 front
        Rvcams.push_back(Rvcams.back() * Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d(0, 1, 0)));      //3 right
        Rvcams.push_back(Rvcams.back() * Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d(0, 1, 0)));      //4 rear
        t_down = Eigen::Quaterniond(Eigen::AngleAxisd(M_PI, Eigen::Vector3d(1, 0, 0)));

        Rcams.emplace_back(Rcam);
        Rcams_down.emplace_back(Rcam_down);

        for (size_t i = 1; i < Rvcams.size(); i ++) {
            Rcams.emplace_back(Rcam*Rvcams[i]);
            Rcams_down.emplace_back(Rcam_down * t_down* Rvcams[i]);
        }
    }
    virtual void image_callback(const sensor_msgs::Image::ConstPtr &img1_msg);
    virtual void image_comp_callback(const sensor_msgs::CompressedImageConstPtr &img1_msg);
    virtual void flattened_image_callback(const vins::FlattenImagesConstPtr & flattened);
    virtual std::vector<TrackedDrone> images_callback(const ros::Time & stamp, 
        const std::vector<const cv::Mat *> & imgs, 
        std::pair<Swarm::Pose, std::map<int, Swarm::Pose>> poses_drones, 
        cv::Mat & show,
        bool is_down_cam = false);
    
    virtual std::vector<TrackedDrone> virtual_cam_callback(const ros::Time & stamp, const cv::Mat & img, 
        int direction, 
        Swarm::Pose pose, 
        bool need_detect,
        bool is_down_cam,
        cv::Mat & debug_img);
    
    virtual std::vector<TrackedDrone> virtual_cam_callback2(const ros::Time & stamp, const cv::Mat & img1, 
        const cv::Mat & img2, 
        int dir1, 
        int dir2, 
        Swarm::Pose drone_pose, 
        bool need_detect, 
        bool is_down_cam, 
        cv::Mat & debug_img1, 
        cv::Mat & debug_img2);

    virtual std::vector<TrackedDrone> pose_estimation(const ros::Time & stamp, 
        std::vector<TrackedDrone> tracked_up, 
        const std::vector<const cv::Mat *> & images_up, 
        const std::vector<const cv::Mat *> & images_down, 
        cv::Mat & _show_up, cv::Mat & _show_down);

    virtual std::vector<TrackedDrone> process_detect_result(const ros::Time & stamp,const cv::Mat & _img, 
        int direction, 
        std::vector<std::pair<cv::Rect2d, double>> detected_drones, 
        Swarm::Pose pose_drone, 
        bool has_detect, 
        bool is_down_cam,
        cv::Mat & debug_img);

    void odometry_callback(const nav_msgs::Odometry & odom);
    void swarm_fused_callback(const swarm_msgs::swarm_fused & sf);
    void publish_tracked_drones(ros::Time stamp, Swarm::Pose local_pose_self, std::vector<TrackedDrone> drones);
    void save_tracked_raw(const ros::Time & stamp, const cv::Mat & image, const TrackedDrone & tracked_drone, const Swarm::Pose & extrinsic, bool is_down);
    virtual std::pair<Swarm::Pose, std::map<int, Swarm::Pose>> get_poses_drones(const ros::Time &  stamp);
    bool detect_drone_landmarks_pose(const cv::Mat & img, TrackedDrone & tracked_drone, const Swarm::Pose & est_drone_pose, 
        Swarm::Pose & drone_pose, std::vector<Vector2d> & pts_unit, std::vector<float> & confs, std::vector<int> & inliers, 
        cv::Rect2d &rect,cv::Mat & crop, bool is_down_cam=false);
    bool debug_show = false;
    bool debug_save_tracked_raw = false;
    bool concat_for_tracking = false;
    bool enable_rear = false;
    bool use_tensorrt = false;
    bool pub_anonymous = false;
    int width;
    int side_height;
    int yolo_height;
    int show_width;
    int min_det_width;
    int min_det_height;
    bool pub_image = false;
    bool pub_track_result = false;
    bool enable_tracker;
    bool enable_triangulation;
    bool collect_data_mode;
    double detect_duration = 0.5;
    double triangulation_thres = 0.01;
    double drone_scale;
    bool debug_only_front = false;
    
    bool enable_gamma_correction;
    bool enable_up_cam;
    bool enable_down_cam;
    bool down_as_main;
    double gamma_;
    double extrinsic_ang_cov = 0.0005;
    std::string output_path;
    cv::Mat lookUpTable;


    double sf_latest = 0;
    int self_id;
    int target_count = 0;
    int img_count = 0;
    int save_img_count = 0;
    int pnpransac_inlier_min = 6;

    std::vector<Eigen::Matrix3d> Rcams, Rcams_down;
    std::vector<Vector3d> drone_landmarks;
    std::vector<cv::Point3f> drone_landmarks_cv;
    Eigen::Quaterniond t_down;

    std::vector<DroneTracker *> drone_trackers;
    std::vector<DroneTracker *> drone_trackers_down;
    VisualDetectionMatcher * visual_detection_matcher_up;
    VisualDetectionMatcher * visual_detection_matcher_down;
    ros::Time last_detect;

    std::mutex buf_lock;

    std::queue<std::pair<ros::Time, Swarm::Pose>> pose_buf;
    std::queue<std::map<int, Swarm::Pose>> swarm_positions_buf;

    Eigen::Vector3d Pcam = Eigen::Vector3d::Zero();
    Eigen::Matrix3d Rcam = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d Rcam_down = Eigen::Matrix3d::Identity();
    Eigen::Vector3d Pcam_down = Eigen::Vector3d::Zero();

    Swarm::Pose extrinsic, extrinsic_down;

    ros::Time last_stamp;
    
    //This is in fake body frame(yaw only)
};

} // namespace swarm_detector_pkg
