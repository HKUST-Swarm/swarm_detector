#include "ros/ros.h"
#include <nodelet/nodelet.h>
#include "sensor_msgs/Image.h"
#include "sensor_msgs/CompressedImage.h"
#include <pluginlib/class_list_macros.h>
#include <eigen3/Eigen/Eigen>
#include <queue>
#include <tuple>
#include <nav_msgs/Odometry.h>
#include <swarm_msgs/swarm_fused_relative.h>
#include <swarm_msgs/Pose.h>
#include <sensor_msgs/Imu.h>
#include <opencv2/opencv.hpp>
#include <vins/FlattenImages.h>

class BaseDetector;
class DroneTracker;
struct TrackedDrone;

typedef std::tuple<ros::Time, Eigen::Quaterniond, Eigen::Vector3d> EigenPoseStamped;

namespace swarm_detector_pkg
{

class FisheyeUndist;
class SwarmDetector : public nodelet::Nodelet
{
public:
    SwarmDetector(): lookUpTable(1, 256, CV_8U)
    {
        Rvcams.push_back(Eigen::Quaterniond::Identity());                                             //0 top (up half)
        Rvcams.push_back(Eigen::Quaterniond(Eigen::AngleAxisd(-M_PI / 2, Eigen::Vector3d(1, 0, 0)))); //1 left
        Rvcams.push_back(Rvcams.back() * Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d(0, 1, 0)));      //2 front
        Rvcams.push_back(Rvcams.back() * Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d(0, 1, 0)));      //3 right
        Rvcams.push_back(Rvcams.back() * Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d(0, 1, 0)));      //4 rear
        t_down = Eigen::Quaterniond(Eigen::AngleAxisd(M_PI, Eigen::Vector3d(1, 0, 0)));
    }

private:
    FisheyeUndist *fisheye = nullptr;
    BaseDetector *detector = nullptr;
    virtual void onInit();
    ros::Subscriber fisheye_img_sub;
    ros::Subscriber fisheye_img_comp_sub;
    ros::Subscriber vins_imgs_sub;
    ros::Subscriber swarm_fused_sub;
    ros::Publisher swarm_detected_pub;
    ros::Publisher image_show_pub;
    ros::Subscriber odom_sub;
    ros::Subscriber imu_sub;
    virtual void image_callback(const sensor_msgs::Image::ConstPtr &img1_msg);
    virtual void image_comp_callback(const sensor_msgs::CompressedImageConstPtr &img1_msg);
    virtual void flattened_image_callback(const vins::FlattenImagesConstPtr & flattened);
    virtual std::vector<TrackedDrone> images_callback(const ros::Time & stamp, const std::vector<const cv::Mat *> & imgs, bool is_down_cam = false);
    virtual std::vector<TrackedDrone> virtual_cam_callback(const cv::Mat & img, int direction, Swarm::Pose, cv::Mat & debug_img, bool is_down_cam);
    virtual std::vector<TrackedDrone> virtual_cam_callback(const cv::Mat & img1, const cv::Mat & img2, int dir1, int dir2, Swarm::Pose drone_pose, cv::Mat & debug_img1, cv::Mat & debug_img2, bool is_down_cam);
    virtual std::pair<std::vector<TrackedDrone>,std::vector<Swarm::Pose>> stereo_triangulate(std::vector<TrackedDrone> tracked_up, std::vector<TrackedDrone> tracked_down);
    virtual std::vector<TrackedDrone> process_detect_result(const cv::Mat & _img, int direction, 
        std::vector<std::pair<cv::Rect2d, double>> detected_drones, Swarm::Pose pose_drone, cv::Mat & debug_img, bool has_detect, bool is_down_cam);
    virtual void odometry_callback(const nav_msgs::Odometry & odom);
    virtual void imu_callback(const sensor_msgs::Imu & imu_data);
    virtual void swarm_fused_callback(const swarm_msgs::swarm_fused_relative & sf);
    virtual void publish_tracked_drones(ros::Time stamp, Swarm::Pose local_pose_self, std::vector<TrackedDrone> drones, std::vector<Swarm::Pose> extrinsics);
    virtual Swarm::Pose get_pose_drone(const ros::Time &  stamp);
    bool debug_show = false;
    bool concat_for_tracking = false;
    bool enable_rear = false;
    bool use_tensorrt = false;
    bool pub_anonymous = false;
    int width;
    int side_height;
    int yolo_height;
    int show_width;
    bool pub_image = false;
    bool pub_track_result = false;
    bool enable_tracker;
    bool enable_triangulation;
    double detect_duration = 0.5;
    double triangulation_thres = 0.006;
    bool enable_gamma_correction;
    bool enable_up_cam;
    bool enable_down_cam;
    double gamma_;
    cv::Mat lookUpTable;


    double sf_latest = 0;
    int self_id;

    std::vector<Eigen::Quaterniond> Rvcams, Rvcams_down;
    Eigen::Quaterniond t_down;

    std::vector<DroneTracker *> drone_trackers;
    std::vector<DroneTracker *> drone_trackers_down;
    std::vector<ros::Time> last_detects;

    std::queue<std::pair<ros::Time, Swarm::Pose>> pose_buf;

    Eigen::Vector3d Pcam = Eigen::Vector3d::Zero();
    Eigen::Matrix3d Rcam = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d Rcam_down = Eigen::Matrix3d::Identity();
    Eigen::Vector3d Pcam_down = Eigen::Vector3d::Zero();

    Swarm::Pose extrinsic, extrinsic_down;

    void update_swarm_pose();

    ros::Time last_stamp;
    
    //This is in fake body frame(yaw only)
    std::map<int, Eigen::Vector3d> swarm_positions;
};

} // namespace swarm_detector_pkg
