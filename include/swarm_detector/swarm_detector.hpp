#include "ros/ros.h"
#include <nodelet/nodelet.h>
#include "sensor_msgs/Image.h"
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
    SwarmDetector()
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
    ros::Subscriber vins_imgs_sub;
    ros::Subscriber swarm_fused_sub;
    ros::Publisher swarm_detected_pub;
    ros::Publisher image_show_pub;
    ros::Subscriber odom_sub;
    ros::Subscriber imu_sub;
    virtual void image_callback(const sensor_msgs::Image::ConstPtr &msg);
    virtual void flattened_image_callback(const vins::FlattenImagesConstPtr & flattened);
    virtual void images_callback(const ros::Time & stamp, const std::vector<const cv::Mat *> & imgs);
    virtual std::vector<TrackedDrone> virtual_cam_callback(const cv::Mat & img, int direction, Swarm::Pose, cv::Mat & debug_img);
    virtual std::vector<TrackedDrone> virtual_cam_callback(const cv::Mat & img1, const cv::Mat & img2, int dir1, int dir2, Swarm::Pose drone_pose, cv::Mat & debug_img1, cv::Mat & debug_img2);
    virtual std::vector<TrackedDrone> process_detect_result(const cv::Mat & _img, int direction, 
        std::vector<std::pair<cv::Rect2d, double>> detected_drones, Swarm::Pose pose_drone, cv::Mat & debug_img, bool has_detect);
    virtual void odometry_callback(const nav_msgs::Odometry & odom);
    virtual void imu_callback(const sensor_msgs::Imu & imu_data);
    virtual void swarm_fused_callback(const swarm_msgs::swarm_fused & sf);
    virtual void publish_tracked_drones(ros::Time stamp, std::vector<TrackedDrone> drones);
    virtual Swarm::Pose get_pose_drone(const ros::Time &  stamp);
    bool debug_show = false;
    bool concat_for_tracking = false;
    bool enable_rear = false;
    bool use_tensorrt = false;
    int width;
    int side_height;
    int yolo_height;
    int show_width;
    bool pub_image = false;
    bool pub_track_result = false;
    double detect_duration = 0.5;
    int self_id;

    std::vector<Eigen::Quaterniond> Rvcams;
    Eigen::Quaterniond t_down;

    std::vector<DroneTracker *> drone_trackers;
    std::vector<ros::Time> last_detects;

    std::queue<std::pair<ros::Time, Swarm::Pose>> pose_buf;

    Eigen::Vector3d Pcam = Eigen::Vector3d::Zero();
    Eigen::Matrix3d Rcam = Eigen::Matrix3d::Identity();

    void update_swarm_pose();

    ros::Time last_stamp;
    
    //This is in fake body frame(yaw only)
    std::map<int, Eigen::Vector3d> swarm_positions;
};

} // namespace swarm_detector_pkg
