#include "ros/ros.h"
#include <nodelet/nodelet.h>
#include "sensor_msgs/Image.h"
#include <pluginlib/class_list_macros.h>
#include <eigen3/Eigen/Eigen>
#include <opencv2/cudawarping.hpp>
#include <queue>
#include <tuple>
#include <nav_msgs/Odometry.h>
#include <swarm_msgs/swarm_fused_relative.h>
#include <swarm_msgs/Pose.h>
#include <sensor_msgs/Imu.h>

class FisheyeUndist;
class DarknetDetector;
class DroneTracker;
struct TrackedDrone;

typedef std::tuple<ros::Time, Eigen::Quaterniond, Eigen::Vector3d> EigenPoseStamped;

namespace swarm_detector_pkg
{
class SwarmDetector : public nodelet::Nodelet
{
public:
    SwarmDetector() {
        Rvcams.push_back(Eigen::Quaterniond::Identity()); //0 top (up half)
        Rvcams.push_back(Eigen::Quaterniond(Eigen::AngleAxisd(-M_PI / 2, Eigen::Vector3d(1, 0, 0)))); //1 left
        Rvcams.push_back(Rvcams.back() * Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d(0, 1, 0))); //2 front
        Rvcams.push_back(Rvcams.back() * Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d(0, 1, 0))); //3 right
        Rvcams.push_back(Rvcams.back() * Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d(0, 1, 0))); //4 rear
        Rvcams.push_back(Eigen::Quaterniond::Identity()); //5 top (down half)
        t_down = Eigen::Quaterniond(Eigen::AngleAxisd(M_PI, Eigen::Vector3d(1, 0, 0)));
    }
private:
    FisheyeUndist * fisheye = nullptr;
    DarknetDetector * detector = nullptr;
    virtual void onInit();
    ros::Subscriber fisheye_img_sub;
    ros::Subscriber swarm_fused_sub;
    ros::Publisher swarm_detected_pub;
    ros::Subscriber odom_sub;
    ros::Subscriber imu_sub;
    virtual void image_callback(const sensor_msgs::Image::ConstPtr &msg);
    virtual std::vector<TrackedDrone> virtual_cam_callback(cv::cuda::GpuMat & img, int direction, Swarm::Pose, cv::Mat & debug_img);
    virtual void odometry_callback(const nav_msgs::Odometry & odom);
    virtual void imu_callback(const sensor_msgs::Imu & imu_data);
    virtual void swarm_fused_callback(const swarm_msgs::swarm_fused_relative & sf);
    virtual void publish_tracked_drones(ros::Time stamp, std::vector<TrackedDrone> drones);
    bool debug_show = false;
    bool concat_for_tracking = false;
    int width;
    int side_height;
    int yolo_height;
    int show_width;
    double detect_duration = 0.5;
    std::vector<Eigen::Quaterniond> Rvcams;
    Eigen::Quaterniond t_down;

    std::vector<DroneTracker*> drone_trackers;
    std::vector<ros::Time> last_detects;

    std::queue<std::pair<ros::Time, Swarm::Pose>> pose_buf;

    Eigen::Vector3d Pcam = Eigen::Vector3d::Zero();
    Eigen::Matrix3d Rcam = Eigen::Matrix3d::Identity();

    void update_swarm_pose();
    
    //This is in fake body frame(yaw only)
    std::map<int, Eigen::Vector3d> swarm_positions;
};

}
