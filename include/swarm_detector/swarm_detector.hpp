#include "ros/ros.h"
#include <nodelet/nodelet.h>
#include "sensor_msgs/Image.h"
#include <pluginlib/class_list_macros.h>
#include <eigen3/Eigen/Eigen>
#include <opencv2/cudawarping.hpp>

class FisheyeUndist;
class DarknetDetector;
class DroneTracker;
struct TrackedDrone;

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
    virtual void image_callback(const sensor_msgs::Image::ConstPtr &msg);
    virtual std::vector<TrackedDrone> virtual_cam_callback(cv::cuda::GpuMat & img, int direction, cv::Mat & debug_img);

    bool debug_show = false;
    int width;
    int side_height;
    int yolo_height;
    int show_width;
    double detect_duration = 0.5;
    std::vector<Eigen::Quaterniond> Rvcams;
    Eigen::Quaterniond t_down;

    std::vector<DroneTracker*> drone_trackers;
    std::vector<ros::Time> last_detects;

    Eigen::Vector3d Pcam = Eigen::Vector3d::Zero();
    Eigen::Matrix3d Rcam = Eigen::Matrix3d::Identity();
};

}
