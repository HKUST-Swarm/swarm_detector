#include "ros/ros.h"
#include <nodelet/nodelet.h>
#include "sensor_msgs/Image.h"
#include <pluginlib/class_list_macros.h>


class FisheyeUndist;
class DarknetDetector;

namespace swarm_detector_pkg
{
class SwarmDetector : public nodelet::Nodelet
{
public:
    SwarmDetector() {}
private:
    FisheyeUndist * fisheye = nullptr;
    DarknetDetector * detector = nullptr;
    virtual void onInit();
    ros::Subscriber fisheye_img_sub;
    void image_callback(const sensor_msgs::Image::ConstPtr &msg);

    bool debug_show = false;
    int width;
};

}
