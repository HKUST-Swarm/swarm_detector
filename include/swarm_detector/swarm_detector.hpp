#include "ros/ros.h"
#include <nodelet/nodelet.h>
#include "sensor_msgs/Image.h"
#include <pluginlib/class_list_macros.h>

namespace swarm_detector_pkg
{
class SwarmDetector : public nodelet::Nodelet
{
public:
    SwarmDetector() {}
private:
    virtual void onInit();

    void image_callback(const sensor_msgs::Image::ConstPtr &msg);
};

}
