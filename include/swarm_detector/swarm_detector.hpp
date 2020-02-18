#include "ros/ros.h"
#include <nodelet/nodelet.h>


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

PLUGINLIB_EXPORT_CLASS(swarm_detector_pkg::SwarmDetector, nodelet::Nodelet);
}
