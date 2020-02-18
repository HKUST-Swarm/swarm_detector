#include "swarm_detector/swarm_detector.hpp"
#include "swarm_detector/fisheye_undist.hpp"

namespace swarm_detector_pkg
{

void SwarmDetector::onInit() {

}

void SwarmDetector::image_callback(const sensor_msgs::Image::ConstPtr &msg) {

}


PLUGINLIB_EXPORT_CLASS(swarm_detector_pkg::SwarmDetector, nodelet::Nodelet);
}
