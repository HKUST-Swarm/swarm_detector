#include "swarm_detector/swarm_detector.hpp"
#include "swarm_detector/fisheye_undist.hpp"
#include "swarm_detector/darknet_detector.hpp"

namespace swarm_detector_pkg
{

void SwarmDetector::onInit() {
    ros::NodeHandle nh = this->getPrivateNodeHandle();
    fisheye_img_sub = nh.subscribe("image_raw", 3, &SwarmDetector::image_callback, this);

    std::string darknet_weights_path;
    std::string darknet_cfg;
    std::string camera_config_file;
    double fov = 235;

    nh.param<bool>("show", debug_show, false);
    nh.param<std::string>("weights", darknet_weights_path, "");
    nh.param<std::string>("darknet_cfg", darknet_cfg, "");
    nh.param<std::string>("cam_file", camera_config_file, "");
    nh.param<double>("fov", fov, 235);
    nh.param<int>("width", width, 512);


    detector = new DarknetDetector(darknet_weights_path, darknet_cfg);
    fisheye = new FisheyeUndist(camera_config_file, fov, true, width);
}

void SwarmDetector::image_callback(const sensor_msgs::Image::ConstPtr &msg) {
    cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg);
    
    int id = 2;
    
    cv::cuda::GpuMat img_cuda = fisheye->undist_id_cuda(cv_ptr->image, id);
    auto rects = detector->detect(img_cuda);

    if (debug_show) {
        cv::Mat img;
        img_cuda.download(img);
        char win_name[100] = {0};
        sprintf(win_name, "Direction %d", id);
        cv::imshow(win_name, img);
        cv::waitKey(3);
    }
}


PLUGINLIB_EXPORT_CLASS(swarm_detector_pkg::SwarmDetector, nodelet::Nodelet);
}
