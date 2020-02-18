#include "swarm_detector/swarm_detector.hpp"
#include "swarm_detector/fisheye_undist.hpp"
#include "swarm_detector/darknet_detector.hpp"
#include "swarm_detector/drone_tracker.hpp"
#include <opencv2/core/eigen.hpp>

#ifdef USE_BACKWARD
#define BACKWARD_HAS_DW 1
#include <backward.hpp>
namespace backward
{
    backward::SignalHandling sh;
}
#endif

namespace swarm_detector_pkg
{

void SwarmDetector::onInit() {
    ros::NodeHandle nh = this->getPrivateNodeHandle();
    fisheye_img_sub = nh.subscribe("image_raw", 3, &SwarmDetector::image_callback, this);

    std::string darknet_weights_path;
    std::string darknet_cfg;
    std::string camera_config_file;
    std::string extrinsic_path;
    bool track_matched_only = false;
    double fov = 235;

    nh.param<bool>("show", debug_show, false);
    nh.param<bool>("track_matched_only", track_matched_only, false);
    nh.param<std::string>("weights", darknet_weights_path, "");
    nh.param<std::string>("darknet_cfg", darknet_cfg, "");
    nh.param<std::string>("cam_file", camera_config_file, "");
    nh.param<double>("fov", fov, 235);
    nh.param<int>("width", width, 512);
    nh.param<int>("yolo_height", yolo_height, 288);
    nh.param<std::string>("extrinsic_path", extrinsic_path, "");
    
    cv::Mat R, T;

    
    FILE *fh = fopen(extrinsic_path.c_str(),"r");
    if(fh == NULL){
        ROS_WARN("config_file dosen't exist; Assume identity camera pose");
    } else {
        cv::FileStorage fsSettings(extrinsic_path, cv::FileStorage::READ);

        fsSettings["R"] >> R;
        fsSettings["T"] >> T;
        cv::cv2eigen(R, Rcam);
        cv::cv2eigen(T, Pcam);
        fsSettings.release();
    }

    detector = new DarknetDetector(darknet_weights_path, darknet_cfg);
    fisheye = new FisheyeUndist(camera_config_file, fov, true, width);

    for (int i = 0; i < 6; i++) {
        last_detects.push_back(ros::Time::now());
        ROS_INFO("Init tracker on %d with P %f %f %f R", i, Pcam.x(), Pcam.y(), Pcam.z());
        std::cout << Rcam*Rvcams[i] << std::endl;
        drone_trackers.push_back(
            new DroneTracker(track_matched_only, Pcam, Rcam*Rvcams[i])
        );
    }

    ROS_INFO("Finish initialize swarm detector, wait for data\n");
}

void SwarmDetector::virtual_cam_callback(cv::cuda::GpuMat & img_cuda, int direction) {
    std::vector<TrackedDrone> tracked_drones;

    bool need_detect = false;
    cv::Mat img;
    img_cuda.download(img);

    std::vector<cv::Rect2d> detected_drones;
    if ((ros::Time::now() - last_detects[direction]).toSec() > detect_duration) {
        need_detect = true;
        last_detects[direction] = ros::Time::now();
    }
    
    if (need_detect) {
        //Detect and update to tracker
        cv::Rect roi(0, 0, img.cols, img.rows);
        if (direction ==0 || direction == 5) {
            //If top, detect half plane and track whole
            if (direction == 0) {
                roi = cv::Rect(0, 0, img.cols, yolo_height);
            } else if(direction == 5) {
                roi = cv::Rect(0, img.rows - yolo_height, img.cols, yolo_height);
            }

            cv::Mat img_roi = img(roi);
            detected_drones = detector->detect(img_roi);

            // if (debug_show) {
            //     char win_name[10] = {0};
            //     sprintf(win_name, "ROI %d", direction);
            //     cv::imshow(win_name, img_roi);
            // }
        } else {
            detected_drones = detector->detect(img);
        }

        tracked_drones = drone_trackers[direction]->process_detect(img, detected_drones);
    } else {
        //Track only
        tracked_drones = drone_trackers[direction]->track(img);
    }

    if (debug_show) {

        char win_name[100] = {0};
        sprintf(win_name, "Direction %d", direction);
        
        for (auto ret: detected_drones) {
            cv::rectangle(img, ret, cv::Scalar(0, 255, 255));
        }

        cv::imshow(win_name, img);
        cv::waitKey(3);
    }
}

void SwarmDetector::image_callback(const sensor_msgs::Image::ConstPtr &msg) {
    cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg);
    
    int id = 2;
    
    // cv::cuda::GpuMat img_cuda = fisheye->undist_id_cuda(cv_ptr->image, id);
    auto imgs = fisheye->undist_all_cuda(cv_ptr->image);

    for (int i = 0; i < 6; i++) {
        ROS_INFO("Using img %d, direction %d", i%5, i);
        virtual_cam_callback(imgs[i%5], i);
    }
    
}


PLUGINLIB_EXPORT_CLASS(swarm_detector_pkg::SwarmDetector, nodelet::Nodelet);
}
