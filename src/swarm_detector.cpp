#include "swarm_detector/swarm_detector.hpp"
#include "swarm_detector/fisheye_undist.hpp"
#include "swarm_detector/darknet_detector.hpp"
#include "swarm_detector/drone_tracker.hpp"
#include <opencv2/core/eigen.hpp>
#include <nav_msgs/Odometry.h>

#ifdef USE_BACKWARD
#define BACKWARD_HAS_DW 1
#include <backward.hpp>
namespace backward
{
    backward::SignalHandling sh;
}
#endif

#define WARN_DT 0.005

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
    double thres, overlap_thres;
    double drone_scale;
    double p_track;
    double min_p;

    nh.param<bool>("show", debug_show, false);
    nh.param<bool>("track_matched_only", track_matched_only, false);
    nh.param<std::string>("weights", darknet_weights_path, "");
    nh.param<std::string>("darknet_cfg", darknet_cfg, "");
    nh.param<std::string>("cam_file", camera_config_file, "");
    nh.param<double>("fov", fov, 235);
    nh.param<double>("thres", thres, 0.2);
    nh.param<double>("overlap_thres", overlap_thres, 0.6);
    nh.param<int>("width", width, 512);
    nh.param<int>("show_width", show_width, 1080);
    nh.param<int>("yolo_height", yolo_height, 288);
    nh.param<std::string>("extrinsic_path", extrinsic_path, "");
    nh.param<double>("detect_duration", detect_duration, 0.0);
    nh.param<double>("drone_scale", drone_scale, 0.6);
    nh.param<double>("p_track", p_track, 0.95);
    nh.param<double>("min_p", min_p, -1);
    
    cv::Mat R, T;

    
    FILE *fh = fopen(extrinsic_path.c_str(), "r");
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

    detector = new DarknetDetector(darknet_weights_path, darknet_cfg, thres, overlap_thres);
    fisheye = new FisheyeUndist(camera_config_file, fov, true, width);

    side_height = fisheye->sideImgHeight;
    for (int i = 0; i < 6; i++) {
        last_detects.push_back(ros::Time(0));
        ROS_INFO("Init tracker on %d with P %f %f %f R", i, Pcam.x(), Pcam.y(), Pcam.z());
        std::cout << Rcam*Rvcams[i] << std::endl;
        camera_model::CameraPtr cam = fisheye->cam_side;
        if (i%5 == 0) {
            cam = fisheye->cam_top;
        }
        drone_trackers.push_back(
            new DroneTracker(Pcam, Rcam*Rvcams[i], cam, drone_scale, p_track, min_p, track_matched_only)
        );
    }

    ROS_INFO("Finish initialize swarm detector, wait for data\n");
}


cv::Scalar ScalarHSV2BGR(uchar H, uchar S, uchar V) {
    cv::Mat rgb;
    cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(H,S,V));
    cv::cvtColor(hsv, rgb, cv::COLOR_HSV2BGR);
    return cv::Scalar(rgb.data[0], rgb.data[1], rgb.data[2]);
}

std::vector<TrackedDrone> SwarmDetector::virtual_cam_callback(cv::cuda::GpuMat & img_cuda, int direction, EigenPoseStamped pose_stamped, cv::Mat & debug_img) {
    std::vector<TrackedDrone> tracked_drones;

    bool need_detect = false;
    cv::Mat img;
    img_cuda.download(img);

    std::vector<std::pair<cv::Rect2d, double>> detected_drones;
    if ((ros::Time::now() - last_detects[direction]).toSec() > detect_duration) {
        need_detect = true;
        last_detects[direction] = ros::Time::now();
    }

    drone_trackers[direction]->update_cam_pose(std::get<2>(pose_stamped), std::get<1>(pose_stamped).toRotationMatrix());
    
    if (need_detect) {
        //Detect and update to tracker
        cv::Rect roi(0, 0, img.cols, img.rows);
        if (direction ==0 || direction == 5) {
            //If top, detect half plane and track whole
            double offset = -1;
            if (direction == 0) {
                ROS_INFO("Top Upper");
                roi = cv::Rect(0, 0, img.cols, yolo_height);
            } else if(direction == 5) {
                roi = cv::Rect(0, img.rows - yolo_height, img.cols, yolo_height);
                offset = img.rows - yolo_height;
            }

            cv::Mat img_roi = img(roi);
            detected_drones = detector->detect(img_roi);
            
            if (offset > 0) {
                for (auto & rect: detected_drones) {
                    rect.first.y = rect.first.y + offset;
                }
            }
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
        if (debug_img.empty()) {
            img.copyTo(debug_img);
        }

        char idtext[20] = {0};

        for (auto ret: tracked_drones) {
            // ROS_INFO("Tracked drone ID %d@%d", ret._id, direction);
            // std::cout << ret.bbox << std::endl;
            cv::rectangle(debug_img, ret.bbox, ScalarHSV2BGR(ret.probaility*128, 255, 255), 3);
            sprintf(idtext, "[%d](%3.1f\%)", ret._id, ret.probaility*100);
            cv::Point2f pos(ret.bbox.x, ret.bbox.y - 10);
	        cv::putText(debug_img, idtext, pos, CV_FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);
        }

        /*
        for (auto ret: detected_drones) {
            cv::Point2f pos(ret.first.x+ret.first.width - 5, ret.first.y - 10);
            sprintf(idtext, "(%3.1f\%)", ret.second*100);
	        cv::putText(debug_img, idtext, pos, CV_FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);
            cv::rectangle(debug_img, ret.first, cv::Scalar(ret.second*255, 10, 10), 3);
        }*/

    }

    return tracked_drones;
}

void SwarmDetector::odometry_callback(const nav_msgs::Odometry & odom) {
    // make_tuple()
    Eigen::Quaterniond quat(
        odom.pose.pose.orientation.w, 
        odom.pose.pose.orientation.x, 
        odom.pose.pose.orientation.y, 
        odom.pose.pose.orientation.z 
    );

    Eigen::Vector3d pos(odom.pose.pose.position.x, 
        odom.pose.pose.position.y, 
        odom.pose.pose.position.z
    );

    auto tup = std::make_tuple(odom.header.stamp, quat, pos);
    pose_buf.push(tup);
}

void SwarmDetector::image_callback(const sensor_msgs::Image::ConstPtr &msg) {
    cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg);
    
    int id = 2;
    
    // cv::cuda::GpuMat img_cuda = fisheye->undist_id_cuda(cv_ptr->image, id);
    auto imgs = fisheye->undist_all_cuda(cv_ptr->image, true);

    double min_dt = 10000;
    Eigen::Vector3d Pdrone = Eigen::Vector3d::Zero();
    Eigen::Quaterniond Qdrone = Eigen::Quaterniond::Identity();
    while(pose_buf.size() > 0) {
        double dt = (std::get<0>(pose_buf.front()) - msg->header.stamp).toSec();
        if (dt < 0) {
            //Pose in buffer is older
            if (fabs(dt) < min_dt) {
                Pdrone = std::get<2>(pose_buf.front());
                Qdrone = std::get<1>(pose_buf.front());
            }
        }

        if (dt > 0) {
            //pose in buffer is newer
            if (fabs(dt) < min_dt) {
                Pdrone = std::get<2>(pose_buf.front());
                Qdrone = std::get<1>(pose_buf.front());
            }
            break;
        }

        pose_buf.pop();
    }

    if (min_dt > 0.01) {
        ROS_WARN("Pose %3.1f dt is too big!", min_dt * 1000);
    }

    std::vector<cv::Mat> debug_imgs;
    debug_imgs.resize(5);
    for (int i = 0; i < 6; i++) {
        // ROS_INFO("Using img %d, direction %d", i%5, i);
        virtual_cam_callback(imgs[i%5], i, std::make_tuple(msg->header.stamp, Qdrone, Pdrone), debug_imgs[i%5]);
    }

    if (debug_show) {
        cv::Mat _show;
        cv::resize(debug_imgs[0], _show, cv::Size(side_height, side_height));
        for (int i = 1; i < 5; i ++) {
            cv::hconcat(_show, debug_imgs[i], _show);
        }

        double f_resize = ((double)show_width)/(double) _show.cols;
        cv::resize(_show, _show, cv::Size(), f_resize, f_resize);

        cv::imshow("DroneTracker", _show);
        cv::waitKey(3);
    }
}


PLUGINLIB_EXPORT_CLASS(swarm_detector_pkg::SwarmDetector, nodelet::Nodelet);
}
