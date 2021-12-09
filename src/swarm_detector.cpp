#include "swarm_detector/swarm_detector.hpp"
#include "swarm_detector/fisheye_undist.hpp"
#include "swarm_detector/darknet_detector.hpp"
#include "swarm_detector/tensorrt_detector.hpp"
#include "swarm_detector/drone_tracker.hpp"
#include "swarm_detector/dronepose_network.hpp"
#include <opencv2/core/eigen.hpp>
#include <nav_msgs/Odometry.h>
#include <swarm_msgs/swarm_detected.h>
#include <swarm_msgs/node_detected_xyzyaw.h>
#include <swarm_msgs/swarm_fused.h>
#include <swarm_msgs/Pose.h>
#include <chrono>
#include <vins/FlattenImages.h>
#include <swarm_msgs/swarm_lcm_converter.hpp>
#include <algorithm>
#include <swarm_detector/stereo_bundle_adjustment.hpp>

#define MAX_DETECTOR_ID 1000000

#define VCAMERA_TOP 0
#define VCAMERA_LEFT 1
#define VCAMERA_FRONT 2
#define VCAMERA_RIGHT 3
#define VCAMERA_REAR 4

#define BBOX_DEPTH_OFFSET 0.12

#define DEBUG_SHOW_HCONCAT

#define DETECTION_MARGIN 5

using namespace swarm_msgs;
#ifdef USE_BACKWARD
#define BACKWARD_HAS_DW 1
#include <backward.hpp>
namespace backward
{
backward::SignalHandling sh;
}
#endif

#define WARN_DT 0.005
#define DISP_RESCALE 4
namespace swarm_detector_pkg
{

void SwarmDetector::onInit()
{
    ros::NodeHandle nh = this->getMTPrivateNodeHandle();

    cv::setNumThreads(1);
    std::string darknet_weights_path;
    std::string darknet_cfg;
    std::string camera_config_file;
    std::string camera_down_config_file;
    std::string extrinsic_path;
    std::string drone_pose_network_model;
    bool track_matched_only = false;
    bool tensorrt_fp16 = false;
    double fov = 235;
    double thres, overlap_thres;
    double p_track;
    double min_p;
    double acpt_overlap_thres;
    int drone_pose_width;
    int drone_pose_height;
    int drone_pose_zoom;

    nh.param<bool>("show", debug_show, false);
    nh.param<bool>("track_matched_only", track_matched_only, false);
    nh.param<bool>("pub_image", pub_image, true);
    nh.param<bool>("use_tensorrt", use_tensorrt, true);
    nh.param<bool>("pub_track_result", pub_track_result, true);
    nh.param<bool>("debug_save_tracked_raw", debug_save_tracked_raw, false);
    nh.param<bool>("tensorrt_fp16", tensorrt_fp16, true);
    nh.param<std::string>("weights", darknet_weights_path, "");
    nh.param<std::string>("darknet_cfg", darknet_cfg, "");
    nh.param<std::string>("cam_file", camera_config_file, "");
    nh.param<std::string>("cam_file_down", camera_down_config_file, "");
    nh.param<double>("fov", fov, 235);
    nh.param<double>("thres", thres, 0.2);
    nh.param<double>("overlap_thres", overlap_thres, 0.6);
    nh.param<int>("width", width, 512);
    nh.param<int>("show_width", show_width, 1080);
    nh.param<int>("min_det_width", min_det_width, 50);
    nh.param<int>("min_det_height", min_det_height, 50);
    nh.param<int>("yolo_height", yolo_height, 288);
    nh.param<std::string>("extrinsic_path", extrinsic_path, "");
    nh.param<double>("detect_duration", detect_duration, 0.0);
    nh.param<double>("drone_scale", drone_scale, 0.6);
    nh.param<double>("p_track", p_track, 0.95);
    nh.param<double>("gamma", gamma_, 1.6);
    nh.param<double>("min_p", min_p, -1);
    nh.param<int>("drone_id", self_id, -1);
    nh.param<bool>("pub_anonymous", pub_anonymous, false);
    nh.param<bool>("enable_tracker", enable_tracker, false);
    nh.param<bool>("enable_triangulation", enable_triangulation, true);
    nh.param<bool>("enable_gamma_correction", enable_gamma_correction, true);
    nh.param<bool>("enable_up_cam", enable_up_cam, false);
    nh.param<bool>("enable_down_cam", enable_down_cam, true);
    nh.param<bool>("down_as_main", down_as_main, true);
    nh.param<bool>("collect_data_mode", collect_data_mode, false);
    nh.param<bool>("debug_only_front", debug_only_front, false);
    nh.param<std::string>("output_path", output_path, "/root/output/");
    nh.param<std::string>("drone_pose_network_model", drone_pose_network_model, "");
    nh.param<int>("drone_pose_width", drone_pose_width, 128);
    nh.param<int>("drone_pose_height", drone_pose_height, 128);
    nh.param<int>("drone_pose_zoom", drone_pose_zoom, 4);
    nh.param<int>("pnpransac_inlier_min", pnpransac_inlier_min, 6);

    //Is in %
    nh.param<double>("acpt_overlap_thres", acpt_overlap_thres, 20);
    nh.param<double>("triangulation_thres", triangulation_thres, 0.1);
    cv::Mat R, T;

    FILE *fh = fopen(extrinsic_path.c_str(), "r");
    ROS_INFO("[SWARM_DETECT] Try to read extrinsic from %s camera from %s", extrinsic_path.c_str(), camera_config_file.c_str());
    if (fh == NULL)
    {
        ROS_WARN("[SWARM_DETECT] Config_file dosen't exist; Assume identity camera pose");
	    Rcam = Eigen::Matrix3d::Identity();
	    Pcam = Eigen::Vector3d(0.105, 0.004614, 0.0898);
	    std::cout << "Translation" << Pcam;
        enable_rear = true;
    }
    else
    {
        cv::FileStorage fsSettings(extrinsic_path, cv::FileStorage::READ);
        cv::Mat _T;
        fsSettings["body_T_cam0"] >> _T;
        fsSettings["image_width"] >> width;
        fsSettings["enable_rear_side"] >> enable_rear;

        Eigen::Matrix4d T;
        cv::cv2eigen(_T, T);
        Rcam = T.block<3, 3>(0, 0);
        Pcam = T.block<3, 1>(0, 3);
        extrinsic = Swarm::Pose(Rcam, Pcam);

        fsSettings["body_T_cam1"] >> _T;
        cv::cv2eigen(_T, T);
        Rcam_down = T.block<3, 3>(0, 0);
        Pcam_down = T.block<3, 1>(0, 3);
        extrinsic_down = Swarm::Pose(Rcam_down, Pcam_down);

        fsSettings.release();

        ROS_INFO("[SWARM_DETECT] Camera width %d, Pose", width);
        std::cout << "R" << Rcam << std::endl;
        std::cout << "P" << Pcam.transpose() << std::endl;

        std::cout << "Rd" << Rcam_down << std::endl;
        std::cout << "Pd" << Pcam_down.transpose() << std::endl;
    }

    init_camera_extrinsics();

    if(use_tensorrt) {
        detector = new TensorRTDetector(darknet_weights_path, darknet_cfg, thres, overlap_thres, tensorrt_fp16);
    } else {
        detector = new DarknetDetector(darknet_weights_path, darknet_cfg, thres, overlap_thres);
    }

    dronepose_network = new Swarm::DronePoseNetwork(drone_pose_network_model, drone_pose_width, drone_pose_height, drone_pose_zoom, false);

    fisheye = new FisheyeUndist(camera_config_file, fov, true, width);
    fisheye_down = new FisheyeUndist(camera_down_config_file, fov, true, width);
    side_height = fisheye->sideImgHeight;


    last_detect = ros::Time(0);

    for (int i = 0; i < 5; i++) {
        ROS_INFO("[SWARM_DETECT] Init tracker on %d with P %f %f %f R", i, Pcam.x(), Pcam.y(), Pcam.z());
        std::cout << Rcams[i] << std::endl;

        camodocal::PinholeCameraPtr cam = fisheye->cam_side;

        double focal_length = fisheye->f_side;
        if (i == 0) {
            focal_length = fisheye->f_center;
        }
        
        drone_trackers.emplace_back(new DroneTracker(i, p_track, min_p, drone_scale, focal_length));

        drone_trackers_down.emplace_back(new DroneTracker(i, p_track, min_p, drone_scale, focal_length));
    }

    visual_detection_matcher_up = new VisualDetectionMatcher(Pcam, Rcams, fisheye, acpt_overlap_thres, self_id, pub_anonymous, debug_show);
    visual_detection_matcher_down = new VisualDetectionMatcher(Pcam_down, Rcams_down, fisheye, acpt_overlap_thres, self_id, pub_anonymous, debug_show);
    auto Gc_imu = visual_detection_matcher_up->Gc_imu;
    drone_landmarks = std::vector<Vector3d>{
        Vector3d(102.15,121.79,52.2)/1000.0 + Gc_imu,
        Vector3d(-102.15,121.79,52.2)/1000.0 + Gc_imu,
        Vector3d(102.15,-121.79,52.2)/1000.0 + Gc_imu,
        Vector3d(-102.15,-121.79,52.2)/1000.0 + Gc_imu,
        Vector3d(0, 0, 115)/1000.0 + Gc_imu,
        Vector3d(0, 0, -55)/1000.0 + Gc_imu,
        Vector3d(98.92,117.94,-71)/1000.0 + Gc_imu,
        Vector3d(-98.92,117.94,-71)/1000.0 + Gc_imu,
        Vector3d(98.92,-117.94,-71)/1000.0 + Gc_imu,
        Vector3d(-98.92,-117.94,-71)/1000.0 + Gc_imu,
        Vector3d(-60,0, 165)/1000.0 + Gc_imu};
    
    for (auto lm: drone_landmarks) {
        drone_landmarks_cv.emplace_back(cv::Point3f(lm.x(), lm.y(), lm.z()));
    }

    uchar* p = lookUpTable.ptr();
    for( int i = 0; i < 256; ++i)
        p[i] = cv::saturate_cast<uchar>(pow(i / 255.0, gamma_) * 255.0);

    swarm_detected_pub = nh.advertise<swarm_msgs::swarm_detected>("/swarm_detection/swarm_detected_raw", 3);
    node_detected_pub = nh.advertise<swarm_msgs::node_detected>("/swarm_drones/node_detected_6d", 3);
    image_show_pub = nh.advertise<sensor_msgs::Image>("show", 1);

    // odom_sub = nh.subscribe("odometry", 3, &SwarmDetector::odometry_callback, this);
    vins_imgs_sub = nh.subscribe("vins_flattened", 3, &SwarmDetector::flattened_image_callback, this);
    swarm_fused_sub = nh.subscribe("swarm_fused", 3, &SwarmDetector::swarm_fused_callback, this);
    fisheye_img_sub = nh.subscribe("image_raw", 3, &SwarmDetector::image_callback, this);
    fisheye_img_comp_sub = nh.subscribe("image_comp", 3, &SwarmDetector::image_comp_callback, this);
    
    ROS_INFO("[SWARM_DETECT] Finish initialize swarm detector, wait for data\n");
}

void SwarmDetector::swarm_fused_callback(const swarm_msgs::swarm_fused & sf) {
    // TODO: Here
    // for (unsigned int i = 0; i < sf.ids.size(); i ++ ) {
    //     swarm_positions[sf.ids[i]] = Eigen::Vector3d(
    //         sf.local_drone_position[i].x,
    //         sf.local_drone_position[i].y,
    //         sf.local_drone_position[i].z
    //     );
    // }

    sf_latest = sf.header.stamp.toSec();
    std::map<int, Swarm::Pose> swarm_positions;
    std::pair<ros::Time, Swarm::Pose> self_pose_stamped;
    for (size_t i = 0; i < sf.ids.size(); i ++) {
        if (sf.ids[i] != self_id) {
            swarm_positions[sf.ids[i]] = Swarm::Pose(sf.local_drone_position[i], sf.local_drone_rotation[i]);
        } else {
            self_pose_stamped = std::make_pair(sf.header.stamp, Swarm::Pose(sf.local_drone_position[i], sf.local_drone_rotation[i]));
        }
    }
    buf_lock.lock();
    pose_buf.push(self_pose_stamped);
    swarm_positions_buf.push(swarm_positions);
    buf_lock.unlock();

}

cv::Scalar ScalarHSV2BGR(uchar H, uchar S, uchar V) {
    cv::Mat rgb;
    cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(H, S, V));
    cv::cvtColor(hsv, rgb, cv::COLOR_HSV2BGR);
    return cv::Scalar(rgb.data[0], rgb.data[1], rgb.data[2]);
}

std::vector<TrackedDrone> SwarmDetector::virtual_cam_callback(const ros::Time & stamp, const cv::Mat & _img, int direction, Swarm::Pose pose_drone,
        bool need_detect,
        bool is_down_cam,
        cv::Mat & debug_img) { 
    std::vector<std::pair<cv::Rect2d, double>> detected_targets;
    if (need_detect) {
        if (debug_only_front && direction != VCAMERA_FRONT) {
            ROS_INFO("Skip for debug reason");
        } else {
            TicToc t_d;
            detected_targets = detector->detect(_img);
            ROS_INFO("[SWARM_DETECT] Detect squared cost %fms \n", t_d.toc());
        }
    }

    return this->process_detect_result(stamp, _img, direction, detected_targets, pose_drone, need_detect, is_down_cam, debug_img);
}

std::vector<TrackedDrone> SwarmDetector::virtual_cam_callback2(const ros::Time & stamp, const cv::Mat & img1, 
        const cv::Mat & img2, 
        int dir1, int dir2, 
        Swarm::Pose pose_drone, 
        bool need_detect, 
        bool is_down_cam,
        cv::Mat & debug_img1, 
        cv::Mat & debug_img2) { 
    BBoxProbArray det1, det2;

    if (need_detect) {
        TicToc t_d;
        auto ret = detector->detect(img1, img2);
        det1 = ret.first;
        det2 = ret.second;
        ROS_INFO("[SWARM_DETECT] Detect squared of 2 images cost %fms\n", t_d.toc());
    }

    if (debug_only_front) {
        if (dir1 == VCAMERA_FRONT) {
            det2.clear();
        } else {
            det1.clear();
            det2.clear();
        }
    }

    if (need_detect || enable_tracker) {
        auto track1 = this->process_detect_result(stamp, img1, dir1, det1, pose_drone, need_detect, 
            is_down_cam, debug_img1);
        auto track2 = this->process_detect_result(stamp, img2, dir2, det2, pose_drone, need_detect,
            is_down_cam, debug_img2);

        track1.insert(track1.end(), track2.begin(), track2.end());
        return track1;
    }
    return std::vector<TrackedDrone>();
}

Swarm::Pose PnPRestoCamPose(cv::Mat rvec, cv::Mat tvec) {
    cv::Mat r;
    cv::Rodrigues(rvec, r);
    Eigen::Matrix3d R_pnp, R_w_c_old;
    cv::cv2eigen(r, R_pnp);
    R_w_c_old = R_pnp.transpose();
    Eigen::Vector3d T_pnp, T_w_c_old;
    cv::cv2eigen(tvec, T_pnp);
    T_w_c_old = R_w_c_old * (-T_pnp);

    return Swarm::Pose(R_w_c_old, T_w_c_old);
}

bool SwarmDetector::detect_drone_landmarks_pose(const cv::Mat & img, TrackedDrone & _target, 
            const Swarm::Pose & cam_pose, Swarm::Pose & est_drone_pose, std::vector<Vector2d> & pts_unit, std::vector<float> & conf,
            std::vector<int> & inliers, cv::Rect2d &rect, cv::Mat & crop, bool is_down_cam) {
    // Act DronePose Here
    //Make
    rect.x = std::max(_target.bbox.x - _target.bbox.width /10, 0.0);
    rect.y = std::max(_target.bbox.y - _target.bbox.height*0.7, 0.0);
    rect.width = std::min(_target.bbox.width*1.2, img.cols - rect.x);
    rect.height = std::min(_target.bbox.height*2, img.rows - rect.y);
    camodocal::PinholeCameraConstPtr cam = fisheye->cam_side;
    if (is_down_cam) {
        if (_target.direction == 0) {
            cam = fisheye_down->cam_top;
        } else {
            cam = fisheye_down->cam_side;
        }
    } else {
        if (_target.direction == 0) {
            cam = fisheye->cam_top;
        }
    }


    if (rect.width > min_det_width && rect.height > min_det_height) {
        auto landmarks_with_conf = dronepose_network->inference(img(rect));
        auto landmarks2d = landmarks_with_conf.first;
        conf = landmarks_with_conf.second;
        //Recove to whole image
        for (auto & landmark: landmarks2d) {
            landmark.x += rect.x;
            landmark.y += rect.y;

            Eigen::Vector2d lm2d(landmark.x, landmark.y);
            Eigen::Vector3d lm3d;
            cam->liftProjective(lm2d, lm3d);
            pts_unit.emplace_back(Vector2d(lm3d.x()/lm3d.z(), lm3d.y()/lm3d.z()));
        }

        cv::Mat r, t, D;
        cv::Mat K = (cv::Mat_<double>(3, 3) << cam->getParameters().fx(), 0, cam->getParameters().cx(), 
                                                0, cam->getParameters().fy(), cam->getParameters().cy(), 
                                                0, 0, 1.0);
        cv::solvePnPRansac(drone_landmarks_cv, landmarks2d, K, D, r, t, false,  100,  5, 0.99,  inliers, cv::SOLVEPNP_EPNP);
        auto p_cam_in_target_frame = PnPRestoCamPose(r, t);
        est_drone_pose = (p_cam_in_target_frame*(cam_pose.inverse())).inverse();

        if (debug_show || pub_image)
        {
            img(rect).copyTo(crop);
            cv::resize(crop, crop, cv::Size(), DISP_RESCALE, DISP_RESCALE);
            for (int i = 0; i < landmarks_with_conf.first.size(); i++) {
                uint8_t conf = landmarks_with_conf.second[i]*255;
                cv::Mat im_gray = (cv::Mat_<uint8_t>(1, 1) << conf);
                cv::Mat im_color;
                cv::applyColorMap(im_gray, im_color, cv::COLORMAP_JET);
                cv::circle(crop, landmarks_with_conf.first[i]*DISP_RESCALE, 2, im_color.at<cv::Vec3b>(0, 0), -1);
            }

            for (auto index: inliers) {
                cv::circle(crop, landmarks_with_conf.first[index]*DISP_RESCALE, 10, cv::Scalar(0, 255, 0), 2);
            }

            char title[128] = {0};
            if (inliers.size() >= pnpransac_inlier_min) {
                sprintf(title, "PnP POSE OK %s", est_drone_pose.tostr().c_str());
                cv::putText(crop, title, cv::Point2f(20, 30), CV_FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
            } else {
                sprintf(title, "PnP POSE Failed %s", est_drone_pose.tostr().c_str());
                cv::putText(crop, title, cv::Point2f(20, 30), CV_FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
            }
        }
    }

    return inliers.size() >= pnpransac_inlier_min;
}

std::vector<TrackedDrone> SwarmDetector::process_detect_result(const ros::Time & stamp,
        const cv::Mat & _img, 
        int direction, 
        BBoxProbArray detected_targets, 
        Swarm::Pose pose_drone, 
        bool need_detect, 
        bool is_down_cam,
        cv::Mat & debug_img) {
    std::vector<TrackedDrone> detected_targets_drones;

    double alpha = 1.5;
    cv::Mat img = _img;
    double focal_length = fisheye->f_side;
    if (direction == 0) {
        focal_length = fisheye->f_center;
    }
    
    double z_calib = 0;
    bool has_detected_or_tracked = false;

    for (auto det : detected_targets) {
        double x1 = det.first.x;
        double y1 = det.first.y;
        double x2 = det.first.x + det.first.width;
        double y2 = det.first.y + det.first.height;
        if (x1 < DETECTION_MARGIN || x2 > _img.cols - DETECTION_MARGIN || y1 < DETECTION_MARGIN || y2 > _img.rows - DETECTION_MARGIN) {
            continue;
        }

        //Act DronePose Here
        //Make
        auto _target = TrackedDrone(-1, det.first, ((double)det.first.width)/(drone_scale*focal_length), 
            det.second, direction);
        detected_targets_drones.emplace_back(_target);
 
        if (debug_save_tracked_raw) {
            if (is_down_cam) {
                save_tracked_raw(stamp, _img, _target, Swarm::Pose(Rcams_down[direction], Pcam_down), is_down_cam);
            } else {
                save_tracked_raw(stamp, _img, _target, Swarm::Pose(Rcams[direction], Pcam), is_down_cam);
            }
        }
        has_detected_or_tracked = true;
    }


    //Track only
    TicToc tic;
    std::vector<TrackedDrone> tracked_drones;
    if(need_detect && pub_track_result || enable_tracker) {
        if (!is_down_cam) {
            tracked_drones = drone_trackers[direction]->track(img);
        } else {
            tracked_drones = drone_trackers_down[direction]->track(img);
        }
    }
    ROS_INFO("[SWARM_DETECT] Visual trackers@%d cost %.1fms", direction, tic.toc());

    if (!has_detected_or_tracked && debug_save_tracked_raw) {
        if (!(direction == 4 && !enable_rear)) {
            auto _target = TrackedDrone(-1, cv::Rect2d(-1, -1, 0, 0), -1, -1, direction);
            if (is_down_cam) {
                save_tracked_raw(stamp, _img, _target, Swarm::Pose(Rcams_down[direction], Pcam_down), is_down_cam);
            } else {
                save_tracked_raw(stamp, _img, _target, Swarm::Pose(Rcams[direction], Pcam), is_down_cam);
            }
        }
    }

    if (debug_show || pub_image)
    {
        if (debug_img.empty())
        {
            img.copyTo(debug_img);
        }

        char idtext[20] = {0};

        for (auto ret: detected_targets) {
            //Draw in top
            cv::Point2f pos(ret.first.x+ret.first.width - 5, ret.first.y - 15);
            // sprintf(idtext, "(%3.1f\%)", ret.second*100);
	        // cv::putText(debug_img, idtext, pos, CV_FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 0, 0), 2);
            // cv::rectangle(debug_img, ret.first, cv::Scalar(255, 10, 0), 2);
            cv::rectangle(debug_img, ret.first, cv::Scalar(0, 0, 255), 2);
            // sprintf(idtext, "(%3.1f,%3.1f)", ret.first.x + ret.first.width/2, ret.first.y + ret.first.height/2);
	        // cv::putText(debug_img, idtext, pos, CV_FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 0, 0), 2);
            cv::Point2f p(ret.first.x + ret.first.width/2, ret.first.y + ret.first.height/2);
            cv::circle(debug_img, p, 4, cv::Scalar(0, 0, 255), 2);
        }

        for (auto ret : tracked_drones)
        {
            // ROS_INFO("Tracked drone ID %d@%d", ret._id, direction);
            // std::cout << ret.bbox << std::endl;
            cv::rectangle(debug_img, ret.bbox, ScalarHSV2BGR(ret.probaility * 128 + 128, 255, 255), 3);
            cv::Point2f p(ret.bbox.x + ret.bbox.width/2, ret.bbox.y + ret.bbox.height/2);
            cv::circle(debug_img, p, 4, ScalarHSV2BGR(ret.probaility * 128 + 128, 255, 255), 2);
            sprintf(idtext, "[%d](%3.1f%%)", ret._id, ret.probaility * 100);
            //Draw bottom
            cv::Point2f pos(ret.bbox.x, ret.bbox.y + ret.bbox.height + 20);
            cv::putText(debug_img, idtext, pos, CV_FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);
        }
    }
    return detected_targets_drones;
}

void SwarmDetector::odometry_callback(const nav_msgs::Odometry & odom) {
    // auto tup = std::make_pair(odom.header.stamp, Swarm::Pose(odom.pose.pose));
    // pose_buf.push(tup);
}

void SwarmDetector::publish_tracked_drones(ros::Time stamp, Swarm::Pose local_pose_self, std::vector<TrackedDrone> drones) {
    swarm_detected sd;
    sd.header.stamp = stamp;
    sd.self_drone_id = self_id;
    sd.is_6d_detect = false;
    auto &detected_nodes = sd.detected_nodes;
    for (int i = 0; i <drones.size(); i++) {
        auto tdrone = drones[i];
        if (!pub_anonymous && tdrone._id >= MAX_DRONE_ID) {
            continue;
        }
        node_detected nd;
        nd.local_pose_self = local_pose_self.to_ros_pose();
        nd.is_yaw_valid = true;
        nd.self_drone_id = self_id;
        nd.remote_drone_id = tdrone._id;
        nd.header.stamp = stamp;
        nd.probaility = tdrone.probaility;
        nd.id = MAX_DETECTOR_ID*self_id + tdrone.detect_no;
        // std::cout << "covariance pose_drone\n" << tdrone.covariance << std::endl;
        
        memcpy(nd.relative_pose.covariance.data(), tdrone.covariance.data(), sizeof(double)*36);
        auto pose_4d = Swarm::Pose::DeltaPose(local_pose_self, local_pose_self*tdrone.relative_pose, true);
        nd.relative_pose.pose = pose_4d.to_ros_pose();
        nd.dof_4 = true;
        ROS_INFO("[SWARM_DETECT] RP_body %s RP_body_4d %s local_pose %s target_pose_local %s", tdrone.relative_pose.tostr().c_str(), pose_4d.tostr().c_str(), 
            local_pose_self.tostr().c_str(), (local_pose_self*tdrone.relative_pose).tostr().c_str());

        //Set cov here
        ROS_INFO("[SWARM_DETECT] Pub drone %d number %d rel pose (4d) %s", tdrone._id, 
                img_count, pose_4d.tostr().c_str());

        node_detected_pub.publish(nd);

        detected_nodes.push_back(nd);
    }
    if (sd.detected_nodes.size() > 0) {
        swarm_detected_pub.publish(sd);
    }
}

cv::cuda::GpuMat concat_side(const std::vector<cv::cuda::GpuMat> & arr, bool enable_rear_side=true) {
    int cols = arr[1].cols;
    int rows = arr[1].rows;
    if (enable_rear_side) {
        cv::cuda::GpuMat NewImg(rows, cols*4, arr[1].type()); 
        for (int i = 1; i < 5; i ++) {
            arr[i].copyTo(NewImg(cv::Rect(cols * (i-1), 0, cols, rows)));
        }
        return NewImg;
    } else {
        cv::cuda::GpuMat NewImg(rows, cols*3, arr[1].type()); 
        for (int i = 1; i < 4; i ++) {
            arr[i].copyTo(NewImg(cv::Rect(cols * (i-1), 0, cols, rows)));
        }
        return NewImg;
    }
}


//TODO: Pose drone should be convert by base_coor.
std::pair<Swarm::Pose, std::map<int, Swarm::Pose>> SwarmDetector::get_poses_drones(const ros::Time & stamp) {
    buf_lock.lock();
    
    double min_dt = 10000;

    Swarm::Pose pose_drone;
    std::map<int, Swarm::Pose> swarm_positions;
    while(pose_buf.size() > 0) {
        double dt = (pose_buf.front().first - stamp).toSec();
        // ROS_INFO("DT %f", dt);
        if (dt < 0) {
            //Pose in buffer is older
            if (fabs(dt) < min_dt) {
                pose_drone = pose_buf.front().second;
                swarm_positions = swarm_positions_buf.front();
                min_dt = fabs(dt);
            }
        }

        if (dt > 0)
        {
            //pose in buffer is newer
            if (fabs(dt) < min_dt) {
                pose_drone = pose_buf.front().second;
                swarm_positions = swarm_positions_buf.front();
                min_dt = fabs(dt);
            }
            break;
        }

        pose_buf.pop();
        swarm_positions_buf.pop();
    }


    if (min_dt > 0.1)
    {
        ROS_WARN("[SWARM_DETECT] Pose dt %3.1fms  is too big!", min_dt * 1000);
    }

    buf_lock.unlock();
    return std::make_pair(pose_drone, swarm_positions);
}


void SwarmDetector::image_callback(const sensor_msgs::Image::ConstPtr &msg) {

    auto cv_ptr = cv_bridge::toCvShare(msg, "rgb8");
    auto imgs = fisheye->undist_all_cuda(cv_ptr->image, true, enable_rear);
    int total_imgs = imgs.size();
    std::vector<cv::Mat> img_cpus;
    std::vector<const cv::Mat*> img_cpus_ptrs;
    img_cpus.resize(total_imgs);
    for (unsigned int i = 0; i < total_imgs; i++) {
        imgs[i].download(img_cpus[i]);
        img_cpus_ptrs.push_back(&(img_cpus[i]));
    }
    auto ret = get_poses_drones(msg->header.stamp);
    cv::Mat show;
    images_callback(msg->header.stamp, img_cpus_ptrs, ret, show);
    if (debug_show) {
        char title[100] = {0};
        sprintf(title, "DroneTracker@%d", self_id);
        cv::imshow(title, show);
        cv::waitKey(3);
    } else if (pub_image){
        cv_bridge::CvImage cvimg;
        cvimg.encoding = sensor_msgs::image_encodings::BGR8;
        cvimg.image = show;
        image_show_pub.publish(cvimg);
    }
}

void SwarmDetector::image_comp_callback(const sensor_msgs::CompressedImageConstPtr &img_msg) {

    auto image = cv::imdecode(img_msg->data, cv::IMREAD_COLOR);
    auto imgs = fisheye->undist_all_cuda(image, true, enable_rear);
    int total_imgs = imgs.size();
    std::vector<cv::Mat> img_cpus;
    std::vector<const cv::Mat*> img_cpus_ptrs;
    img_cpus.resize(total_imgs);
    for (unsigned int i = 0; i < total_imgs; i++) {
        imgs[i].download(img_cpus[i]);
        img_cpus_ptrs.push_back(&(img_cpus[i]));
    }
    img_cpus_ptrs.push_back(&image);
    auto ret = get_poses_drones(img_msg->header.stamp);
    cv::Mat show;
    images_callback(img_msg->header.stamp, img_cpus_ptrs, ret, show);
    if (debug_show) {
        char title[100] = {0};
        sprintf(title, "DroneTracker@%d", self_id);
        cv::imshow(title, show);
        cv::waitKey(3);
    } else if (pub_image){
        cv_bridge::CvImage cvimg;
        cvimg.encoding = sensor_msgs::image_encodings::BGR8;
        cvimg.image = show;
        image_show_pub.publish(cvimg);
    }
}

cv::Mat img_empty;
void SwarmDetector::flattened_image_callback(const vins::FlattenImagesConstPtr &flattened) {
    if (fabs(sf_latest - flattened->header.stamp.toSec()) > 0.1) 
    {
        ROS_WARN("[SWARM_DETECT] sf_latest.t - flattened.t high %.1fms", (sf_latest - flattened->header.stamp.toSec())*1000);
    }

    std::vector<const cv::Mat *> img_cpus, img_cpus_down;
    std::vector<cv_bridge::CvImageConstPtr> ptrs;
    for (int i = 0; i < flattened->up_cams.size(); i++) {
        if (flattened->up_cams[i].width > 0) {
            auto cv_ptr = cv_bridge::toCvShare(flattened->up_cams[i], flattened);
            // auto cv_ptr = cv_bridge::toCvCopy(flattened->up_cams[i], sensor_msgs::image_encodings::BGR8);
            ptrs.push_back(cv_ptr);
            img_cpus.push_back(&(cv_ptr->image));
        } else {
            img_cpus.push_back(&img_empty);
        }
    }

    for (int i = 0; i < flattened->down_cams.size(); i++) {
        if (flattened->down_cams[i].width > 0) {
            auto cv_ptr = cv_bridge::toCvShare(flattened->down_cams[i], flattened);
            // auto cv_ptr = cv_bridge::toCvCopy(flattened->up_cams[i], sensor_msgs::image_encodings::BGR8);
            ptrs.push_back(cv_ptr);
            img_cpus_down.push_back(&(cv_ptr->image));
        } else {
            img_cpus_down.push_back(&img_empty);
        }
    }

    auto ret = get_poses_drones(flattened->header.stamp);
    std::vector<TrackedDrone> tracked_drones_up, tracked_drones_down;
    cv::Mat show_up, show_down;
    if (enable_up_cam && !down_as_main || collect_data_mode) {
        tracked_drones_up = images_callback(flattened->header.stamp, img_cpus, ret, show_up);
    }

    // //Not detect on down but do feature track.
    if (enable_down_cam && down_as_main || collect_data_mode) {
        tracked_drones_down = images_callback(flattened->header.stamp, img_cpus_down, ret, show_down, true);
    }

    auto pose_drone = ret.first;
    visual_detection_matcher_down->set_swarm_state(pose_drone, ret.second);
    auto tracked_drones = pose_estimation(flattened->header.stamp, tracked_drones_up, img_cpus, img_cpus_down, show_up, show_down);
    publish_tracked_drones(flattened->header.stamp, pose_drone, tracked_drones);

    if (debug_show || pub_image) {
        cv::Mat show;
        if (enable_up_cam && enable_down_cam && enable_triangulation) {
            cv::vconcat(show_up, show_down, show);
        } else if(enable_up_cam) {
            show = show_up;
        } else if(enable_down_cam) {
            show = show_down;
        }

        char title[100] = {0};
        if (debug_save_tracked_raw) {
            sprintf(title, "%s/DroneTracker/DroneTracker%d-%06d.jpg", output_path.c_str(), self_id, img_count);
            cv::imwrite(title, show);
        }
        
        double f_resize = ((double)show_width) / (double)show.cols;
        cv::resize(show, show, cv::Size(), f_resize, f_resize);

        if (debug_show) {
            char title[100] = {0};
            sprintf(title, "DroneTracker@%d", self_id);
            cv::imshow(title, show);
            cv::waitKey(3);
        } else if (pub_image){
            cv_bridge::CvImage cvimg;
            cvimg.encoding = sensor_msgs::image_encodings::BGR8;
            cvimg.image = show;
            image_show_pub.publish(cvimg);
        }
    }
    img_count ++;
}


double triangulatePoint3DPts(Eigen::Matrix<double, 3, 4> &Pose0, Eigen::Matrix<double, 3, 4> &Pose1,
                        Eigen::Vector3d &point0, Eigen::Vector3d &point1, Eigen::Vector3d &point_3d)
{
    //TODO:Rewrite this for 3d point
    
    double p0x = point0[0];
    double p0y = point0[1];
    double p0z = point0[2];

    double p1x = point1[0];
    double p1y = point1[1];
    double p1z = point1[2];

    Eigen::Matrix4d design_matrix = Eigen::Matrix4d::Zero();
    design_matrix.row(0) = p0x * Pose0.row(2) - p0z*Pose0.row(0);
    design_matrix.row(1) = p0y * Pose0.row(2) - p0z*Pose0.row(1);
    design_matrix.row(2) = p1x * Pose1.row(2) - p1z*Pose1.row(0);
    design_matrix.row(3) = p1y * Pose1.row(2) - p1z*Pose1.row(1);
    Eigen::Vector4d triangulated_point;
    triangulated_point =
              design_matrix.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>();
    point_3d(0) = triangulated_point(0) / triangulated_point(3);
    point_3d(1) = triangulated_point(1) / triangulated_point(3);
    point_3d(2) = triangulated_point(2) / triangulated_point(3);


    Eigen::MatrixXd pts(4, 1);
    pts << point_3d.x(), point_3d.y(), point_3d.z(), 1;
    Eigen::MatrixXd errs = design_matrix*pts;
    return errs.norm()/ errs.rows(); 
}

void SwarmDetector::save_tracked_raw(const ros::Time & stamp, const cv::Mat & image, const TrackedDrone & tracked_drone, const Swarm::Pose & extrinsic, bool is_down) {
    char path[256] = {0};
    char path_txt[256] = {0};
    if (!is_down) {
        sprintf(path, "%s/DetectedRaw/DroneDetected%d-%06d.jpg", output_path.c_str(), self_id, save_img_count);
        sprintf(path_txt, "%s/DetectedRaw/DroneDetected%d-%06d.txt", output_path.c_str(), self_id, save_img_count);
    }  else {
        sprintf(path, "%s/DetectedRaw/DroneDetected%d_down-%06d.jpg", output_path.c_str(), self_id, save_img_count);
        sprintf(path_txt, "%s/DetectedRaw/DroneDetected%d_down-%06d.txt", output_path.c_str(), self_id, save_img_count);
    }

    imwrite(path, image);
    FILE * f = fopen(path_txt, "w");
    auto cam_ptr = static_cast<camodocal::PinholeCameraPtr>(fisheye->cam_side);
    if (tracked_drone.direction == 0) {
        cam_ptr = static_cast<camodocal::PinholeCameraPtr>(fisheye->cam_top);
    }
    
    fprintf(f, "%.4f %.2f %.2f %.2f %.2f\n%s\n%.3f %.3f %.3f %.3f", stamp.toSec(), tracked_drone.bbox.x, tracked_drone.bbox.y, 
        tracked_drone.bbox.width, tracked_drone.bbox.height, extrinsic.tostr(true).c_str(), 
            cam_ptr->getParameters().cx(),cam_ptr->getParameters().cy(),
            cam_ptr->getParameters().fx(),cam_ptr->getParameters().fy());
    fclose(f);
    save_img_count ++;
}

std::vector<TrackedDrone> SwarmDetector::pose_estimation(const ros::Time & stamp, std::vector<TrackedDrone> tracked_up, 
        const std::vector<const cv::Mat *> & images_up, const std::vector<const cv::Mat *> & images_down, 
        cv::Mat & _show_up, cv::Mat & _show_down) {
    std::vector<TrackedDrone> tracked_drones;
    std::vector<Swarm::Pose> extrinsics;
    std::set<int> stereo_drones;
    std::vector<cv::Mat> debug_imgs;
    if (debug_show || pub_image) {
        for (auto mat_ptr : images_down) {
            debug_imgs.emplace_back(cv::Mat());
            mat_ptr->copyTo(debug_imgs.back());
        }
        visual_detection_matcher_down->draw_debug(debug_imgs);
    }

    for (auto drone_up: tracked_up) {
        //First init a visual tracker and track to down image.
        auto dir = drone_up.direction;
        drone_up.detect_no = (++target_count);
        
        Swarm::Pose drone_pose, drone_pose_down;
        Swarm::Pose cam_pose_up(Rcams[dir], Pcam);
        Swarm::Pose cam_pose_down(Rcams_down[dir], Pcam_down);
        TicToc tic_du;
        std::vector<Vector2d> pts_unit_up, pts_unit_down;
        std::vector<int> inliers_up, inliers_down;
        std::vector<float> confs_up, confs_down;
        cv::Mat crop_up, crop_down;
        cv::Rect2d rect_up, rect_down;

        Swarm::Pose est_drone_pose;

        auto succ = detect_drone_landmarks_pose(*images_up[dir], drone_up, cam_pose_up, drone_pose, 
                pts_unit_up, confs_up, inliers_up, rect_up, crop_up, false);
        double t_du = tic_du.toc();
        if (succ) {
            if (enable_triangulation && dir != 0) {
                if (succ) {
                    auto ret = visual_detection_matcher_down->reproject_drone_to_vcam(dir, drone_pose, cam_pose_down);
                    auto drone_down = drone_up;
                    drone_down.bbox = ret.second;
                    TicToc tic_dd;
                    auto succ_down = detect_drone_landmarks_pose(*images_down[dir], drone_down, cam_pose_down, drone_pose_down, 
                            pts_unit_down, confs_down, inliers_down, rect_down, crop_down, true);
                    double t_dd = tic_dd.toc();
                    if (succ_down) {
                        Swarm::StereoBundleAdjustment stereo_ba(drone_landmarks, pts_unit_up, pts_unit_down, inliers_up, inliers_down, 
                            confs_up, confs_down, cam_pose_up, cam_pose_down);
                        TicToc tic_ba;
                        auto drone_pose_stereo = stereo_ba.solve(drone_pose, true);
                        ROS_INFO("[SWARM_DETECT] TargetCount %d LMup %.2fms TargetCount %d LMdown %.2fms StereoBA %.2fms bbox_w %f succ %d %d RP %s", drone_up.detect_no, t_du, t_dd, tic_ba.toc(), 
                            succ, succ_down, drone_pose_stereo.first.tostr().c_str());
                        drone_up.relative_pose = drone_pose_stereo.first; //In body frame
                        drone_up.covariance = drone_pose_stereo.second; //In body frame
                        est_drone_pose = drone_pose_stereo.first;
                        tracked_drones.emplace_back(drone_up);
                        cam_pose_down = stereo_ba.cam_pose_2_est;
                        
                    } else {
                        ROS_INFO("[SWARM_DETECT] LMup %.2fms LMdown %.2fms bbox_w %f succ %d %d", t_du, t_dd, drone_up.bbox.width, succ, succ_down);
                    }

                    if (debug_show || pub_image) {
                        cv::rectangle(debug_imgs[dir], ret.second, cv::Scalar(0, 0, 255), 2);
                    }

                } else {
                    ROS_INFO("[SWARM_DETECT] LMup %.2fms %d", t_du, succ);
                }
            } else {
                TicToc tic_ba;
                Swarm::StereoBundleAdjustment stereo_ba(drone_landmarks, pts_unit_up, inliers_up, confs_up, cam_pose_up);

                auto drone_pose_stereo = stereo_ba.solve(drone_pose, false);
                ROS_INFO("[SWARM_DETECT] TargetCount %d LMup %.2fms MonoBA %.2fms bbox_w %f succ %d RP %s", drone_up.detect_no, t_du, tic_ba.toc(), drone_up.bbox.width, succ, 
                    drone_pose_stereo.first.tostr().c_str());
                drone_up.relative_pose = drone_pose_stereo.first; //In body frame
                // drone_up.relative_pose = drone_pose; //In body frame PnP
                drone_up.covariance = drone_pose_stereo.second; //In body frame
                drone_up.covariance(0, 0) += extrinsic_ang_cov*drone_pose_stereo.first.pos().norm();
                drone_up.covariance(1, 1) += extrinsic_ang_cov*drone_pose_stereo.first.pos().norm();
                drone_up.covariance(2, 2) += extrinsic_ang_cov*drone_pose_stereo.first.pos().norm();
                tracked_drones.emplace_back(drone_up);  
                est_drone_pose = drone_pose_stereo.first;
            }
        }

        if (debug_show || pub_image) {

            for (auto pt: drone_landmarks) {
                Vector2d pt2d;
                auto pt3d = cam_pose_up.apply_inv_pose_to(est_drone_pose*pt);
                fisheye->cam_side->spaceToPlane(pt3d, pt2d);
                cv::circle(crop_up, cv::Point2f(pt2d.x() - rect_up.x, pt2d.y() - rect_up.y)*DISP_RESCALE, 5, cv::Scalar(255, 255, 0), 2);
                if (enable_triangulation) {
                    pt3d = cam_pose_down.apply_inv_pose_to(est_drone_pose*pt);
                    fisheye->cam_side->spaceToPlane(pt3d, pt2d);
                    cv::circle(crop_down, cv::Point2f(pt2d.x() - rect_down.x, pt2d.y() - rect_down.y)*DISP_RESCALE, 5, cv::Scalar(255, 255, 0), 2);
                }
            }
            
            char title[128] = {0};

            if (enable_triangulation) {
                double rate = ((double)crop_up.cols)/((double)crop_down.cols);
                cv::resize(crop_down, crop_down, cv::Size(crop_up.cols, crop_down.rows*rate));
                cv::vconcat(crop_up, crop_down, crop_up);

                if (succ) {
                    sprintf(title, "%d: StereoBA %s Tgt No %d", drone_up._id, est_drone_pose.tostr().c_str(), drone_up.detect_no);
                    cv::putText(crop_up, title, cv::Point2f(20, 50), CV_FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
                }
            } else {
                if (succ) {
                    sprintf(title, "%d: MonoBA %s Tgt No %d", drone_up._id,  est_drone_pose.tostr().c_str(), drone_up.detect_no);
                    cv::putText(crop_up, title, cv::Point2f(20, 50), CV_FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
                }                
            }

            if (!crop_up.empty()) {
                sprintf(title, "Drone tracking @%d", self_id);
                cv::imshow(title, crop_up);

                if (debug_save_tracked_raw) {
                    sprintf(title, "%s/DroneTracker/DroneTracker%d-%06d.jpg", output_path.c_str(), self_id, drone_up.detect_no);
                    cv::imwrite(title, crop_up);
                }
            }
        }   
        
    }

    if (debug_show || pub_image) {
        if (!debug_imgs[0].empty()) {
            cv::resize(debug_imgs[0], _show_down, cv::Size(debug_imgs[1].rows, debug_imgs[1].rows));
        } else{
            _show_down = debug_imgs[1];
        }

        cv::hconcat(_show_down, debug_imgs[2], _show_down);
        cv::hconcat(_show_down, debug_imgs[3], _show_down);
        if (enable_rear) {
            cv::hconcat(_show_down, debug_imgs[4], _show_down);
        }
    }
    return tracked_drones;
}


std::vector<TrackedDrone> SwarmDetector::images_callback(const ros::Time & stamp, const std::vector<const cv::Mat *> &imgs, 
        std::pair<Swarm::Pose, std::map<int, Swarm::Pose>> poses_drones, cv::Mat & _show,
        bool is_down_cam) {
    last_stamp = stamp;
    TicToc t_cb;
    static double t_cb_sum = 0;
    static int t_cb_count = 0;

    int total_imgs = imgs.size();
    auto pose_drone = poses_drones.first;
    auto swarm_positions = poses_drones.second;
    std::vector<TrackedDrone> detected_targets;

    std::vector<cv::Mat> debug_imgs(5);

    bool need_detect = false;
    if ((stamp - last_detect).toSec() > detect_duration)
    {
        need_detect = true;
        last_detect = stamp;
    }

    if (use_tensorrt) {
        //Detect on 512x512
        TicToc tic;
        if (!(*imgs[VCAMERA_TOP]).empty()) {
            ROS_INFO("[SWARM_DETECT] will detect top image");
            auto ret = virtual_cam_callback(stamp, *imgs[VCAMERA_TOP], VCAMERA_TOP, pose_drone, need_detect, is_down_cam, debug_imgs[VCAMERA_TOP]);
            detected_targets.insert(detected_targets.end(), ret.begin(), ret.end());
        }

        //Left right
        auto ret = virtual_cam_callback2(stamp, *imgs[VCAMERA_LEFT], *imgs[VCAMERA_RIGHT], VCAMERA_LEFT, VCAMERA_RIGHT, pose_drone, need_detect, is_down_cam, 
            debug_imgs[VCAMERA_LEFT], debug_imgs[VCAMERA_RIGHT]);
        detected_targets.insert(detected_targets.end(), ret.begin(), ret.end());

        //Front rear
        if(enable_rear) {
            assert("[SWARM_DETECT] Rear must not be empty" && imgs.size()>=VCAMERA_REAR && !imgs[VCAMERA_REAR].empty());
            ret = virtual_cam_callback2(stamp, *imgs[VCAMERA_FRONT], *imgs[VCAMERA_REAR], VCAMERA_FRONT, VCAMERA_REAR, pose_drone, need_detect, is_down_cam, 
                debug_imgs[VCAMERA_FRONT], debug_imgs[VCAMERA_REAR]);
        } else {
            static cv::Mat rear = cv::Mat::zeros(imgs[VCAMERA_FRONT]->rows, imgs[VCAMERA_FRONT]->cols, imgs[VCAMERA_FRONT]->type());
            ret = virtual_cam_callback2(stamp, *imgs[VCAMERA_FRONT], rear, VCAMERA_FRONT, VCAMERA_REAR, pose_drone, need_detect, is_down_cam, 
                debug_imgs[VCAMERA_FRONT], debug_imgs[VCAMERA_REAR]);
        }

        detected_targets.insert(detected_targets.end(), ret.begin(), ret.end());
        if (need_detect) {
            ROS_INFO("[SWARM_DETECT](%d) Trackings cost %.1fms, targets %ld.", img_count, tic.toc(), detected_targets.size());
        } else {
            ROS_INFO("[SWARM_DETECT](%d) Detection & trackings cost %.1fms, targets %ld.", img_count, tic.toc(), detected_targets.size());
        }

    } else 
    {
        //Detect on 512x288
        for (int i = 1; i < total_imgs; i++) {
            if (!(*imgs[i]).empty()) {
                ROS_INFO("V cam %d", i);
                auto ret = virtual_cam_callback(stamp, *imgs[i%total_imgs], i, pose_drone, 
                    need_detect, is_down_cam, debug_imgs[i%total_imgs]);
                detected_targets.insert(detected_targets.end(), ret.begin(), ret.end());
            }
        }
    }

    std::vector<TrackedDrone> tracked_drones;
    for (int i = 0; i < total_imgs; i++) {
        if (!is_down_cam) {
            auto ret = drone_trackers[i]->get_tracked_drones();
            tracked_drones.insert(tracked_drones.end(), ret.begin(), ret.end());
        } else {
            auto ret = drone_trackers_down[i]->get_tracked_drones();
            tracked_drones.insert(tracked_drones.end(), ret.begin(), ret.end());
        }
    }

    TicToc tt_match;
    static double t_match_sum = 0;
    static int t_match_count = 0;
    std::vector<TrackedDrone> detected_drones;
    if (!is_down_cam) {
        visual_detection_matcher_up->set_swarm_state(pose_drone, swarm_positions);
        if (detected_targets.size() > 0) {
            detected_drones = visual_detection_matcher_up->match_targets(detected_targets, tracked_drones);
        }
    } else {
        visual_detection_matcher_down->set_swarm_state(pose_drone, swarm_positions);
        if (detected_targets.size() > 0) {
            detected_drones = visual_detection_matcher_down->match_targets(detected_targets, tracked_drones);
        }
    }

    t_match_sum += t_cb.toc();
    t_match_count += 1;
    ROS_INFO("[SWARM_DETECT] Full match_targets avg %.1fms cur %.1fms", t_match_sum/t_match_count, tt_match.toc());

    //Now we start detector on trackers
    if(pub_track_result || enable_tracker) {
        for (auto & target: detected_drones) {
            int dir = target.direction;
            if (target._id >= MAX_DRONE_ID && !pub_anonymous) {
                continue; //Skip anonyumous
            }
            if (!is_down_cam) {
                drone_trackers[dir]->start_tracker_tracking(target, *imgs[dir]);
            } else {
                drone_trackers_down[dir]->start_tracker_tracking(target, *imgs[dir]);
            }
        }
    }

    std::set<int> dets;
    std::vector<TrackedDrone> ret;
    for (auto &det: detected_drones) {
        auto cam = fisheye->cam_side;
        
        if (det.direction == 0) {
            cam = fisheye->cam_top;
        }
        auto R = Rcams[det.direction];
        if (is_down_cam) {
            R = Rcams_down[det.direction];
        }
        det.setCameraIntrinsicExtrinsic(R, cam);

        ret.emplace_back(det);
        dets.insert(det._id);
    }

    for (auto &trk: tracked_drones) {
        if (pub_track_result) {
            auto cam = fisheye->cam_side;
            if (trk.direction == 0) {
                cam = fisheye->cam_top;
            }
            auto R = Rcams[trk.direction];
            if (is_down_cam) {
                R = Rcams_down[trk.direction];
            }
            trk.setCameraIntrinsicExtrinsic(R, cam);

            if (dets.find(trk._id) == dets.end() ) {
                ret.emplace_back(trk);
            }
        }
    }

    if (debug_show || pub_image)
    {
        if (!is_down_cam) {
            visual_detection_matcher_up->draw_debug(debug_imgs);
        } else {
            visual_detection_matcher_down->draw_debug(debug_imgs);
        }

        if (imgs.size() > 5) {
            _show = *imgs[5];
            _show = _show(cv::Rect((_show.cols - _show.rows)/2, 0, _show.rows, _show.rows));
            cv::resize(_show, _show, cv::Size(debug_imgs[1].rows, debug_imgs[1].rows));
        }

        cv::Mat _top;
        if (!_show.empty()) {
            cv::resize(debug_imgs[0], _top, cv::Size(debug_imgs[1].rows, debug_imgs[1].rows));
            cv::hconcat(_show, _top, _show);
            cv::hconcat(_show, debug_imgs[1], _show);
        } else {
            if (!debug_imgs[0].empty()) {
                cv::resize(debug_imgs[0], _show, cv::Size(debug_imgs[1].rows, debug_imgs[1].rows));
            } else{
                _show = debug_imgs[1];
            }
        }

        cv::hconcat(_show, debug_imgs[2], _show);
        cv::hconcat(_show, debug_imgs[3], _show);
        if (enable_rear) {
            cv::hconcat(_show, debug_imgs[4], _show);
        }
    }

    t_cb_sum += t_cb.toc();
    t_cb_count ++;
    ROS_INFO("[SWARM_DETECT] Full sum %.1fms avg %.1fms", t_cb_sum/t_cb_count, t_cb.toc());
    return ret;
}

PLUGINLIB_EXPORT_CLASS(swarm_detector_pkg::SwarmDetector, nodelet::Nodelet);
} // namespace swarm_detector_pkg
