#include "swarm_detector/swarm_detector.hpp"
#include "swarm_detector/fisheye_undist.hpp"
#include "swarm_detector/darknet_detector.hpp"
#include "swarm_detector/tensorrt_detector.hpp"
#include "swarm_detector/drone_tracker.hpp"
#include <opencv2/core/eigen.hpp>
#include <nav_msgs/Odometry.h>
#include <swarm_msgs/swarm_detected.h>
#include <swarm_msgs/node_detected_xyzyaw.h>
#include <swarm_msgs/swarm_fused.h>
#include <swarm_msgs/Pose.h>
#include <chrono>
#include <vins/FlattenImages.h>
#include <swarm_msgs/swarm_lcm_converter.hpp>

#define VCAMERA_TOP 0
#define VCAMERA_LEFT 1
#define VCAMERA_FRONT 2
#define VCAMERA_RIGHT 3
#define VCAMERA_REAR 4

#define BBOX_DEPTH_OFFSET 0.12
#define DOWN_Z_OFFSET -0.08
#define UP_Z_OFFSET 0.06
// #define ASSUME_IDENTITY_CAMERA_ATT

#define FIXED_CAM_UP_Z 0.12
#define FIXED_CAM_DOWN_Z 0.0

#define DEBUG_SHOW_HCONCAT
class TicToc
{
  public:
    TicToc()
    {
        tic();
    }

    void tic()
    {
        start = std::chrono::system_clock::now();
    }

    double toc()
    {
        end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        return elapsed_seconds.count() * 1000;
    }

  private:
    std::chrono::time_point<std::chrono::system_clock> start, end;
};


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
namespace swarm_detector_pkg
{

void SwarmDetector::onInit()
{
    ros::NodeHandle nh = this->getMTPrivateNodeHandle();

    cv::setNumThreads(1);
    std::string darknet_weights_path;
    std::string darknet_cfg;
    std::string camera_config_file;
    std::string extrinsic_path;
    bool track_matched_only = false;
    bool tensorrt_fp16 = false;
    double fov = 235;
    double thres, overlap_thres;
    double p_track;
    double min_p;
    double acpt_direction_thres;
    double acpt_inv_dep_thres;

    nh.param<bool>("show", debug_show, false);
    nh.param<bool>("track_matched_only", track_matched_only, false);
    nh.param<bool>("pub_image", pub_image, true);
    nh.param<bool>("use_tensorrt", use_tensorrt, true);
    nh.param<bool>("pub_track_result", pub_track_result, true);
    nh.param<bool>("tensorrt_fp16", tensorrt_fp16, true);
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
    nh.param<double>("gamma", gamma_, 1.6);
    nh.param<double>("min_p", min_p, -1);
    nh.param<int>("drone_id", self_id, -1);
    nh.param<bool>("pub_anonymous", pub_anonymous, false);
    nh.param<bool>("enable_tracker", enable_tracker, false);
    nh.param<bool>("enable_triangulation", enable_triangulation, true);
    nh.param<bool>("enable_gamma_correction", enable_gamma_correction, true);
    nh.param<bool>("enable_up_cam", enable_up_cam, false);
    nh.param<bool>("enable_down_cam", enable_down_cam, true);

    //Is in degree
    nh.param<double>("acpt_direction_thres", acpt_direction_thres, 10);
    //Is in pixels
    nh.param<double>("acpt_inv_dep_thres", acpt_inv_dep_thres, 10);
    cv::Mat R, T;

    FILE *fh = fopen(extrinsic_path.c_str(), "r");
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

    fisheye = new FisheyeUndist(camera_config_file, fov, true, width);
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

    visual_detection_matcher_up = new VisualDetectionMatcher(Pcam, Rcams, fisheye, acpt_direction_thres, acpt_inv_dep_thres, debug_show);
    visual_detection_matcher_down = new VisualDetectionMatcher(Pcam, Rcams_down, fisheye, acpt_direction_thres, acpt_inv_dep_thres, debug_show);

    uchar* p = lookUpTable.ptr();
    for( int i = 0; i < 256; ++i)
        p[i] = cv::saturate_cast<uchar>(pow(i / 255.0, gamma_) * 255.0);

    fisheye_img_sub = nh.subscribe("image_raw", 3, &SwarmDetector::image_callback, this);
    fisheye_img_comp_sub = nh.subscribe("image_comp", 3, &SwarmDetector::image_comp_callback, this);
    vins_imgs_sub = nh.subscribe("vins_flattened", 3, &SwarmDetector::flattened_image_callback, this);
    swarm_fused_sub = nh.subscribe("swarm_fused", 3, &SwarmDetector::swarm_fused_callback, this);
    swarm_detected_pub = nh.advertise<swarm_msgs::swarm_detected>("/swarm_detection/swarm_detected_raw", 3);
    odom_sub = nh.subscribe("odometry", 3, &SwarmDetector::odometry_callback, this);

    image_show_pub = nh.advertise<sensor_msgs::Image>("show", 1);
    
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
    for (size_t i = 0; i < sf.ids.size(); i ++) {
        if (sf.ids[i] != self_id) {
            swarm_positions[sf.ids[i]] = Swarm::Pose(sf.local_drone_position[i], sf.local_drone_rotation[i]);
        }
    }
}

cv::Scalar ScalarHSV2BGR(uchar H, uchar S, uchar V) {
    cv::Mat rgb;
    cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(H, S, V));
    cv::cvtColor(hsv, rgb, cv::COLOR_HSV2BGR);
    return cv::Scalar(rgb.data[0], rgb.data[1], rgb.data[2]);
}

std::vector<TrackedDrone> SwarmDetector::virtual_cam_callback(const cv::Mat & _img, int direction, Swarm::Pose pose_drone,
        bool need_detect,
        bool is_down_cam,
        cv::Mat & debug_img) { 
    std::vector<std::pair<cv::Rect2d, double>> detected_targets;
    if (need_detect) {
        TicToc t_d;

        cv::Rect roi(0, 0, _img.cols, _img.rows);
        detected_targets = detector->detect(_img);
        ROS_INFO("[SWARM_DETECT] Detect squared cost %fms \n", t_d.toc());
    }

    return this->process_detect_result(_img, direction, detected_targets, pose_drone, need_detect, is_down_cam, debug_img);
}

std::vector<TrackedDrone> SwarmDetector::virtual_cam_callback2(const cv::Mat & img1, 
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

    if (need_detect || enable_tracker) {
        auto track1 = this->process_detect_result(img1, dir1, det1, pose_drone, need_detect, 
            is_down_cam, debug_img1);
        auto track2 = this->process_detect_result(img2, dir2, det2, pose_drone, need_detect,
            is_down_cam, debug_img2);

        track1.insert(track1.end(), track2.begin(), track2.end());
        return track1;
    }
    return std::vector<TrackedDrone>();
}


std::vector<TrackedDrone> SwarmDetector::process_detect_result(const cv::Mat & _img, 
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
    if (is_down_cam) {
        z_calib = DOWN_Z_OFFSET;
    }

    for (auto det : detected_targets) {
        auto _target = TrackedDrone(-1, det.first, ((double)det.first.width)/(drone_scale*focal_length), 
            det.second, z_calib, direction);
        detected_targets_drones.emplace_back(_target);
    }

    //Track only
    if(need_detect && pub_track_result || enable_tracker) {
        if (!is_down_cam) {
            drone_trackers[direction]->track(img);
        } else {
            drone_trackers_down[direction]->track(img);
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
            sprintf(idtext, "(%3.1f\%)", ret.second*100);
	        // cv::putText(debug_img, idtext, pos, CV_FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 0, 0), 2);
            // cv::rectangle(debug_img, ret.first, cv::Scalar(255, 10, 0), 2);
            cv::rectangle(debug_img, ret.first, cv::Scalar(0, 255, 0), 2);
            // sprintf(idtext, "(%3.1f,%3.1f)", ret.first.x + ret.first.width/2, ret.first.y + ret.first.height/2);
	        // cv::putText(debug_img, idtext, pos, CV_FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 0, 0), 2);
            cv::Point2f p(ret.first.x + ret.first.width/2, ret.first.y + ret.first.height/2);
            cv::circle(debug_img, p, 4, cv::Scalar(0, 0, 255), 2);
        }

        // for (auto ret : tracked_drones)
        // {
        //     // ROS_INFO("Tracked drone ID %d@%d", ret._id, direction);
        //     // std::cout << ret.bbox << std::endl;
        //     cv::rectangle(debug_img, ret.bbox, ScalarHSV2BGR(ret.probaility * 128 + 128, 255, 255), 3);
        //     sprintf(idtext, "[%d](%3.1f%%)", ret._id, ret.probaility * 100);
        //     //Draw bottom
        //     cv::Point2f pos(ret.bbox.x, ret.bbox.y + ret.bbox.height + 20);
        //     cv::putText(debug_img, idtext, pos, CV_FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);
        // }
    }
    return detected_targets_drones;
}

void SwarmDetector::odometry_callback(const nav_msgs::Odometry & odom) {
    auto tup = std::make_pair(odom.header.stamp, Swarm::Pose(odom.pose.pose));
    pose_buf.push(tup);
}

void SwarmDetector::publish_tracked_drones(ros::Time stamp, Swarm::Pose local_pose_self, std::vector<TrackedDrone> drones, std::vector<Swarm::Pose> extrinsics) {
    swarm_detected sd;
    sd.header.stamp = stamp;
    sd.self_drone_id = self_id;
    sd.is_6d_detect = false;
    auto &detected_nodes = sd.detected_nodes_xyz_yaw;
    for (int i = 0; i <drones.size(); i++) {
        auto tdrone = drones[i];
        auto extrinsic = extrinsics[i];
        if (!pub_anonymous && tdrone._id >= MAX_DRONE_ID) {
            continue;
        }
        node_detected_xyzyaw nd;
        std::pair<Eigen::Vector3d, double> det;
        det = tdrone.get_detection_drone_frame();
        Eigen::Vector3d p_drone = det.first;
        Eigen::Vector3d p_drone_yaw_only = Eigen::AngleAxisd(-local_pose_self.yaw(), Eigen::Vector3d::UnitZ())*local_pose_self.att() * p_drone;
        p_drone_yaw_only.normalize();
        nd.dpos.x = p_drone_yaw_only.x();
        nd.dpos.y = p_drone_yaw_only.y();
        nd.dpos.z = p_drone_yaw_only.z();

        Swarm::Pose pose_cam = local_pose_self*extrinsic;
        nd.camera_extrinsic = extrinsic.to_ros_pose();
        Swarm::Pose pose_cam_yaw_only = pose_cam;
        pose_cam_yaw_only.set_yaw_only();
        nd.local_pose_self = pose_cam_yaw_only.to_ros_pose();


        nd.enable_scale = true;
        nd.is_yaw_valid = false;
        nd.self_drone_id = self_id;
        nd.remote_drone_id = tdrone._id;
        nd.header.stamp = stamp;
        nd.probaility = tdrone.probaility;
        nd.inv_dep = det.second;

        ROS_INFO("[SWARM_DETECT] Pub drone %d dir: [%3.2f, %3.2f, %3.2f] dep %3.2f Cam Pose", tdrone._id, 
            nd.dpos.x, nd.dpos.y, nd.dpos.z,
            1/nd.inv_dep
        );

        pose_cam_yaw_only.print();

        detected_nodes.push_back(nd);
    }

    swarm_detected_pub.publish(sd);
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


//Pose drone should be convert by base_coor.
Swarm::Pose SwarmDetector::get_pose_drone(const ros::Time & stamp) {
    double min_dt = 10000;

    Swarm::Pose pose_drone;
    while(pose_buf.size() > 0) {
        double dt = (pose_buf.front().first - stamp).toSec();
        // ROS_INFO("DT %f", dt);
        if (dt < 0) {
            //Pose in buffer is older
            if (fabs(dt) < min_dt) {
                pose_drone = pose_buf.front().second;
                min_dt = fabs(dt);
            }
        }

        if (dt > 0)
        {
            //pose in buffer is newer
            if (fabs(dt) < min_dt) {
                pose_drone = pose_buf.front().second;
                min_dt = fabs(dt);
            }
            break;
        }

        pose_buf.pop();
    }


    if (min_dt > 0.01)
    {
        ROS_WARN("[SWARM_DETECT] Pose %3.1f dt is too big!", min_dt * 1000);
    }

    return pose_drone;
}


void SwarmDetector::image_callback(const sensor_msgs::Image::ConstPtr &msg) {
    update_swarm_pose();

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
    images_callback(msg->header.stamp, img_cpus_ptrs);
}

void SwarmDetector::image_comp_callback(const sensor_msgs::CompressedImageConstPtr &img_msg) {
    update_swarm_pose();

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
    images_callback(img_msg->header.stamp, img_cpus_ptrs);
}

cv::Mat img_empty;
void SwarmDetector::flattened_image_callback(const vins::FlattenImagesConstPtr &flattened) {
    // if (fabs(sf_latest - flattened->header.stamp.toSec()) > 0.1) 
    {
        ROS_WARN("[SWARM_DETECT] sf_latest.t - flattened.t high %5.2fms", (sf_latest - flattened->header.stamp.toSec())*1000);
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

    std::vector<TrackedDrone> tracked_drones_up, tracked_drones_down;
    if (enable_up_cam) {
        tracked_drones_up = images_callback(flattened->header.stamp, img_cpus);
    }
    if (enable_down_cam) {
        tracked_drones_down = images_callback(flattened->header.stamp, img_cpus_down, true);
    }

    auto pose_drone = get_pose_drone(flattened->header.stamp);
    
    if (enable_up_cam && enable_down_cam && enable_triangulation) {
        auto tracked_drones = stereo_triangulate(tracked_drones_up, tracked_drones_down);
        publish_tracked_drones(flattened->header.stamp, pose_drone, tracked_drones.first, tracked_drones.second);
    } else {
        std::vector<TrackedDrone> tracked_drones;
        std::vector<Swarm::Pose> extrinsics;
        for (auto tracked: tracked_drones_up) {
            tracked.unit_p_cam.normalize();
            tracked_drones.push_back(tracked);
            extrinsics.push_back(Swarm::Pose(Rcam, Pcam));
        }

        for (auto tracked: tracked_drones_down) {
            tracked.unit_p_cam.z() += DOWN_Z_OFFSET;
            tracked.unit_p_cam.normalize();
            tracked_drones.push_back(tracked);
            extrinsics.push_back(Swarm::Pose(Rcam_down, Pcam_down));
        }
        
        publish_tracked_drones(flattened->header.stamp, pose_drone, tracked_drones, extrinsics);
    }
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


std::pair<std::vector<TrackedDrone>,std::vector<Swarm::Pose>> SwarmDetector::stereo_triangulate(std::vector<TrackedDrone> tracked_up, std::vector<TrackedDrone> tracked_down) {
    std::vector<TrackedDrone> tracked_drones;
    std::vector<Swarm::Pose> extrinsics;
    std::set<int> stereo_drones;
    for (auto drone_up: tracked_up) {
        for (auto drone_down: tracked_down) {
            if (drone_up._id == drone_down._id) {
                //Perform triangulation here
                //To simplify, we use identity pose since the DroneTracker gives direction already included the ric.
                //Result is at drone frame.
                Eigen::Matrix<double, 3, 4> upPose, downPose;
                Eigen::Vector3d _Pcam(0, 0, FIXED_CAM_UP_Z), _Pcam_down(0, 0, FIXED_CAM_DOWN_Z);
                upPose.leftCols<3>() = Eigen::Matrix3d::Identity();
                upPose.rightCols<1>() = -Eigen::Matrix3d::Identity() * _Pcam;

                downPose.leftCols<3>() = Eigen::Matrix3d::Identity();
                downPose.rightCols<1>() = -Eigen::Matrix3d::Identity() * _Pcam_down;

                Eigen::Vector3d pos_drone;
                auto det_up = drone_up.get_detection_drone_frame();
                auto det_down = drone_down.get_detection_drone_frame();
                std::cout << "det_up" << det_up.first.transpose() << "det_down" << det_down.first.transpose() << std::endl;
                std::cout << "Pcam" << Pcam.transpose() << "Pcam_down" << Pcam_down.transpose() << std::endl;
                double err = triangulatePoint3DPts(upPose, downPose, 
                    det_up.first, det_down.first, pos_drone);
                if (err < triangulation_thres) {
                    drone_up.is_stereo = true;
                    drone_up.unit_p_cam = pos_drone;
                    drone_up.unit_p_cam.normalize();
                    drone_up.inv_dep = 1/(pos_drone.norm()+BBOX_DEPTH_OFFSET);
                    drone_up.ric = Eigen::Matrix3d::Identity();

                    ROS_INFO("[SWARM_DETECT] Stereo drone %d, pos_drone: %3.2f %3.2f %3.2f tri_err %3.2f", drone_up._id, pos_drone.x(), pos_drone.y(), pos_drone.z(), err*1000);
                    tracked_drones.push_back(drone_up);
                    extrinsics.push_back(Swarm::Pose());
                    stereo_drones.insert(drone_up._id);
                    break;
                } else {
                    ROS_WARN("[SWARM_DETECT] Stereo drone %d, pos_drone: %3.2f %3.2f %3.2f tri_err %3.2f", drone_up._id, pos_drone.x(), pos_drone.y(), pos_drone.z(), err*1000);
                    // exit(-1);
                }
            }
        }
    }

    //Only stereo detection is used
    return std::make_pair(tracked_drones, extrinsics);
}


std::vector<TrackedDrone> SwarmDetector::images_callback(const ros::Time & stamp, const std::vector<const cv::Mat *> &imgs, bool is_down_cam) {
    // bool require_detect = false;
    // if ((stamp - last_stamp).toSec() < detect_duration) {
    //     if (enable_tracker) {
    //         require_detect = true;
    //     } else {
    //         return std::vector<TrackedDrone>(); 
    //     }
    // }

    last_stamp = stamp;

    int total_imgs = imgs.size();
    auto pose_drone = get_pose_drone(stamp);
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
            auto ret = virtual_cam_callback(*imgs[VCAMERA_TOP], VCAMERA_TOP, pose_drone, need_detect, is_down_cam, debug_imgs[VCAMERA_TOP]);
            detected_targets.insert(detected_targets.end(), ret.begin(), ret.end());
        }

        //Left right
        auto ret = virtual_cam_callback2(*imgs[VCAMERA_LEFT], *imgs[VCAMERA_RIGHT], VCAMERA_LEFT, VCAMERA_RIGHT, pose_drone, need_detect, is_down_cam, 
            debug_imgs[VCAMERA_LEFT], debug_imgs[VCAMERA_RIGHT]);
        detected_targets.insert(detected_targets.end(), ret.begin(), ret.end());

        //Front rear
        if(enable_rear) {
            assert("[SWARM_DETECT] Rear must not be empty" && imgs.size()>=VCAMERA_REAR && !imgs[VCAMERA_REAR].empty());
            ret = virtual_cam_callback2(*imgs[VCAMERA_FRONT], *imgs[VCAMERA_REAR], VCAMERA_FRONT, VCAMERA_REAR, pose_drone, need_detect, is_down_cam, 
                debug_imgs[VCAMERA_FRONT], debug_imgs[VCAMERA_REAR]);
        } else {
            static cv::Mat rear = cv::Mat::zeros(imgs[VCAMERA_FRONT]->rows, imgs[VCAMERA_FRONT]->cols, imgs[VCAMERA_FRONT]->type());
            ret = virtual_cam_callback2(*imgs[VCAMERA_FRONT], rear, VCAMERA_FRONT, VCAMERA_REAR, pose_drone, need_detect, is_down_cam, 
                debug_imgs[VCAMERA_FRONT], debug_imgs[VCAMERA_REAR]);
        }

        detected_targets.insert(detected_targets.end(), ret.begin(), ret.end());
        if (need_detect) {
            ROS_INFO("[SWARM_DETECT] Whole trackings cost %fms. Total targets %ld is", tic.toc(), detected_targets.size());
        } else {
            ROS_INFO("[SWARM_DETECT] Whole detection & trackings cost %fms. Total Targets %ld", tic.toc(), detected_targets.size());
        }

    } else 
    {
        //Detect on 512x288
        for (int i = 0; i < total_imgs; i++) {
            // ROS_INFO("V cam %d", i);
            auto ret = virtual_cam_callback(*imgs[i%total_imgs], i, pose_drone, 
                need_detect, is_down_cam, debug_imgs[i%total_imgs]);
            detected_targets.insert(detected_targets.end(), ret.begin(), ret.end());
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

    if (!is_down_cam) {
        visual_detection_matcher_up->set_swarm_state(pose_drone, swarm_positions);
        tracked_drones = visual_detection_matcher_up->match_targets(detected_targets, tracked_drones);
    } else {
        visual_detection_matcher_down->set_swarm_state(pose_drone, swarm_positions);
        tracked_drones = visual_detection_matcher_down->match_targets(detected_targets, tracked_drones);
    }

    if (debug_show || pub_image)
    {
        if (!is_down_cam) {
            visual_detection_matcher_up->draw_debug(debug_imgs);
        } else {
            visual_detection_matcher_down->draw_debug(debug_imgs);
        }

        cv::Mat _show;
#ifdef DEBUG_SHOW_HCONCAT
        cv::Mat _show_l2;
        cv::Mat _show_l3;
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
        if (!debug_imgs[4].empty()) {
            cv::hconcat(_show, debug_imgs[4], _show);
        }

        // cv::imwrite("/home/xuhao/detector.png", _show);
#else
        cv::Mat _show_l2;
        cv::Mat _show_l3;
        _show = debug_imgs[0];
        if (_show.empty()) {
            _show = cv::Mat(cv::Size(debug_imgs[1].rows, debug_imgs[1].rows), CV_8UC3, cv::Scalar(0 , 0, 0));
        }

        if (debug_imgs[4].empty()) {
            debug_imgs[4] = cv::Mat(cv::Size(debug_imgs[1].rows, debug_imgs[1].cols), CV_8UC3, cv::Scalar(0 , 0, 0));
        }

        cv::resize(_show, _show, cv::Size(debug_imgs[1].rows, debug_imgs[1].rows));
        cv::hconcat(_show, debug_imgs[1], _show);
        cv::hconcat(_show, debug_imgs[2], _show);

        cv::hconcat(debug_imgs[3], debug_imgs[4], _show_l2);

        cv::resize(_show_l2, _show_l2, cv::Size(0, 0), 
            ((double) _show.cols)/ _show_l2.cols,  ((double) _show.cols)/ _show_l2.cols) ;

        cv::vconcat(_show, _show_l2, _show);

        cv::line(_show, cv::Point(_show.cols/3, 0), cv::Point(_show.cols/3, _show.rows), cv::Scalar(255, 255, 255));
        cv::line(_show, cv::Point(_show.cols*2/3, 0), cv::Point(_show.cols*2/3, _show.rows), cv::Scalar(255, 255, 255));

#endif
        double f_resize = ((double)show_width) / (double)_show.cols;
        cv::resize(_show, _show, cv::Size(), f_resize, f_resize);

	    if (debug_show) {
            char title[100] = {0};
            if (is_down_cam) {
                sprintf(title, "DroneTrackerDown@%d", self_id);
            } else {
                sprintf(title, "DroneTrackerUP@%d", self_id);
            }
       	    cv::imshow(title, _show);
            cv::waitKey(3);
	    } else {
	        cv_bridge::CvImage cvimg;
	        cvimg.encoding = sensor_msgs::image_encodings::BGR8;
	        cvimg.image = _show;
            image_show_pub.publish(cvimg);
	    }
    }

    return tracked_drones;
}

PLUGINLIB_EXPORT_CLASS(swarm_detector_pkg::SwarmDetector, nodelet::Nodelet);
} // namespace swarm_detector_pkg
