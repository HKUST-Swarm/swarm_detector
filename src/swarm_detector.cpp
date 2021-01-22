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

// #define DEBUG_SHOW_HCONCAT
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


    std::string darknet_weights_path;
    std::string darknet_cfg;
    std::string camera_config_file;
    std::string extrinsic_path;
    bool track_matched_only = false;
    bool tensorrt_fp16 = false;
    double fov = 235;
    double thres, overlap_thres;
    double drone_scale;
    double p_track;
    double min_p;
    double acpt_direction_thres;
    double acpt_inv_dep_thres;
    bool enable_tracker;

    nh.param<bool>("show", debug_show, false);
    nh.param<bool>("track_matched_only", track_matched_only, false);
    nh.param<bool>("pub_image", pub_image, true);
    nh.param<bool>("use_tensorrt", use_tensorrt, true);
    nh.param<bool>("enable_rear", enable_rear, false);
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
    nh.param<double>("min_p", min_p, -1);
    nh.param<int>("drone_id", self_id, -1);
    nh.param<bool>("pub_anonymous", pub_anonymous, false);
    nh.param<bool>("enable_tracker", enable_tracker, true);

    //Is in degree
    nh.param<double>("acpt_direction_thres", acpt_direction_thres, 10);
    //Is in pixels
    nh.param<double>("acpt_inv_dep_thres", acpt_inv_dep_thres, 10);
    cv::Mat R, T;

    FILE *fh = fopen(extrinsic_path.c_str(), "r");
    if (fh == NULL)
    {
        ROS_WARN("config_file dosen't exist; Assume identity camera pose");
	    Rcam = Eigen::Matrix3d::Identity();
	    Pcam = Eigen::Vector3d(0.105, 0.004614, 0.0898);
	    std::cout << "Translation" << Pcam;
    }
    else
    {
        cv::FileStorage fsSettings(extrinsic_path, cv::FileStorage::READ);
        cv::Mat _T;
        fsSettings["body_T_cam0"] >> _T;
        fsSettings["image_width"] >> width;

        Eigen::Matrix4d T;
        cv::cv2eigen(_T, T);
        Rcam = T.block<3, 3>(0, 0);
        Pcam = T.block<3, 1>(0, 3);
        extrinsic = Swarm::Pose(Rcam, Pcam);
        fsSettings.release();

        ROS_INFO("Camera width %d, Pose", width);
        std::cout << "R" << Rcam << std::endl;
        std::cout << "P" << Pcam.transpose() << std::endl;
    }

    if(use_tensorrt) {
        detector = new TensorRTDetector(darknet_weights_path, darknet_cfg, thres, overlap_thres, tensorrt_fp16);
    } else {
        detector = new DarknetDetector(darknet_weights_path, darknet_cfg, thres, overlap_thres);
    }

    fisheye = new FisheyeUndist(camera_config_file, fov, true, width);
    side_height = fisheye->sideImgHeight;


    for (int i = 0; i < 5; i++) {
        last_detects.push_back(ros::Time(0));
        ROS_INFO("Init tracker on %d with P %f %f %f R", i, Pcam.x(), Pcam.y(), Pcam.z());
        std::cout << Rcam * Rvcams[i] << std::endl;
        camodocal::PinholeCameraPtr cam = fisheye->cam_side;
        if (i % 5 == 0)
        {
            cam = fisheye->cam_top;
        }
        drone_trackers.push_back(
            new DroneTracker(Pcam, Rcam * Rvcams[i], cam, drone_scale, p_track, min_p,
                             acpt_direction_thres, acpt_inv_dep_thres, track_matched_only, enable_tracker));
    }



    fisheye_img_sub = nh.subscribe("image_raw", 3, &SwarmDetector::image_callback, this);
    fisheye_img_comp_sub = nh.subscribe("image_comp", 3, &SwarmDetector::image_comp_callback, this);
    vins_imgs_sub = nh.subscribe("vins_flattened", 3, &SwarmDetector::flattened_image_callback, this);
    swarm_fused_sub = nh.subscribe("swarm_fused_relative", 3, &SwarmDetector::swarm_fused_callback, this);
    swarm_detected_pub = nh.advertise<swarm_msgs::swarm_detected>("/swarm_detection/swarm_detected_raw", 3);
    odom_sub = nh.subscribe("odometry", 3, &SwarmDetector::odometry_callback, this);
    // imu_sub = nh.subscribe("imu", 10, &SwarmDetector::imu_callback, this);

    image_show_pub = nh.advertise<sensor_msgs::Image>("show", 1);
    
    ROS_INFO("Finish initialize swarm detector, wait for data\n");
}

void SwarmDetector::swarm_fused_callback(const swarm_msgs::swarm_fused_relative & sf) {
    Eigen::AngleAxisd rotate_by_yaw(sf.self_yaw, Eigen::Vector3d::UnitZ());
    for (unsigned int i = 0; i < sf.ids.size(); i ++ ) {
        if (self_id != sf.ids[i]) {
            swarm_positions[sf.ids[i]] = rotate_by_yaw * Eigen::Vector3d(
                    sf.relative_drone_position[i].x,
                    sf.relative_drone_position[i].y,
                    sf.relative_drone_position[i].z
            );
        }
    }

    sf_latest = sf.header.stamp.toSec();
}

void SwarmDetector::update_swarm_pose() {
    //May update by velocity later
}


cv::Scalar ScalarHSV2BGR(uchar H, uchar S, uchar V) {
    cv::Mat rgb;
    cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(H, S, V));
    cv::cvtColor(hsv, rgb, cv::COLOR_HSV2BGR);
    return cv::Scalar(rgb.data[0], rgb.data[1], rgb.data[2]);
}

std::vector<TrackedDrone> SwarmDetector::virtual_cam_callback(const cv::Mat & _img, int direction, Swarm::Pose pose_drone, cv::Mat & debug_img) { 
    bool need_detect = false;
    std::vector<std::pair<cv::Rect2d, double>> detected_drones;
    if ((ros::Time::now() - last_detects[direction]).toSec() > detect_duration)
    {
        need_detect = true;
        last_detects[direction] = ros::Time::now();
    }

    if (need_detect) {
        TicToc t_d;

        cv::Rect roi(0, 0, _img.cols, _img.rows);
        detected_drones = detector->detect(_img);
        ROS_INFO("Detect squared cost %fms \n", t_d.toc());
    }

    return this->process_detect_result(_img, direction, detected_drones, pose_drone, debug_img, need_detect);
}

std::vector<TrackedDrone> SwarmDetector::virtual_cam_callback(const cv::Mat & img1, const cv::Mat & img2, int dir1, int dir2, 
    Swarm::Pose pose_drone, cv::Mat & debug_img1, cv::Mat & debug_img2) { 
    
    bool need_detect = false;
    BBoxProbArray det1, det2;
    if ((ros::Time::now() - last_detects[dir1]).toSec() > detect_duration)
    {
        need_detect = true;
        last_detects[dir1] = ros::Time::now();
    }

    if (need_detect) {
        TicToc t_d;
        auto ret = detector->detect(img1, img2);
        det1 = ret.first;
        det2 = ret.second;
        ROS_INFO("Detect squared of 2 images cost %fms\n", t_d.toc());
    }

    auto track1 = this->process_detect_result(img1, dir1, det1, pose_drone, debug_img1, need_detect);
    auto track2 = this->process_detect_result(img2, dir2, det2, pose_drone, debug_img2, need_detect);

    track1.insert(track1.end(), track2.begin(), track2.end());
    return track1;
}


std::vector<TrackedDrone> SwarmDetector::process_detect_result(const cv::Mat & img, int direction, 
    BBoxProbArray detected_drones, Swarm::Pose pose_drone, cv::Mat & debug_img, bool has_detect) {
    std::vector<TrackedDrone> tracked_drones;

    double alpha = 1.5;

   
    drone_trackers[direction]->update_cam_pose(pose_drone.pos(), pose_drone.att().toRotationMatrix());
    drone_trackers[direction]->update_swarm_pose(swarm_positions);
    //Track only
    if(has_detect) {
        tracked_drones = drone_trackers[direction]->process_detect(img, detected_drones);
    } else {
        if (pub_track_result) {
            tracked_drones = drone_trackers[direction]->track(img);
        } else {
            drone_trackers[direction]->track(img);
        }
    }

    if (debug_show || pub_image)
    {
        if (debug_img.empty())
        {
            img.copyTo(debug_img);
        }

        char idtext[20] = {0};

        for (auto ret: detected_drones) {
            //Draw in top
            cv::Point2f pos(ret.first.x+ret.first.width - 5, ret.first.y - 15);
            sprintf(idtext, "(%3.1f\%)", ret.second*100);
	        // cv::putText(debug_img, idtext, pos, CV_FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 0, 0), 2);
            // cv::rectangle(debug_img, ret.first, cv::Scalar(255, 10, 0), 2);
            cv::rectangle(debug_img, ret.first, cv::Scalar(0, 255, 0), 2);
        }

        for (auto ret : tracked_drones)
        {
            // ROS_INFO("Tracked drone ID %d@%d", ret._id, direction);
            // std::cout << ret.bbox << std::endl;
            // cv::rectangle(debug_img, ret.bbox, ScalarHSV2BGR(ret.probaility * 128 + 128, 255, 255), 3);
            sprintf(idtext, "[%d](%3.1f%%)", ret._id, ret.probaility * 100);
            //Draw bottom
            cv::Point2f pos(ret.bbox.x, ret.bbox.y + ret.bbox.height + 20);
            cv::putText(debug_img, idtext, pos, CV_FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);
        }
    }
    NODELET_DEBUG("4");

    return tracked_drones;
}

void SwarmDetector::odometry_callback(const nav_msgs::Odometry & odom) {
    auto tup = std::make_pair(odom.header.stamp, Swarm::Pose(odom.pose.pose));
    pose_buf.push(tup);
}


void SwarmDetector::imu_callback(const sensor_msgs::Imu & imu_data) {
    // Eigen::Quaterniond quat(imu_data.orientation.w, imu_data.orientation.x, imu_data.orientation.y, imu_data.orientation.z);
    // auto tup = std::make_pair(imu_data.header.stamp, Swarm::Pose(quat, Eigen::Vector3d::Zero()));
    // pose_buf.push(tup);
}



void SwarmDetector::publish_tracked_drones(ros::Time stamp, Swarm::Pose local_pose_self, std::vector<TrackedDrone> drones) {
    swarm_detected sd;
    sd.header.stamp = stamp;
    sd.self_drone_id = self_id;
    sd.is_6d_detect = false;
    auto &detected_nodes = sd.detected_nodes_xyz_yaw;
    for (TrackedDrone &tdrone : drones) {
        if (!pub_anonymous && tdrone._id >= MAX_DRONE_ID) {
            continue;
        }
        node_detected_xyzyaw nd;
        auto det = tdrone.get_cam_pose_yaw_only(Rcam);
        Eigen::Vector3d p_cam = det.first;
        Swarm::Pose pose_cam = local_pose_self*extrinsic;
        Swarm::Pose pose_cam_yaw_only = pose_cam;
        pose_cam_yaw_only.set_yaw_only();
        Eigen::Vector3d p_cam_only = pose_cam_yaw_only.att().inverse() * pose_cam.att() * p_cam;
        nd.local_pose_self = pose_cam_yaw_only.to_ros_pose();

        nd.dpos.x = p_cam_only.x();
        nd.dpos.y = p_cam_only.y();
        nd.dpos.z = p_cam_only.z();

        nd.enable_scale = true;
        nd.is_yaw_valid = false;
        nd.self_drone_id = self_id;
        nd.remote_drone_id = tdrone._id;
        nd.header.stamp = stamp;
        nd.probaility = tdrone.probaility;
        nd.inv_dep = det.second;


        ROS_INFO("Pub drone %ld dir: [%3.2f, %3.2f, %3.2f] dep %3.2f Cam Pose", tdrone._id, 
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
        ROS_WARN("Pose %3.1f dt is too big!", min_dt * 1000);
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
    ROS_INFO("sf_latest.t - flattened.t %5.2f", (sf_latest - flattened->header.stamp.toSec())*1000);

    std::vector<const cv::Mat *> img_cpus;
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

    images_callback(flattened->header.stamp, img_cpus);
}


void SwarmDetector::images_callback(const ros::Time & stamp, const std::vector<const cv::Mat *> &imgs) {

    if ((stamp - last_stamp).toSec() < detect_duration) {
        return;
    }

    last_stamp = stamp;

    int total_imgs = imgs.size();
    auto pose_drone = get_pose_drone(stamp);
    std::vector<TrackedDrone> track_drones;

    std::vector<cv::Mat> debug_imgs(5);
    if (use_tensorrt) {
        //Detect on 512x512
        TicToc tic;
        if (!(*imgs[VCAMERA_TOP]).empty()) {
            auto ret = virtual_cam_callback(*imgs[VCAMERA_TOP], VCAMERA_TOP, pose_drone, debug_imgs[VCAMERA_TOP]);
            track_drones.insert(track_drones.end(), ret.begin(), ret.end());
        }

        //Left right
        auto ret = virtual_cam_callback(*imgs[VCAMERA_LEFT], *imgs[VCAMERA_RIGHT], VCAMERA_LEFT, VCAMERA_RIGHT, 
            pose_drone, debug_imgs[VCAMERA_LEFT], debug_imgs[VCAMERA_RIGHT]);
        track_drones.insert(track_drones.end(), ret.begin(), ret.end());

        //Front rear
        if(enable_rear) {
            assert("Rear must not be empty" && imgs.size()>=VCAMERA_REAR && !imgs[VCAMERA_REAR].empty());
            ret = virtual_cam_callback(*imgs[VCAMERA_FRONT], *imgs[VCAMERA_REAR], VCAMERA_FRONT, 
                VCAMERA_REAR, pose_drone, debug_imgs[VCAMERA_FRONT], debug_imgs[VCAMERA_REAR]);
        } else {
            static cv::Mat rear = cv::Mat::zeros(imgs[VCAMERA_FRONT]->rows, imgs[VCAMERA_FRONT]->cols, imgs[VCAMERA_FRONT]->type());
            ret = virtual_cam_callback(*imgs[VCAMERA_FRONT], rear, VCAMERA_FRONT, 
                VCAMERA_REAR, pose_drone, debug_imgs[VCAMERA_FRONT], debug_imgs[VCAMERA_REAR]);
        }

        track_drones.insert(track_drones.end(), ret.begin(), ret.end());
        ROS_INFO("Whole detection & Tracking cost %fms Total Targets %ld", tic.toc(), track_drones.size());

    } else 
    {
        //Detect on 512x288
        for (int i = 0; i < total_imgs + 1; i++) {
            // ROS_INFO("V cam %d", i);
            auto ret = virtual_cam_callback(*imgs[i%total_imgs], i, pose_drone, debug_imgs[i%total_imgs]);
            track_drones.insert(track_drones.end(), ret.begin(), ret.end());
        }
    }

    publish_tracked_drones(stamp, pose_drone, track_drones);

    if (debug_show || pub_image)
    {
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
            sprintf(title, "DroneTracker@%d", self_id);
       	    cv::imshow(title, _show);
            cv::waitKey(3);
	    } else {
	        cv_bridge::CvImage cvimg;
	        cvimg.encoding = sensor_msgs::image_encodings::BGR8;
	        cvimg.image = _show;
            image_show_pub.publish(cvimg);
	    }
    }
}

PLUGINLIB_EXPORT_CLASS(swarm_detector_pkg::SwarmDetector, nodelet::Nodelet);
} // namespace swarm_detector_pkg
