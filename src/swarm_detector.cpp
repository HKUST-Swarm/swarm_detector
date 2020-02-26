#include "swarm_detector/swarm_detector.hpp"
#include "swarm_detector/fisheye_undist.hpp"
#include "swarm_detector/darknet_detector.hpp"
#include "swarm_detector/drone_tracker.hpp"
#include <opencv2/core/eigen.hpp>
#include <nav_msgs/Odometry.h>
#include <swarm_msgs/swarm_detected.h>
#include <swarm_msgs/node_detected_xyzyaw.h>
#include <swarm_msgs/swarm_fused.h>
#include <swarm_msgs/Pose.h>
#include <chrono>

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
    ros::NodeHandle nh = this->getPrivateNodeHandle();


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
    double acpt_direction_thres;
    double acpt_inv_dep_thres;

    nh.param<bool>("enable_rear", enable_rear, false);
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

    //Is in degree
    nh.param<double>("acpt_direction_thres", acpt_direction_thres, 10);
    //Is in pixels
    nh.param<double>("acpt_inv_dep_thres", acpt_inv_dep_thres, 10);

    cv::Mat R, T;

    FILE *fh = fopen(extrinsic_path.c_str(), "r");
    if (fh == NULL)
    {
        ROS_WARN("config_file dosen't exist; Assume identity camera pose");
    }
    else
    {
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
    for (int i = 0; i < 6; i++)
    {
        last_detects.push_back(ros::Time(0));
        ROS_INFO("Init tracker on %d with P %f %f %f R", i, Pcam.x(), Pcam.y(), Pcam.z());
        std::cout << Rcam * Rvcams[i] << std::endl;
        camera_model::PinholeCameraPtr cam = fisheye->cam_side;
        if (i % 5 == 0)
        {
            cam = fisheye->cam_top;
        }
        drone_trackers.push_back(
            new DroneTracker(Pcam, Rcam * Rvcams[i], cam, drone_scale, p_track, min_p,
                             acpt_direction_thres, acpt_inv_dep_thres, track_matched_only));
    }



    fisheye_img_sub = nh.subscribe("image_raw", 3, &SwarmDetector::image_callback, this);
    singleview_img_sub = nh.subscribe("image_front", 3, &SwarmDetector::front_image_callback, this);
    swarm_fused_sub = nh.subscribe("swarm_fused", 3, &SwarmDetector::swarm_fused_callback, this);
    swarm_detected_pub = nh.advertise<swarm_msgs::swarm_detected>("swarm_fused_relative", 3);
    odom_sub = nh.subscribe("odometry", 3, &SwarmDetector::odometry_callback, this);
    imu_sub = nh.subscribe("imu", 3, &SwarmDetector::imu_callback, this);
    
    ROS_INFO("Finish initialize swarm detector, wait for data\n");
}




void SwarmDetector::swarm_fused_callback(const swarm_msgs::swarm_fused_relative & sf) {
    Eigen::AngleAxisd rotate_by_yaw(sf.self_yaw, Eigen::Vector3d::UnitZ());
    for (unsigned int i = 0; i < sf.ids.size(); i ++ ) {
        swarm_positions[i] = rotate_by_yaw * Eigen::Vector3d(
                sf.relative_drone_position[i].x,
                sf.relative_drone_position[i].y,
                sf.relative_drone_position[i].z
        );

    }
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

std::vector<TrackedDrone> SwarmDetector::virtual_cam_callback(cv::Mat & _img, int direction, Swarm::Pose pose_drone, cv::Mat & debug_img) {
    std::vector<TrackedDrone> tracked_drones;

    bool need_detect = false;
    // img_cuda.download(img);
    double alpha = 1.5;
    // cv::Mat img;
    // double beta = 30;
    // _img.convertTo(img, -1, alpha, beta);
    cv::Mat & img = _img;
    std::vector<std::pair<cv::Rect2d, double>> detected_drones;
    if ((ros::Time::now() - last_detects[direction]).toSec() > detect_duration)
    {
        need_detect = true;
        last_detects[direction] = ros::Time::now();
    }

    drone_trackers[direction]->update_cam_pose(pose_drone.pos(), pose_drone.att().toRotationMatrix());
    drone_trackers[direction]->update_swarm_pose(swarm_positions);
    if (need_detect) {
        TicToc t_d;

        //Detect and update to tracker
        cv::Rect roi(0, 0, img.cols, img.rows);
        if (direction == 0 || direction == 5)
        {
            //If top, detect half plane and track whole
            double offset = -1;
            if (direction == 0)
            {
                roi = cv::Rect(0, 0, img.cols, yolo_height);
            }
            else if (direction == 5)
            {
                roi = cv::Rect(0, img.rows - yolo_height, img.cols, yolo_height);
                offset = img.rows - yolo_height;
            }

            // ROS_INFO("1");
            cv::Mat img_roi = img(roi);
            detected_drones = detector->detect(img_roi);
            NODELET_DEBUG("2");

            if (offset > 0)
            {
                for (auto &rect : detected_drones)
                {
                    rect.first.y = rect.first.y + offset;
                }
            }
        }
        else
        {
            detected_drones = detector->detect(img);
        }

        ROS_INFO("Detect cost %fms", t_d.toc());

        tracked_drones = drone_trackers[direction]->process_detect(img, detected_drones);
    }
    else
    {
        //Track only
        tracked_drones = drone_trackers[direction]->track(img);
    }

    if (debug_show)
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
	        cv::putText(debug_img, idtext, pos, CV_FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 0, 0), 2);
            cv::rectangle(debug_img, ret.first, cv::Scalar(255, 10, 0), 1);
        }

        for (auto ret : tracked_drones)
        {
            // ROS_INFO("Tracked drone ID %d@%d", ret._id, direction);
            // std::cout << ret.bbox << std::endl;
            cv::rectangle(debug_img, ret.bbox, ScalarHSV2BGR(ret.probaility * 128 + 128, 255, 255), 3);
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
    Eigen::Quaterniond quat(imu_data.orientation.w, imu_data.orientation.x, imu_data.orientation.y, imu_data.orientation.z);
    auto tup = std::make_pair(imu_data.header.stamp, Swarm::Pose(quat, Eigen::Vector3d::Zero()));
    pose_buf.push(tup);
}



void SwarmDetector::publish_tracked_drones(ros::Time stamp, std::vector<TrackedDrone> drones) {
    swarm_detected sd;
    sd.header.stamp = stamp;
    sd.self_drone_id = -1;
    sd.is_6d_detect = false;
    auto &detected_nodes = sd.detected_nodes_xyz_yaw;
    for (TrackedDrone &tdrone : drones)
    {
        node_detected_xyzyaw nd;
        nd.dpos.x = tdrone.unit_p_body_yaw_only.x();
        nd.dpos.y = tdrone.unit_p_body_yaw_only.y();
        nd.dpos.z = tdrone.unit_p_body_yaw_only.z();

        nd.enable_scale = true;
        nd.is_yaw_valid = false;
        nd.self_drone_id = -1;
        nd.remote_drone_id = tdrone._id;
        nd.header.stamp = stamp;
        nd.probaility = tdrone.probaility;
        nd.inv_dep = tdrone.inv_dep;

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


void SwarmDetector::front_image_callback(const sensor_msgs::Image::ConstPtr &msg) {
    cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, "rgb8");

    int id = 2;
    update_swarm_pose();
    double min_dt = 10000;
    Swarm::Pose pose_drone;
    while(pose_buf.size() > 0) {
        double dt = (pose_buf.front().first - msg->header.stamp).toSec();
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

    cv::Mat _show;
    auto ret = virtual_cam_callback(cv_ptr->image, 2, pose_drone, _show);
    // track_drones.insert(track_drones.end(), ret.begin(), ret.end());
    // publish_tracked_drones(msg->header.stamp, track_drones);
    if (debug_show)
    {
        double f_resize = ((double)show_width) / (double)_show.cols;
        cv::cvtColor(_show, _show, cv::COLOR_RGB2BGR);
        cv::resize(_show, _show, cv::Size(), f_resize, f_resize);
        cv::imshow("DroneTracker", _show);
        cv::waitKey(3);
    }
}



void SwarmDetector::image_callback(const sensor_msgs::Image::ConstPtr &msg) {
    cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, "rgb8");

    int id = 2;
    update_swarm_pose();
    // cv::cuda::GpuMat img_cuda = fisheye->undist_id_cuda(cv_ptr->image, id);
    auto imgs = fisheye->undist_all_cuda(cv_ptr->image, true, enable_rear);
    int total_imgs = imgs.size();

    double min_dt = 10000;
    Swarm::Pose pose_drone;
    while(pose_buf.size() > 0) {
        double dt = (pose_buf.front().first - msg->header.stamp).toSec();
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

    std::vector<cv::Mat> debug_imgs;
    debug_imgs.resize(total_imgs);
    std::vector<cv::Mat> img_cpus;
    img_cpus.resize(total_imgs);
    for (unsigned int i = 0; i < total_imgs; i++) {
        imgs[i].download(img_cpus[i]);
    }

    std::vector<TrackedDrone> track_drones;
    for (int i = 0; i < total_imgs + 1; i++) {
        // ROS_INFO("V cam %d", i);
        auto ret = virtual_cam_callback(img_cpus[i%total_imgs], i, pose_drone, debug_imgs[i%total_imgs]);
        track_drones.insert(track_drones.end(), ret.begin(), ret.end());
    }

    publish_tracked_drones(msg->header.stamp, track_drones);
    if (debug_show)
    {
        cv::Mat _show;
        cv::Mat _show_l2;
        cv::Mat _show_l3;
        cv::resize(cv_ptr->image(cv::Rect(190, 62, 900, 900)), _show, cv::Size(width, width));
        cv::Mat tmp;
        cv::resize(debug_imgs[0], tmp, cv::Size(width, width));
        cv::hconcat(_show, tmp, _show);

        cv::hconcat(debug_imgs[1], debug_imgs[2], _show_l2);
        cv::hconcat(debug_imgs[3], debug_imgs[4], _show_l3);

        cv::vconcat(_show, _show_l2, _show);
        cv::vconcat(_show, _show_l3, _show);

        cv::line(_show, cv::Point(_show.cols/2, 0), cv::Point(_show.cols/2, _show.rows), cv::Scalar(255, 255, 255));

        double f_resize = ((double)show_width) / (double)_show.cols;
        cv::cvtColor(_show, _show, cv::COLOR_RGB2BGR);
        cv::resize(_show, _show, cv::Size(), f_resize, f_resize);

        cv::imshow("DroneTracker", _show);
        cv::waitKey(3);
    }
}

PLUGINLIB_EXPORT_CLASS(swarm_detector_pkg::SwarmDetector, nodelet::Nodelet);
} // namespace swarm_detector_pkg
