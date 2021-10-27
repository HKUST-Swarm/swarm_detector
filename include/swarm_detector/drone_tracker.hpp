#pragma once
#include "opencv2/opencv.hpp"
#include <vector>
#include <opencv2/tracking.hpp>
#include <map>
#include <eigen3/Eigen/Eigen>
#include <camodocal/camera_models/PinholeCamera.h>
#include <algorithm>

static Eigen::Vector3d R2ypr(const Eigen::Matrix3d &R, int degress = true);

#define INV_DEP_COEFF 0.5
#define MAX_DRONE_ID 100

struct TrackedDrone {
    int _id;
    bool is_stereo = false;

    cv::Rect2d bbox;
    Eigen::Vector3d unit_p_cam;
    double probaility = 1.0;
    double inv_dep = 0;
    Eigen::Vector2d center;
    Eigen::Matrix3d Rdrone;
    Eigen::Matrix3d ric;
    Eigen::Vector3d tic;
    double z_calib = 0;
    camodocal::PinholeCameraPtr cam;
    int direction;
    TrackedDrone() {}

    TrackedDrone(int id, cv::Rect2d _rect, double _inv_dep, double _p, double _z_calib, int _direction):
        _id(id), bbox(_rect), inv_dep(_inv_dep), probaility(_p), center(_rect.x + _rect.width/2.0, _rect.y + _rect.height/2.0), 
        z_calib(_z_calib), direction(_direction)
    {
    }

    //This is self camera position and quat
    void setCameraIntrinsicExtrinsic(
        Eigen::Vector3d _tic, Eigen::Matrix3d _ric, 
        Eigen::Matrix3d _Rdrone,
        camodocal::PinholeCameraPtr _cam) {
        Eigen::Vector3d p3d;
        cam = _cam;
        Rdrone = _Rdrone;
        ric = _ric;
        tic = _tic;

        //Ignore Extrinsic XY since the camera is at middle of the drone.
        tic.x() = 0;
        tic.y() = 0;

        _cam->liftProjective(center, p3d);
        unit_p_cam = p3d.normalized() + Eigen::Vector3d(0, -z_calib, 0);
        unit_p_cam.normalize();
    }

    //Note this function's the direction and inv_dep return in drone fram
    //ret = ric*unit_p_cam
    std::pair<Eigen::Vector3d, double> get_detection_drone_frame() {
        Eigen::Vector3d _unit_p_cam = ric * unit_p_cam; 
        return std::make_pair(_unit_p_cam, inv_dep);
    }

    //Return a virtual distance
    Eigen::Vector2d distance_to_drone(Eigen::Vector3d _pos, 
        Eigen::Vector3d tic, Eigen::Matrix3d ric, 
        Eigen::Matrix3d Rdrone, 
        Eigen::Vector3d Tdrone, 
        double focal_length = 256, 
        double scale = 0.6) {
        // auto ypr_drone =  R2ypr(Rdrone, false);
        // Rdrone = Eigen::AngleAxisd(-ypr_drone(0), Eigen::Vector3d(0, 0, 1)) * Rdrone;
        _pos = _pos - Tdrone;
        std::cout <<"\nFused Pbody "<< (Rdrone.transpose() * _pos).transpose() << std::endl;
        std::cout <<"Detec Pbody "<< (ric * unit_p_cam / inv_dep + tic).transpose() << std::endl;
        // auto up = ric*Eigen::Vector3d(0,0,1);
        // std::cout <<"Cam up_body " <<  up.transpose() << std::endl;
        // std::cout << "ric" << ric <<std::endl;
        Eigen::Vector3d d_cam = ric.transpose() * (Rdrone.transpose() * _pos - tic);
        double _inv_dep = 1/d_cam.norm();
        d_cam.normalize();

        double est_bbx_width = _inv_dep*focal_length*scale;
	    printf("Fused Pos [%3.2f, %3.2f,%3.2f]: inv_dep detection: %f fused %f\n", _pos.x(), _pos.y(), _pos.z(), _inv_dep, inv_dep);
        std::cout << "Pcam detection: " << unit_p_cam.transpose() << " Pcam fused:" << d_cam.transpose() << std::endl;
        return Eigen::Vector2d(d_cam.adjoint()*unit_p_cam, inv_dep - _inv_dep);
    }

    double bbox_cx() const { 
        return bbox.x + bbox.width/2;
    }
    
    double bbox_cy() const { 
        return bbox.x + bbox.width/2;
    }

    
    double overlap(const TrackedDrone &drone2) const {
        return overlap(drone2.bbox);
    }

    double overlap(const cv::Rect2d bbox2) const {
        double XA1 = bbox.x;
        double XA2 = bbox.x + bbox.width;
        double XB1 = bbox2.x;
        double XB2 = bbox2.x + bbox2.width;

        double YA1 = bbox.y;
        double YA2 = bbox.y + bbox.height;
        double YB1 = bbox2.y;
        double YB2 = bbox2.y + bbox2.height;

        double SA = bbox.area();
        double SB = bbox2.area();

        double SI = std::max(0.0, std::min(XA2, XB2) - std::max(XA1, XB1)) * std::max(0.0, std::min(YA2, YB2) - std::max(YA1, YB1));
        double SU = SA + SB - SI;
        return SI / std::max(SA, SB);
    }

    //Return a virtual distance
    Eigen::Vector2d distance_to_drone(TrackedDrone tracked_drone) {
        return Eigen::Vector2d(tracked_drone.unit_p_cam.adjoint()*unit_p_cam, inv_dep - tracked_drone.inv_dep);
    }

    double distance_bbox_to_drone(TrackedDrone tracked_drone) {
        // return ...;
        return 0;
    }
};



class DroneTracker {

    std::map<int, cv::Ptr<cv::Tracker>> trackers;
    camodocal::PinholeCameraPtr cam; 

    bool enable_tracker = false;
    std::map<int, TrackedDrone> tracking_drones;

    std::map<int, Eigen::Vector3d> swarm_drones;

    int last_create_id = rand()%200;
    double p_track;

    bool track_matched_only = false;

    double min_p;
    int direction;

    double drone_scale;
    double focal_length;

    // int match_with_trackers(TrackedDrone &tdrone) {
    //     //Match with trackers
    //     //Here we should use bbox to match the result.
    //     int best_id_tracker = -1;

    //     printf("[SWARM_DETECT](DroneTracker) Process detected drone: depth %3.2f prob %3.2f width %3.0f \n", 
    //         1/tdrone.inv_dep, tdrone.probaility, tdrone.width);

    //     for (auto & it: tracking_drones) {
    //         auto dis2d = tdrone.distance_to_drone(it.second);
    //         double angle = acos(dis2d.x());
            
    //         //Maybe we should compare only bounding box for tracker drone
    //         double w_det = tdrone.bbox.width;
    //         double pixel_error = fabs(dis2d.y())/w_det;
    //         double angle_error = fabs(angle/(drone_scale*it.second.inv_dep));

    //         ROS_INFO("[SWARM_DETECT](DroneTracker) Match tracker %d dis [%f, %f] err [%f/%f, %f/%f]", it.first, 
    //             dis2d.x(), dis2d.y(),
    //             angle*180.0/M_PI, accept_direction_thres,
    //             pixel_error, accept_inv_depth_thres);

    //         if (angle < accept_direction_thres && pixel_error < accept_inv_depth_thres) {
    //             if (angle + INV_DEP_COEFF*pixel_error < best_cost) {
    //                 if (matched_on_estimate_drone && it.first < MAX_DRONE_ID)
    //                 best_cost = angle + INV_DEP_COEFF*pixel_error;
    //                 best_id_tracker = it.first;
    //                 ROS_INFO("[SWARM_DETECT](DroneTracker) Matched on tracker drone %d...", best_id);
    //             }
    //         }
    //     }

    //     return best_id_tracker;
    // }



public:

    DroneTracker(int _direction, double _p_track, double _min_p, double _drone_scale, double _focal_length):
            p_track(_p_track), min_p(_min_p), direction(_direction),drone_scale(_drone_scale), focal_length(_focal_length)
    {
    }   

    std::vector<TrackedDrone> track(const cv::Mat & _img) {
        std::vector<TrackedDrone> ret;
        std::vector<int> failed_id;
        for (auto & it : trackers) {
            cv::Rect2d rect;
            int _id = it.first;
            bool success = it.second->update(_img, rect);
            if (success) {
                assert(tracking_drones.find(_id)!=tracking_drones.end() && "Tracker not found in tracked drones!");
                auto old_tracked = tracking_drones[_id];
                TrackedDrone TDrone(_id, rect, ((double)rect.width)/(drone_scale*focal_length), old_tracked.probaility*p_track, 
                    old_tracked.z_calib, direction);

                if (TDrone.probaility > min_p) {
                    ret.push_back(TDrone);
                    tracking_drones[_id] = TDrone;
                } else {
                    failed_id.push_back(_id);
                }
            } else {
                failed_id.push_back(_id);
            }
        }

        for (auto _id : failed_id) {
            ROS_INFO("Remove tracker of drone %d", _id);
            trackers.erase(_id);
            tracking_drones.erase(_id);
        }

        return ret;
    }

    // std::vector<TrackedDrone> process_detected_targets(const cv::Mat & img, std::vector<TrackedDrone> detected_drones) {
    //     std::vector<TrackedDrone> ret;
    //     this->track(img);
    //     for (auto drone: detected_drones) {
    //         TrackedDrone tracked_drone = match_with_trackers(drone);
    //         ret.push_back(tracked_drone);
    //     }
    //     return ret;
    // }

    void start_tracker_tracking(const TrackedDrone & detected_drones, const cv::Mat & frame) {
        if (trackers.find(detected_drones._id) != trackers.end()) {
            trackers.erase(detected_drones._id);
        }
        auto tracker = cv ::TrackerMOSSE::create();
        // auto tracker = cv::TrackerMedianFlow::create();
        tracker->init(frame, detected_drones.bbox);
        trackers[detected_drones._id] = tracker;
        tracking_drones[detected_drones._id] = detected_drones;
    }

    std::vector<TrackedDrone> get_tracked_drones() const {
        std::vector<TrackedDrone> ret;
        for (auto & it : tracking_drones) {
            ret.emplace_back(it.second);
        }
        return ret;
    }

};

static Eigen::Vector3d R2ypr(const Eigen::Matrix3d &R, int degress)
{
    Eigen::Vector3d n = R.col(0);
    Eigen::Vector3d o = R.col(1);
    Eigen::Vector3d a = R.col(2);

    Eigen::Vector3d ypr(3);
    double y = atan2(n(1), n(0));
    double p = atan2(-n(2), n(0) * cos(y) + n(1) * sin(y));
    double r = atan2(a(0) * sin(y) - a(1) * cos(y), -o(0) * sin(y) + o(1) * cos(y));
    ypr(0) = y;
    ypr(1) = p;
    ypr(2) = r;

    if (degress) {
        return ypr / M_PI * 180.0;
    } else {
        return ypr;
    }

}
