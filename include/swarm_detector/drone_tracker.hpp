#include "opencv2/opencv.hpp"
#include <vector>
#include <opencv2/tracking.hpp>
#include <map>
#include <eigen3/Eigen/Eigen>
#include <camera_model/camera_models/PinholeCamera.h>

static Eigen::Vector3d R2ypr(const Eigen::Matrix3d &R, int degress = true);

struct TrackedDrone {
    int _id;

    cv::Rect2d bbox;
    Eigen::Vector3d unit_p_body;
    Eigen::Vector3d unit_p_body_yaw_only;
    double probaility = 1.0;
    double scale = 0;
    Eigen::Vector2d center;
    TrackedDrone() {}

    TrackedDrone(int id, cv::Rect2d _rect, double _scale, double _p):
        _id(id), bbox(_rect), scale(_scale), probaility(_p), center(_rect.x + _rect.width/2.0, _rect.y + _rect.height/2.0)
    {
    }


    //This is self camera position and quat
    void update_position(
        Eigen::Vector3d tic, Eigen::Matrix3d ric, 
        Eigen::Vector3d Pdrone, Eigen::Matrix3d Rdrone,
        camera_model::CameraPtr cam) {
        auto ypr = R2ypr(Rdrone, false);
        double yaw = ypr.x();
        Eigen::Vector3d p3d;
        cam->liftProjective(center, p3d);
        unit_p_body = ric * p3d;
        unit_p_body.normalize();

        //No scale so assume camera is on CG
        unit_p_body_yaw_only = Rdrone*unit_p_body;
        unit_p_body_yaw_only = Eigen::AngleAxisd(-yaw, Eigen::Vector3d::UnitZ()) * unit_p_body_yaw_only;
        unit_p_body_yaw_only.normalize();
    }
};



class DroneTracker {

    std::map<int, cv::Ptr<cv::Tracker>> trackers;
    camera_model::CameraPtr cam; 

    std::map<int, TrackedDrone> tracking_drones;

    int last_create_id = rand()%1000*100;
    double p_track;
    int match_id(cv::Rect2d rect) {
        return -1;
    }

    bool update_bbox(cv::Rect2d rect, double p, cv::Mat & frame,  TrackedDrone & drone) {
        int _id = match_id(rect);

        if(_id < 0) {
            if(track_matched_only) {
                return false;
            } else {
            //Gives a random id
               last_create_id ++;
                _id = last_create_id;
            }
        }

        start_tracker_tracking(_id, frame, rect);

        ROS_INFO("New detected drone: %d", _id);

        //Simple use width as scale
        drone = TrackedDrone(_id, rect, rect.width, p);
        drone.update_position(tic, ric, Pdrone, Rdrone, cam);

        tracking_drones[_id] = drone;
        return true;
    }

    bool track_matched_only = false;

    Eigen::Vector3d Pdrone = Eigen::Vector3d::Zero();
    Eigen::Matrix3d Rdrone = Eigen::Matrix3d::Identity();
    Eigen::Vector3d tic;
    Eigen::Matrix3d ric;
    double drone_scale;
    double min_p;

public:

    void update_cam_pose(Eigen::Vector3d _Pdrone, Eigen::Matrix3d _Rdrone) {
        Pdrone = _Pdrone;
        Rdrone = _Rdrone;
    }

    DroneTracker(Eigen::Vector3d _tic, Eigen::Matrix3d _ric, camera_model::CameraPtr _cam, 
        double _drone_scale, double _p_track, double _min_p, bool _track_matched_only):
        tic(_tic), ric(_ric), cam(_cam), drone_scale(_drone_scale), p_track(_p_track), min_p(_min_p), track_matched_only(_track_matched_only)
    {
        
    }   

    std::vector<TrackedDrone> track(cv::Mat & _img) {
        std::vector<TrackedDrone> ret;
        std::vector<int> failed_id;
        for (auto & it : trackers) {
            cv::Rect2d rect;
            int _id = it.first;
            bool success = it.second->update(_img, rect);
            if (success) {
                assert(tracking_drones.find(_id)!=tracking_drones.end() && "Tracker not found in tracked drones!");
                auto old_tracked = tracking_drones[_id];
                TrackedDrone TDrone(_id, rect, rect.width, old_tracked.probaility*p_track);

                if (TDrone.probaility > min_p) {
                    TDrone.update_position(tic, ric, Pdrone, Rdrone, cam);
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



    std::vector<TrackedDrone> process_detect(cv::Mat & img, std::vector<std::pair<cv::Rect2d, double>> detected_drones) {
        std::vector<TrackedDrone> ret;

        std::vector<TrackedDrone> new_tracked = track(img);
        for (auto rect: detected_drones) {
            TrackedDrone tracked_drones;
            bool success = update_bbox(rect.first, rect.second, img, tracked_drones);
            if(success) {
                ret.push_back(tracked_drones);
            }
        }
        return ret;
    }

    void start_tracker_tracking(int _id, cv::Mat & frame, cv::Rect2d bbox) {
        if (trackers.find(_id) != trackers.end()) {
            trackers.erase(_id);
        }
        // cv::Ptr<cv::TrackerMOSSE> tracker = cv ::TrackerMOSSE::create();
        auto tracker = cv::TrackerMedianFlow::create();
        tracker->init(frame, bbox);
        trackers[_id] = tracker;
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