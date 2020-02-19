#include "opencv2/opencv.hpp"
#include <vector>
#include <opencv2/tracking.hpp>
#include <map>
#include <eigen3/Eigen/Eigen>
#include <camera_model/camera_models/PinholeCamera.h>

struct TrackedDrone {
    int _id;

    cv::Rect2d bbox;
    Eigen::Vector3d world_p;
    Eigen::Vector3d body_p;
    Eigen::Vector3d unit_p_body;
    double probaility = 1.0;
    TrackedDrone() {

    }

    TrackedDrone(int id): _id(id) {

    }


    //This is self camera position and quat
    void update_position(Eigen::Vector3d Pcam, Eigen::Matrix3d Qcam, camera_model::CameraPtr cam, double drone_scale) {
        //TODO:
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
        drone = TrackedDrone(_id);
        drone.bbox = rect;
        drone.probaility = p;
        drone.update_position(Pcam, Qcam, cam, drone_scale);

        tracking_drones[_id] = drone;
        return true;
    }

    bool track_matched_only = false;

    Eigen::Vector3d Pcam = Eigen::Vector3d::Zero();
    Eigen::Matrix3d Qcam = Eigen::Matrix3d::Identity();
    Eigen::Vector3d tic;
    Eigen::Matrix3d ric;
    double drone_scale;
    double min_p;

public:

    void update_cam_pose(Eigen::Vector3d Pdrone, Eigen::Matrix3d Qdrone) {
        this->Pcam = Pdrone + Qdrone * tic;
        this->Qcam = Qdrone * ric;
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
                TrackedDrone TDrone(_id);
                TDrone.bbox = rect;
                TDrone.probaility = old_tracked.probaility*p_track;

                if (TDrone.probaility > min_p) {
                    TDrone.update_position(Pcam, Qcam, cam, drone_scale);
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