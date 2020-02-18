#include "opencv2/opencv.hpp"
#include <vector>
#include <opencv2/tracking.hpp>
#include <map>
#include <eigen3/Eigen/Eigen>

struct TrackedDrone {
    int _id;

    cv::Rect2f bbox;
    Eigen::Vector3d world_p;
    Eigen::Vector3d body_p;
    Eigen::Vector3d unit_p_body;

    TrackedDrone() {

    }

    TrackedDrone(int id): _id(id) {

    }


    //This is self camera position and quat
    void update_position(Eigen::Vector3d Pcam, Eigen::Matrix3d Qcam) {
        //TODO:
    }
};

class DroneTracker {

    std::map<int, cv::Ptr<cv::TrackerMOSSE>> trackers;

    std::vector<TrackedDrone> tracking_drones;

    int last_create_id = 100;
    int match_id(cv::Rect2d rect) {
        return 0;
    }

    bool update_bbox(cv::Rect2d rect, cv::Mat & frame,  TrackedDrone & drone) {
        int _id = match_id(rect);

        if(_id < 0) {
            if(track_matched_only) {
                return false;
            }
        } else {
            //Gives a random id
            last_create_id ++;
            _id = last_create_id;
        }

        start_tracker_tracking(_id, frame, rect);

        drone = TrackedDrone(_id);
        drone.bbox = rect;
        drone.update_position(Pcam, Qcam);

        
        return true;
    }

    bool track_matched_only = false;

    Eigen::Vector3d Pcam = Eigen::Vector3d::Zero();
    Eigen::Matrix3d Qcam = Eigen::Matrix3d::Identity();
    Eigen::Vector3d tic;
    Eigen::Matrix3d ric;
public:

    void update_cam_pose(Eigen::Vector3d Pdrone, Eigen::Matrix3d Qdrone) {
        this->Pcam = Pdrone + Qdrone * tic;
        this->Qcam = Qdrone * ric;
    }

    DroneTracker(bool _track_matched_only, Eigen::Vector3d _tic, Eigen::Matrix3d _ric):
        track_matched_only(_track_matched_only), tic(_tic), ric(_ric)
    {
        
    }   

    std::vector<TrackedDrone> track(cv::Mat & _img) {
        std::vector<TrackedDrone> ret;
        return ret;
    }



    std::vector<TrackedDrone> process_detect(cv::Mat & img, std::vector<cv::Rect2d> detected_drones) {
        std::vector<TrackedDrone> ret;
        for (auto rect: detected_drones) {
            TrackedDrone tracked_drones;
            bool success = update_bbox(rect, img, tracked_drones);
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
        cv::Ptr<cv::TrackerMOSSE> tracker = cv ::TrackerMOSSE::create();
        tracker->init(frame, bbox);
        trackers[_id] = tracker;
    }

};