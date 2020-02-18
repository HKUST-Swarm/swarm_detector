#include "swarm_detector/darknet_detector.hpp"

std::vector<cv::Rect2d> DarknetDetector::detect(cv::Mat & image) {
    std::vector<cv::Rect2d> ret;

    ret.push_back(cv::Rect2d(100, 100, 50, 30));
    ret.push_back(cv::Rect2d(200, 200, 60, 20));
    return ret;
}