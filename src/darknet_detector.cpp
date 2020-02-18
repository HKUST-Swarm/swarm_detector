#include "swarm_detector/darknet_detector.hpp"

std::vector<std::pair<cv::Rect2d, double>> DarknetDetector::detect(cv::Mat & image) {
    std::vector<std::pair<cv::Rect2d, double>> ret;

    ret.push_back(std::make_pair(cv::Rect2d(100, 100, 50, 30), 1.0));
    ret.push_back(std::make_pair(cv::Rect2d(200, 200, 60, 20), 1.0));
    return ret;
}