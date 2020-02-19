#include "swarm_detector/darknet_detector.hpp"

std::vector<std::pair<cv::Rect2d, double>> DarknetDetector::detect(cv::Mat & image) {
    std::vector<std::pair<cv::Rect2d, double>> ret;

    ret.push_back(std::make_pair(cv::Rect2d(rand()%400, rand()%200, 60, 40), (rand()%100)/100.0));
    return ret;
}