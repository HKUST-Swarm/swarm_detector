#pragma once

#include "opencv2/opencv.hpp"
#include "class_detector.h"
#include <utility>

class BaseDetector
{
public:
    BaseDetector(double _thres, double _overlap_thres):
        thres(_thres), overlap_thres(_overlap_thres) {}
    //First is rect
    //Second is probaility
    virtual std::vector<std::pair<cv::Rect2d, double>> detect(cv::Mat &image) = 0;
    virtual std::vector<std::pair<cv::Rect2d, double>> detect(cv::Mat &image1, cv::Mat image2) = 0;

protected:
    double thres;
    double overlap_thres;
};