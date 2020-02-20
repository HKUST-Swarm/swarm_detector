#include "ros/ros.h"
#include "opencv2/opencv.hpp"
#include "darknet.h"

class DarknetDetector
{
public:
    DarknetDetector(std::string weights,
                    std::string cfg,
                    double thres, double overlap_thres);

    //First is rect
    //Second is probaility
    std::vector<std::pair<cv::Rect2d, double>> detect(cv::Mat &image);

protected:
    network *net;
    double thres;
    double overlap_thres;
};