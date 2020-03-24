#include "ros/ros.h"
#include "opencv2/opencv.hpp"
#include "darknet.h"
#include "base_detector.hpp"

class DarknetDetector : public BaseDetector
{
public:
    DarknetDetector(std::string weights,
                    std::string cfg,
                    double thres, double overlap_thres);

    //First is rect
    //Second is probaility
    virtual std::vector<std::pair<cv::Rect2d, double>> detect(cv::Mat &image) override;
    virtual std::vector<std::pair<cv::Rect2d, double>> detect(cv::Mat &image1, cv::Mat image2) override {
        
    }

protected:
    network *net;
};