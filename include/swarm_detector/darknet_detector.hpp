#include "ros/ros.h"
#include "opencv2/opencv.hpp"
#ifdef USE_DARKNET
#include "darknet.h"
#endif

#include "base_detector.hpp"

class DarknetDetector : public BaseDetector
{
public:
    DarknetDetector(std::string weights,
                    std::string cfg,
                    double thres, double overlap_thres);

    //First is rect
    //Second is probaility
    virtual std::vector<std::pair<cv::Rect2d, double>> detect(const cv::Mat &image) override;

protected:
#ifdef USE_DARKNET
    network *net;
#endif

};