#include "opencv2/opencv.hpp"



class DarknetDetector {
public:
    DarknetDetector(std::string weights, std::string cfg) {

    }
    
    //First is rect
    //Second is probaility
    std::vector<std::pair<cv::Rect2d, double>> detect(cv::Mat & image);
};