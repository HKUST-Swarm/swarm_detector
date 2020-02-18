#include "opencv2/opencv.hpp"



class DarknetDetector {
public:
    DarknetDetector(std::string weights, std::string cfg, double thres, double overlap_thres) {
        printf("Loading darknet weights from %s cfg from %s\n", weights.c_str(), cfg.c_str());
        printf("Yolo Thres %f Overlap %f\n", thres, overlap_thres);
    }
    
    //First is rect
    //Second is probaility
    std::vector<std::pair<cv::Rect2d, double>> detect(cv::Mat & image);
};