#include "opencv2/opencv.hpp"



class DarknetDetector {
public:
    DarknetDetector(std::string weights, std::string cfg) {

    }

    std::vector<cv::Rect2d> detect(cv::cuda::GpuMat image);
};