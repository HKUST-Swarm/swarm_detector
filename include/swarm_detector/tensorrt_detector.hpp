#include "opencv2/opencv.hpp"
#include "class_detector.h"
#include <utility>
#include "base_detector.hpp"

class TensorRTDetector: public BaseDetector
{
    Detector detector;
public:
    TensorRTDetector(std::string weights,
                    std::string cfg,
                    double _thres, double _overlap_thres):
        BaseDetector(_thres, _overlap_thres) {
        Config config;
        config.file_model_cfg = cfg;
        config.file_model_weights = weights;
        config.calibration_image_list_file_txt = "";
        config.inference_precison = FP32;
        detector.init(config);

    }

    virtual std::vector<std::pair<cv::Rect2d, double>> detect(cv::Mat &image) override {
        std::vector<std::pair<cv::Rect2d, double>> ret;
	    std::vector<Result> res;
    	detector.detect(image, res);
        for (const auto &r : res) {
		    std::cout << "id:" << r.id << " prob:" << r.prob << " rect:" << r.rect << std::endl;
            ret.push_back(std::make_pair(r.rect, r.prob));
            cv::rectangle(image, r.rect, cv::Scalar(255, 0, 0), 2);
	    }
        return ret;
    }

    //hconcat two image for detect
    virtual std::vector<std::pair<cv::Rect2d, double>> detect(cv::Mat &image1, cv::Mat image2) override {
        cv::Mat _new;
        cv::hconcat(image1, image2, _new);
        cv::imshow("new img to detect", _new);
        this->detect(_new);
    }

};