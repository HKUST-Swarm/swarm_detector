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
                    double _thres, double _overlap_thres, bool fp16=true):
        BaseDetector(_thres, _overlap_thres) {
        Config config;
        config.file_model_cfg = cfg;
        config.file_model_weights = weights;
        config.calibration_image_list_file_txt = "";
        if (fp16) {
            config.inference_precison = FP16;
        }
        config.detect_thresh = _thres;
        config.net_type = YOLOV4_TINY;
        detector.init(config);

    }

    virtual std::vector<std::pair<cv::Rect2d, double>> detect(const cv::Mat &image) override {
        std::vector<std::pair<cv::Rect2d, double>> ret;
        std::vector<cv::Mat> imgs;
        cv::Mat img;
        // cv::cvtColor(image, img, cv::COLOR_BGR2RGB);

        img = image;
        imgs.emplace_back(img);
	    std::vector<BatchResult> ress;
    	detector.detect(imgs, ress);

        for (auto det : ress[0]) {
            ret.push_back(std::make_pair(det.rect, det.prob));
        }

        return ret;
    }

};