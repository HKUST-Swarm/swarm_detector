#pragma once

#include "swarm_loop/tensorrt_generic.h"
#include <torch/csrc/autograd/variable.h>
#include <ATen/ATen.h>
#include <torch/csrc/api/include/torch/types.h>
#include <Eigen/Dense>

namespace Swarm {
class DronePoseNetwork: public TensorRTInferenceGeneric {

public:
    int output_width;
    int output_height;
    int feature_num = 11;
    bool enable_perf;
    int max_num = 200;
    int output_zoom = 1;
    DronePoseNetwork(std::string engine_path, int _width, int _height, int zoom, bool _enable_perf = false);

    std::pair<std::vector<cv::Point2f>, std::vector<float>> inference(const cv::Mat & input);
};
}