#include "swarm_detector/dronepose_network.hpp"
#include "ATen/Parallel.h"
#include "swarm_msgs/swarm_types.hpp"

namespace Swarm {
DronePoseNetwork::DronePoseNetwork(std::string engine_path, 
    int _width, int _height, int _zoom,
    bool _enable_perf):
    TensorRTInferenceGeneric("image", _width, _height), 
    output_zoom(_zoom), enable_perf(_enable_perf) {
    output_width = width/output_zoom;
    output_height = height/output_zoom;

    std::cout << "Width: " << output_width << " Height: " << output_height << " Zoom: " << output_zoom << std::endl;
    at::set_num_threads(1);
    Swarm::TensorInfo outputTensorHeatmap;
    outputTensorHeatmap.blobName = "heatmap";
    outputTensorHeatmap.volume = feature_num*output_width*output_height;
    m_InputSize = height*width*3;
    m_OutputTensors.push_back(outputTensorHeatmap);
    std::cout << "Trying to init TRT engine of DronePoseNetwork" << engine_path << std::endl;
    init(engine_path);
}

std::pair<std::vector<cv::Point2f>, std::vector<float>> DronePoseNetwork::inference(const cv::Mat & input) {
    TicToc tic;
    cv::Mat _input;
    std::vector<cv::Point2f> pts;
    std::vector<float> confs;
    if (input.rows != height || input.cols != width) {
        cv::resize(input, _input, cv::Size(width, height));
        _input.convertTo(_input, CV_32FC3, 1.0/ 127.5, -1);
    } else {
        input.convertTo(_input, CV_32FC3, 1.0/ 127.5, -1);
    }
    doInference(_input);
    if (enable_perf) {
        std::cout << "DronePoseNetwork Inference Time " << tic.toc();
    }

    TicToc tic1;
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    auto heatmap = at::from_blob(m_OutputTensors[0].hostBuffer, {feature_num, height/output_zoom, width/output_zoom}, options);
    for (int i = 0; i < feature_num; i++) {
        // std::cout << "Heatmap " << i << std::endl;
        // std::cout << heatmap[i] << std::endl;
        float conf = at::max(heatmap[i]).data<float>()[0];
        auto coor = at::where(heatmap[i] == conf);
        // std::cout << "coor: " << coor << std::endl << "conf" << conf << std::endl;
        // float conf = heatmap[i][coor[0]][coor[1]].data<float>()[0];
        cv::Point2f pt;
        float y = coor[0].data<long>()[0]/((float)output_width);
        float x = coor[1].data<long>()[0]/((float)output_height);
        pt.x = x * ((float)input.cols);
        pt.y = y * ((float)input.rows);
        // std::cout << "coor " << coor << " conf " << conf << " x " << x << " y " << y << " pt " << pt << std::endl;
        confs.emplace_back(conf);
        pts.emplace_back(pt);
    }
    return std::make_pair(pts, confs);
}
}