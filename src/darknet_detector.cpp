#include "swarm_detector/darknet_detector.hpp"

#ifdef USE_DARKNET
#include "darknet.h"
float m_table[256];
void init_table()
{
	for (int i = 0; i < 256; i++)
	{
		m_table[i] = i * 1.0 / 255.0f;
	}
	return;
}



image mat_to_image(const cv::Mat &mat) {
    int w = mat.cols;
    int h = mat.rows;
	image im = make_image(w, h, mat.channels());
	for(int c = 0; c < mat.channels(); ++c){
		for(int y = 0; y < mat.rows; ++y){
			for(int x = 0; x < mat.cols; ++x){
				im.data[c*im.h*im.w + y*im.w + x] = m_table[mat.data[y*mat.step + x*mat.channels() + c]];
			}
		}
	}
 
	return im;
}

DarknetDetector::DarknetDetector(std::string weights,
                                 std::string cfg,
                                 double thres, double overlap_thres): BaseDetector(thres, overlap_thres)
{
    init_table();
    printf("Loading darknet weights from %s cfg from %s\n", weights.c_str(), cfg.c_str());
    printf("Yolo Thres %f Overlap %f\n", thres, overlap_thres);
    this->net = load_network((char *)cfg.c_str(), (char *)weights.c_str(), 0);
    set_batch_network(this->net, 0);
}


std::vector<std::pair<cv::Rect2d, double>> DarknetDetector::detect(const cv::Mat &cvImg)
{
    image img = mat_to_image(cvImg);

    network_predict_image(this->net, img);

    int num_boxes = 0;
    detection *dets = get_network_boxes(net, img.w, img.h, this->thres, 0.5, nullptr, 1, &num_boxes);
    if (overlap_thres > 0) {
        do_nms_sort(dets, num_boxes, 1, overlap_thres);
    }

    std::vector<std::pair<cv::Rect2d, double>> ret;
    for (unsigned i = 0; i < num_boxes; i++)
    {
        if (*(dets[i].prob) > this->thres) {
            ROS_WARN("%f %f", dets[i].bbox.x, dets[i].bbox.y);
            ret.push_back(std::make_pair(cv::Rect2d((dets[i].bbox.x - dets[i].bbox.w/2) * img.w,
                                                    (dets[i].bbox.y - dets[i].bbox.h/2) * img.h,
                                                    dets[i].bbox.w * img.w,
                                                    dets[i].bbox.h * img.h),
                                        *(dets[i].prob)));
        }
    }

    free_detections(dets, num_boxes);
    free_image(img);
    return ret;
};
#else
std::vector<std::pair<cv::Rect2d, double>> DarknetDetector::detect(const cv::Mat &cvImg) {}
DarknetDetector::DarknetDetector(std::string weights,
                                 std::string cfg,
                                 double thres, double overlap_thres): BaseDetector(thres, overlap_thres) {}
#endif
