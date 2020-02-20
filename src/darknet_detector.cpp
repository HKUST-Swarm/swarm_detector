#include "swarm_detector/darknet_detector.hpp"
#include "darknet.h"


image ipl_to_image(IplImage *src)
{
    int h = src->height;
    int w = src->width;
    int c = src->nChannels;
    image im = make_image(w, h, c);
    unsigned char *data = (unsigned char *)src->imageData;
    int step = src->widthStep;
    int i, j, k;

    for (i = 0; i < h; ++i)
    {
        for (k = 0; k < c; ++k)
        {
            for (j = 0; j < w; ++j)
            {
                im.data[k * w * h + i * w + j] = data[i * step + j * c + k] / 255.;
            }
        }
    }
    return im;
}

DarknetDetector::DarknetDetector(std::string weights,
                                 std::string cfg,
                                 double thres, double overlap_thres)
{
    printf("Loading darknet weights from %s cfg from %s\n", weights.c_str(), cfg.c_str());
    printf("Yolo Thres %f Overlap %f\n", thres, overlap_thres);
    this->net = load_network((char *)cfg.c_str(), (char *)weights.c_str(), 0);
    set_batch_network(this->net, 0);
    this->thres = thres;
    this->overlap_thres = overlap_thres;
}

std::vector<std::pair<cv::Rect2d, double>> DarknetDetector::detect(cv::Mat &cvImg)
{
    cv::Mat temp;
    if (cvImg.channels() == 1)
        cv::cvtColor(cvImg, temp, cv::COLOR_GRAY2BGR);
    else
        temp = cvImg;
    IplImage iplImg = cvIplImage(temp);
    image img = ipl_to_image(&iplImg);
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
