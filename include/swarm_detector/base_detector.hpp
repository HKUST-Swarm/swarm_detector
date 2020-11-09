#pragma once

#include "opencv2/opencv.hpp"
#include "class_detector.h"
#include <utility>

typedef std::pair<cv::Rect2d, double> BBoxProb;
typedef std::vector<BBoxProb> BBoxProbArray;

class BaseDetector
{
    cv::Mat rectangle;
public:
    BaseDetector(double _thres, double _overlap_thres):
        thres(_thres), overlap_thres(_overlap_thres) {}
    //First is rect
    //Second is probaility
    virtual BBoxProbArray detect(const cv::Mat &image) = 0;
    virtual std::pair<BBoxProbArray, BBoxProbArray> detect(const cv::Mat &image1, const cv::Mat & image2) {
        int rows_first = image1.rows;

        if(rectangle.rows == 0) {
            rectangle = cv::Mat(image1.rows*2, image1.rows*2, CV_8UC3, cv::Scalar(0, 0, 0));
        }

        int dx = image1.rows - image1.cols/2;
        cv::Mat rectup(rectangle, cv::Rect(dx, 0, image1.cols, image1.rows));
        cv::Mat rectdown(rectangle, cv::Rect(dx, image1.rows, image1.cols, image1.rows));

        assert(!image1.empty() && "Image 1 must not be empty");
        assert(!image2.empty() && "Image 2 must not be empty");
        image1.copyTo(rectup);
        image2.copyTo(rectdown);
        // cv::imshow("Rectangle Image", rectangle);

        auto ret = this->detect(rectangle);
        BBoxProbArray arr1;
        BBoxProbArray arr2;
        for (auto bbox:ret) {
            bbox.first.x  = bbox.first.x - dx;

            if(bbox.first.y < rows_first) {
                arr1.push_back(bbox);
            } else {
                bbox.first.y = bbox.first.y - rows_first;
                arr2.push_back(bbox);
            }
        }
        return std::make_pair(arr1, arr2);
    }

protected:
    double thres;
    double overlap_thres;
};