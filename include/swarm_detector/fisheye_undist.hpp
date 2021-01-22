#pragma once

#include "ros/ros.h"
#include "sensor_msgs/Image.h"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/cudaimgproc.hpp>
#include <camodocal/camera_models/CameraFactory.h>
#include <camodocal/camera_models/PinholeCamera.h>
#include "cv_bridge/cv_bridge.h"
#include <opencv2/cudawarping.hpp>

#define DEG_TO_RAD (M_PI / 180.0)


namespace swarm_detector_pkg{
class FisheyeUndist {

    camodocal::CameraPtr cam;

    std::vector<cv::Mat> undistMaps;
    std::vector<cv::cuda::GpuMat> undistMapsGPUX;
    std::vector<cv::cuda::GpuMat> undistMapsGPUY;
public:
    camodocal::PinholeCameraPtr cam_top;
    camodocal::PinholeCameraPtr cam_side;
    double f_side = 0;
    double f_center = 0;
    double cx_side = 0, cy_side = 0;
    int imgWidth = 0;
    double fov = 0; //in degree
    Eigen::Vector3d cameraRotation;
    bool enable_cuda = false;
    int cam_id = 0;
    int sideImgHeight = 0;

    FisheyeUndist(const std::string & camera_config_file, double _fov, bool _enable_cuda = true, int imgWidth = 600, int _cam_id = 0):
    imgWidth(imgWidth), fov(_fov), cameraRotation(0, 0, 0), enable_cuda(_enable_cuda), cam_id(_cam_id) {
        cam = camodocal::CameraFactory::instance()
            ->generateCameraFromYamlFile(camera_config_file);

        undistMaps = generateAllUndistMap(cam, cameraRotation, imgWidth, fov);
        ROS_INFO("undismap size %ld", undistMaps.size());
        if (enable_cuda) {
            for (auto mat : undistMaps) {
                cv::Mat xy[2];
                cv::split(mat, xy);
                undistMapsGPUX.push_back(cv::cuda::GpuMat(xy[0]));
                undistMapsGPUY.push_back(cv::cuda::GpuMat(xy[1]));
            }
        }
    }

    // 0 TOP or DOWN
    // 1 left 2 front 3 right 4 back
    cv::cuda::GpuMat undist_id_cuda(cv::Mat image, int _id) {
        cv::cuda::GpuMat img_cuda(image);
        cv::cuda::GpuMat output;
        cv::cuda::remap(img_cuda, output, undistMapsGPUX[_id], undistMapsGPUY[_id], cv::INTER_LINEAR);
        return output;
    }

    std::vector<cv::cuda::GpuMat> undist_all_cuda(const cv::Mat & image, bool use_rgb = false, bool enable_rear = true) {
        cv::cuda::GpuMat img_cuda;
        if (use_rgb) {
            img_cuda.upload(image);
        } else {
            cv::Mat _tmp;
            cv::cvtColor(image, _tmp, cv::COLOR_BGR2GRAY);
            img_cuda.upload(_tmp);
        }

        std::vector<cv::cuda::GpuMat> ret;

        if (enable_rear) {
            for (unsigned int i = 0; i < undistMaps.size(); i++) {
                cv::cuda::GpuMat output;
                cv::cuda::remap(img_cuda, output, undistMapsGPUX[i], undistMapsGPUY[i], cv::INTER_LINEAR);
                ret.push_back(output);
            }
            return ret;
        } else {
            for (unsigned int i = 0; i < undistMaps.size() - 1; i++) {
                cv::cuda::GpuMat output;
                cv::cuda::remap(img_cuda, output, undistMapsGPUX[i], undistMapsGPUY[i], cv::INTER_LINEAR);
                ret.push_back(output);
            }
            return ret; 
        }
    }


    std::vector<cv::Mat> undist_all(const cv::Mat & image, bool use_rgb = false, bool enable_rear = true) {
        std::vector<cv::Mat> ret;

        if (enable_rear) {
            cv::Mat output;
            for (unsigned int i = 0; i < undistMaps.size(); i++) {
                cv::remap(image, output, undistMaps[i], cv::Mat(), cv::INTER_LINEAR);
                ret.push_back(output);
            }
            return ret;
        } else {
            for (unsigned int i = 0; i < undistMaps.size() - 1; i++) {
                cv::Mat output;
                cv::remap(image, output, undistMaps[i], cv::Mat(), cv::INTER_LINEAR);
                ret.push_back(output);
            }
            return ret; 
        }
    }


    std::vector<cv::Mat> generateAllUndistMap(camodocal::CameraPtr p_cam,
                                          Eigen::Vector3d rotation,
                                          const unsigned &imgWidth,
                                          const double &fov //degree
    ) {
        // ROS_INFO("Generating undistortion maps:");
        double sideVerticalFOV = (fov - 180) * DEG_TO_RAD;
        if (sideVerticalFOV < 0)
            sideVerticalFOV = 0;
        double centerFOV = fov * DEG_TO_RAD - sideVerticalFOV * 2;
        ROS_INFO("Center FOV: %f_center", centerFOV);

        // calculate focal length of fake pinhole cameras (pixel size = 1 unit)
        f_center = (double)imgWidth / 2 / tan(centerFOV / 2);
        f_side = (double)imgWidth / 2;

        sideImgHeight = 2 * f_side * tan(sideVerticalFOV/2);

        ROS_INFO("Side image height: %d", sideImgHeight);
        std::vector<cv::Mat> maps;
        maps.reserve(5);

        // test points
        Eigen::Vector3d testPoints[] = {
            Eigen::Vector3d(0, 0, 1),
            Eigen::Vector3d(1, 0, 1),
            Eigen::Vector3d(0, 1, 1),
            Eigen::Vector3d(1, 1, 1),
        };
        for (unsigned int i = 0; i < sizeof(testPoints) / sizeof(Eigen::Vector3d); i++)
        {
            Eigen::Vector2d temp;
            p_cam->spaceToPlane(testPoints[i], temp);
            // ROS_INFO("Test point %d : (%.2f,%.2f,%.2f) projected to (%.2f,%.2f)", i,
                    // testPoints[i][0], testPoints[i][1], testPoints[i][2],
                    // temp[0], temp[1]);
        }
        
        auto t = Eigen::Quaterniond::Identity();

        // ROS_INFO("Pinhole cameras focal length: center %f side %f", f_center, f_side);

        cam_top = camodocal::PinholeCameraPtr( new camodocal::PinholeCamera("top",
                  imgWidth, imgWidth,0, 0, 0, 0,
                  f_center, f_center, imgWidth/2, imgWidth/2));
         
        cx_side = imgWidth/2;
        cy_side = sideImgHeight/2;
        cam_side = camodocal::PinholeCameraPtr(new camodocal::PinholeCamera("side",
                  imgWidth, sideImgHeight,0, 0, 0, 0,
                  f_side, f_side, imgWidth/2, sideImgHeight/2));

        maps.push_back(genOneUndistMap(p_cam, t, imgWidth, imgWidth, f_center));

        if (cam_id == 1) {
            t = Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitX());
        };

        if (sideImgHeight > 0)
        {
            //facing y
            t = t * Eigen::AngleAxis<double>(-M_PI / 2, Eigen::Vector3d(1, 0, 0));
            maps.push_back(genOneUndistMap(p_cam, t, imgWidth, sideImgHeight, f_side));

            //turn right/left?
            t = t * Eigen::AngleAxis<double>(M_PI / 2, Eigen::Vector3d(0, 1, 0));
            maps.push_back(genOneUndistMap(p_cam, t, imgWidth, sideImgHeight, f_side));
            t = t * Eigen::AngleAxis<double>(M_PI / 2, Eigen::Vector3d(0, 1, 0));
            maps.push_back(genOneUndistMap(p_cam, t, imgWidth, sideImgHeight, f_side));
            t = t * Eigen::AngleAxis<double>(M_PI / 2, Eigen::Vector3d(0, 1, 0));
            maps.push_back(genOneUndistMap(p_cam, t, imgWidth, sideImgHeight, f_side));
        }
        return maps;
    }

    cv::Mat genOneUndistMap(
        camodocal::CameraPtr p_cam,
        Eigen::Quaterniond rotation,
        const unsigned &imgWidth,
        const unsigned &imgHeight,
        const double &f_center) {
                cv::Mat map = cv::Mat(imgHeight, imgWidth, CV_32FC2);
        ROS_DEBUG("Generating map of size (%d,%d)", map.size[0], map.size[1]);
        ROS_DEBUG("Perspective facing (%.2f,%.2f,%.2f)",
                (rotation * Eigen::Vector3d(0, 0, 1))[0],
                (rotation * Eigen::Vector3d(0, 0, 1))[1],
                (rotation * Eigen::Vector3d(0, 0, 1))[2]);
        for (unsigned int x = 0; x < imgWidth; x++)
            for (unsigned int y = 0; y < imgHeight; y++)
            {
                Eigen::Vector3d objPoint =
                    rotation *
                    Eigen::Vector3d(
                        ((double)x - (double)imgWidth / 2),
                        ((double)y - (double)imgHeight / 2),
                        f_center);
                Eigen::Vector2d imgPoint;
                p_cam->spaceToPlane(objPoint, imgPoint);
                map.at<cv::Vec2f>(cv::Point(x, y)) = cv::Vec2f(imgPoint.x(), imgPoint.y());
            }

        ROS_DEBUG("Upper corners: (%.2f, %.2f), (%.2f, %.2f)",
                map.at<cv::Vec2f>(cv::Point(0, 0))[0],
                map.at<cv::Vec2f>(cv::Point(0, 0))[1],
                map.at<cv::Vec2f>(cv::Point(imgWidth - 1, 0))[0],
                map.at<cv::Vec2f>(cv::Point(imgWidth - 1, 0))[1]);

        Eigen::Vector3d objPoint =
            rotation *
            Eigen::Vector3d(
                ((double)0 - (double)imgWidth / 2),
                ((double)0 - (double)imgHeight / 2),
                f_center);
        // std::cout << objPoint << std::endl;

        objPoint =
            rotation *
            Eigen::Vector3d(
                ((double)imgWidth / 2),
                ((double)0 - (double)imgHeight / 2),
                f_center);
        // std::cout << objPoint << std::endl;

        return map;
    }

};

}
