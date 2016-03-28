#include "GuidedImageStereo.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

/************ DEBUGGING STUFF **************/

#include <opencv2/highgui/highgui.hpp>
#include <iostream>

/*******************************************/

// Central differences in x direction only
void gradient(const cv::Mat &src, cv::Mat &dst) {
  if (src.cols != dst.cols || src.rows != dst.rows || dst.type() != CV_32FC1) {
    dst = cv::Mat::zeros(src.rows, src.cols, CV_32FC1);
  }

  // Left border
  for (int row = 0; row < src.rows; row++) {
    dst.at<float>(row, 0) = src.at<uchar>(row, 1) - src.at<uchar>(row, 0);
  }

  // Internal pixels
  for (int row = 0; row < src.rows; row++) {
    for (int col = 1; col < src.cols-1; col++) {
      dst.at<float>(row, col) = src.at<uchar>(row, col+1) - src.at<uchar>(row, col-1);
    }
  }

  // Right border
  for (int row = 0; row < src.rows; row++) {
    dst.at<float>(row, src.cols-1) = src.at<uchar>(row, src.cols-1) - src.at<uchar>(row, src.cols-2);
  }

  // Transform from -255..255 range into 0..1 range
  dst = (dst + 255) / 510;
}

GuidedImageStereo::GuidedImageStereo() {

}

void GuidedImageStereo::compute_disparity(const cv::Mat &left, const cv::Mat &right) {
  cv::Mat dx_left, dx_right;
  gradient(left, dx_left);
  gradient(right, dx_right);

  cv::imshow("Left", dx_left);
  cv::imshow("Right", dx_right);
  cv::waitKey(0);
}
