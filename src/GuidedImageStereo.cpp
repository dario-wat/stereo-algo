#include "GuidedImageStereo.h"

#include <algorithm>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "st_util.h"

/************ DEBUGGING STUFF **************/

#include <opencv2/highgui/highgui.hpp>
#include <iostream>

/*******************************************/

GuidedImageStereo::GuidedImageStereo(int max_disparity) {
  su::require(max_disparity > 0, "Max disparity must be positive");
  this->max_disparity= max_disparity;
}

// Central differences in x direction only
void gradient(const cv::Mat &src, cv::Mat &dst) {
  if (src.cols != dst.cols || src.rows != dst.rows || dst.type() != CV_32FC1) {
    dst = cv::Mat::zeros(src.rows, src.cols, CV_32FC1);
  }

  // Left border
  for (int row = 0; row < src.rows; row++) {
    dst.at<float>(row, 0) = src.at<float>(row, 1) - src.at<float>(row, 0);
  }

  // Internal pixels
  for (int row = 0; row < src.rows; row++) {
    for (int col = 1; col < src.cols-1; col++) {
      dst.at<float>(row, col) = src.at<float>(row, col+1) - src.at<float>(row, col-1);
    }
  }

  // Right border
  for (int row = 0; row < src.rows; row++) {
    dst.at<float>(row, src.cols-1) = src.at<float>(row, src.cols-1) - src.at<float>(row, src.cols-2);
  }

  // Transform from -255..255 range into 0..1 range
  dst = (dst + 1.0) / 2.0;
}

// Using TAD to compute costs
void GuidedImageStereo::initial_cost_volume() {
  // wrt left image
  for (int d = 0; d < max_disparity; d++) {
    for (int row = 0; row < rows; row++) {
      for (int col = d; col < cols; col++) {
        float color = std::min(fabsf(left.at<float>(row, col) - right.at<float>(row, col-d)), COLOR_THR);
        float grad = std::min(fabsf(dx_left.at<float>(row, col) - dx_right.at<float>(row, col-d)), GRAD_THR);
        cost_volume_l[d*cols*rows + row*cols + col] = GAMMA*color + (1-GAMMA)*grad;
      }
    }
  }

  // wrt right image
  for (int d = 0; d < max_disparity; d++) {
    for (int row = 0; row < rows; row++) {
      for (int col = 0; col < cols-d; col++) {
        float color = std::min(fabsf(left.at<float>(row, col+d) - right.at<float>(row, col)), COLOR_THR);
        float grad = std::min(fabsf(dx_left.at<float>(row, col+d) - dx_right.at<float>(row, col)), GRAD_THR);
        cost_volume_r[d*cols*rows + row*cols + col] = GAMMA*color + (1-GAMMA)*grad;
      }
    }
  }
}

void GuidedImageStereo::compute_disparity(const cv::Mat &left, const cv::Mat &right) {
  su::require(left.rows == right.rows && left.cols == right.cols, "Image dimension must match");
  su::require(left.type() == CV_8UC1 && right.type() == CV_8UC1, "Images must be grayscale");

  this->rows = left.rows;
  this->cols = left.cols;

  left.convertTo(this->left, CV_32FC1);
  this->left /= 255.0;
  right.convertTo(this->right, CV_32FC1);
  this->right /= 255.0;

  gradient(this->left, dx_left);
  gradient(this->right, dx_right);

  cost_volume_l = new float[left.rows * left.cols * max_disparity] ();
  cost_volume_r = new float[left.rows * left.cols * max_disparity] ();
  std::fill_n(cost_volume_l, left.rows*left.cols*max_disparity, BORDER_THR);
  std::fill_n(cost_volume_r, left.rows*left.cols*max_disparity, BORDER_THR);

  initial_cost_volume();

  cv::imshow("Left", dx_left);
  cv::imshow("Right", dx_right);
  cv::waitKey(0);

  delete[] cost_volume_l;
  delete[] cost_volume_r;
}
