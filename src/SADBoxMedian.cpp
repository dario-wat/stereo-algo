#include "SADBoxMedian.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <algorithm>
#include "wta.h"
#include "st_util.h"

#define FL_MAX 10000.0f


SADBoxMedian::SADBoxMedian(int max_disparity, int box_size, int median_size) {
  // su::require(max_disparity > 0 && max_disparity < 256, "Max disparity must be > 0 and < 256");
  su::require(box_size > 0 && box_size % 2, "Box filter size must be positive odd");
  su::require(median_size > 0 && median_size % 2, "Median filter size must be positive odd");
  this->max_disparity = max_disparity;
  this->box_size = box_size;
  this->median_size = median_size;
}

cv::Mat SADBoxMedian::compute_disparity(const cv::Mat &left, const cv::Mat &right) {
  su::require(left.cols == right.cols && left.rows == right.rows, "Image dimensions must match");
  su::require(left.type() == CV_8UC1 && right.type() == CV_8UC1, "Images must be grayscale");

  int rows = left.rows, cols = left.cols;
  float *cost_volume = new float[rows*cols*max_disparity];
  std::fill_n(cost_volume, rows*cols*max_disparity, FL_MAX);

  // SAD cost matching
  for (int d = 0; d < max_disparity; d++) {
    for (int row = 0; row < rows; row++) {
      for (int col = d; col < cols; col++) {
        cost_volume[d*rows*cols + row*cols + col] =
            fabsf((int)left.at<uchar>(row, col) - right.at<uchar>(row, col-d));
      }
    }
  }

  // Box cost aggregating
  for (int d = 0; d < max_disparity; d++) {
    cv::Mat cost_plane = cv::Mat(rows, cols, CV_32FC1, cost_volume + d*cols*rows);
    cv::boxFilter(cost_plane, cost_plane, -1, cv::Size(box_size, box_size));
  }

  // WTA optimization
  cv::Mat disparity;
  su::wta(disparity, cost_volume, max_disparity, rows, cols);

  // TODO this is stupid, I should make my own median filter that works on integers, duh
  disparity.convertTo(disparity, CV_32FC1);

  // Post-processing
  cv::medianBlur(disparity, disparity, median_size);

  disparity.convertTo(disparity, CV_32SC1);

  delete[] cost_volume;

  return disparity;
}
