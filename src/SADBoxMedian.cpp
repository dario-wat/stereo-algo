#include "SADBoxMedian.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <algorithm>
#include "AbstractStereoAlgorithm.h"
#include "wta.h"
#include "st_util.h"

#define FL_MAX 10000.0f


SADBoxMedian::SADBoxMedian( int min_disparity, int max_disparity, int rows, int cols,
                            int box_size, int median_size)
    : AbstractStereoAlgorithm(min_disparity, max_disparity, rows, cols) {
  su::require(box_size > 0 && box_size % 2, "Box filter size must be positive odd");
  su::require(median_size > 0 && median_size % 2, "Median filter size must be positive odd");
  this->box_size = box_size;
  this->median_size = median_size;
}

cv::Mat SADBoxMedian::compute_disparity(const cv::Mat &left, const cv::Mat &right) {
  su::require(left.cols == right.cols && left.rows == right.rows, "Image dimensions must match");
  su::require(left.type() == CV_8UC1 && right.type() == CV_8UC1, "Images must be grayscale");
  su::require(left.cols == cols && left.rows == rows, "Images must fit the sizes given in the constructor");

  std::fill_n(cost_volume, rows*cols*disparity_range, FL_MAX);

  // SAD cost matching
  for (int d = min_disparity; d < max_disparity; d++) {
    for (int row = 0; row < rows; row++) {
      for (int col = d; col < cols; col++) {
        cost_volume[(d-min_disparity)*rows*cols + row*cols + col] =
            fabsf((int)left.at<uchar>(row, col) - right.at<uchar>(row, col-d));
      }
    }
  }

  // Box cost aggregating
  for (int d = min_disparity; d < max_disparity; d++) {
    cv::Mat cost_plane = cv::Mat(rows, cols, CV_32FC1, cost_volume + (d-min_disparity)*cols*rows);
    cv::boxFilter(cost_plane, cost_plane, -1, cv::Size(box_size, box_size));
  }

  // WTA optimization
  cv::Mat disparity;
  su::wta(disparity, cost_volume, min_disparity, max_disparity, rows, cols);

  // Post-processing
  disparity.convertTo(disparity, CV_16SC1);
  cv::medianBlur(disparity, disparity, median_size);
  disparity.convertTo(disparity, CV_32SC1);

  return disparity;
}
