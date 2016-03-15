#include "wta.h"

#include <opencv2/core/core.hpp>

void su::wta(cv::Mat &disparity, float *cost_volume, int max_d, int rows, int cols) {
  if (disparity.cols != cols || disparity.rows != rows || disparity.type() != CV_32SC1) {
    disparity = cv::Mat(rows, cols, CV_32SC1);
  }
  
  cv::Mat min_costs = cv::Mat::ones(rows, cols, CV_32FC1) * FL_MAX;
  for (int d = 0; d < max_d; d++) {
    for (int row = 0; row < rows; row++) {
      for (int col = 0; col < cols; col++) {
        if (cost_volume[d*cols*rows + row*cols + col] < min_costs.at<float>(row, col)) {
          min_costs.at<float>(row, col) = cost_volume[d*cols*rows + row*cols + col];
          disparity.at<int>(row, col) = d;
        }
      }
    }
  }
}