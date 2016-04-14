#include "wta.h"

#include <opencv2/core/core.hpp>

// TODO this is for backwards compatibility, will remove if there is time
void su::wta(cv::Mat &disparity, const float *cost_volume, int max_d, int rows, int cols) {
  wta(disparity, cost_volume, 0, max_d, rows, cols);
}

void su::wta(cv::Mat &disparity, const float *cost_volume, int min_d, int max_d, int rows, int cols) {
  disparity = cv::Mat::ones(rows, cols, CV_32SC1) * min_d;
  cv::Mat min_costs = cv::Mat::ones(rows, cols, CV_32FC1) * FL_MAX;
  for (int d = min_d; d < max_d; d++) {
    for (int row = 0; row < rows; row++) {
      for (int col = 0; col < cols; col++) {
        if (cost_volume[(d-min_d)*cols*rows + row*cols + col] < min_costs.at<float>(row, col)) {
          min_costs.at<float>(row, col) = cost_volume[(d-min_d)*cols*rows + row*cols + col];
          disparity.at<int>(row, col) = d;
        }
      }
    }
  }
}
