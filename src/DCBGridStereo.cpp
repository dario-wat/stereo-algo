#include "DCBGridStereo.h"

#include <cmath>
#include <ctime>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "st_util.h"
#include "wta.h"


inline float euclidean(float x1, float y1, float x2, float y2) {
  return sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2));
}

// This is the weight used for Yoon and Kweon adaptive support weight
float DCBGridStereo::weight(  float pxl_p, float pxl_q, float xp, float yp, float xq, float yq,
                              float gamma_c=GAMMA_C, float gamma_p=GAMMA_P) {
  return exp(-fabsf(pxl_p-pxl_q)/gamma_c - euclidean(xp, yp, xq, yq)/gamma_p);
}

// Gaussian weight
// flaot DCBGridStereo::weight_g() {
  
// }

// Truncated absolute difference
float DCBGridStereo::tad(float pxl_l, float pxl_r, float threshold=THRESHOLD) {
  return std::min(fabsf(pxl_l-pxl_r), threshold);
}



float DCBGridStereo::yoon_kweon_pxl(const cv::Mat &left, const cv::Mat &right,
                                    int row, int col, int d, int ws, float *table) {
  float nom = 0, den = 0;
  for (int i = row - ws/2; i < row + ws/2; i++) {
    for (int j = col - ws/2; j < col + ws/2; j++) {
      float w =  weight(left.at<uchar>(row, col), left.at<uchar>(i, j), row, col, i, j)
          * weight(right.at<uchar>(row, col-d), right.at<uchar>(i, j-d), row, col-d, i, j-d);
      nom += w;
      den += w * table[255 + static_cast<int>(left.at<uchar>(i, j)) - right.at<uchar>(i, j-d)];
      //tad(left.at<uchar>(i, j), right.at<uchar>(i, j-d));
    }
  }
  return nom / den;
}

void DCBGridStereo::yoon_kweon( float *cost_volume, const cv::Mat &left, const cv::Mat &right,
                                int ws=WINDOW_SIZE) {
  float *table = new float[512];
  for (int diff = -255; diff < 256; diff++) {
    table[diff+255] = std::min(fabsf(diff), THRESHOLD);
  }

  for (int d = 0; d < max_disparity; d++) {
    clock_t start = clock();
    std::cout << d << std::endl;
    for (int row = ws/2; row < left.rows-ws/2; row++) {
      for (int col = d; col < left.cols-ws/2; col++) {
        cost_volume[d*left.cols*left.rows + row*left.cols + col] =
            yoon_kweon_pxl(left, right, row, col, d, ws, table);
      }
    }
    std::cout << float(clock() - start) / CLOCKS_PER_SEC << std::endl;
  }
  
  delete[] table;
}

DCBGridStereo::DCBGridStereo(int max_disparity) {
  su::require(max_disparity > 0, "Max disparity must be positive");
  this->max_disparity = max_disparity;
}

void DCBGridStereo::compute_disparity(const cv::Mat &left, const cv::Mat &right) {
  cv::Mat disparity = cv::Mat(left.rows, left.cols, CV_8UC1);
  float *cost_volume = new float[left.cols*left.rows*max_disparity];
  yoon_kweon(cost_volume, left, right);
  su::wta(disparity, cost_volume, max_disparity, left.rows, left.cols);
  disparity.convertTo(disparity, CV_8UC1);
  cv::imshow("disp", disparity);
  cv::waitKey(0);
  delete[] cost_volume;
}
