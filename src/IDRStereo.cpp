#include "IDRStereo.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cmath>
#include "st_util.h"
#include <algorithm>

#define LAMBDA_C 30.91
#define LAMBDA_G 28.21

/////// Temporary
#include <iostream>
using std::cout;
using std::cerr;
using std::endl;
////////////////


IDRStereo::IDRStereo(int max_disparity, int win_size) {
  this->max_disparity = max_disparity;
  this->win_size = win_size;
}

void IDRStereo::compute_disparity(const cv::Mat &left, const cv::Mat &right) {
  float *cost_volume = new float[left.cols*left.rows*max_disparity];
  std::fill_n(cost_volume, left.cols*left.rows*max_disparity, 1000.0f);
  cost_aggregation(left, right, cost_volume);
  delete[] cost_volume;
}

// inline float asw_vert(const cv::Mat &left, const cv::Mat &right, int win_size, int i, int j) {
//   float nom = 0.0f, den = 0.0f;
//   int offset = win_size / 2;
//   for (int c = offset; c < left.rows - offset; c++) {

//   }
// }


void IDRStereo::cost_aggregation(const cv::Mat &left, const cv::Mat &right, float *cost_volume) {
  int offset = win_size / 2;
  
  for (int d = 0; d < max_disparity; d++) {
    for (int row = offset; row < left.rows-offset; row++) {
      for (int col = offset+d; col < left.cols-offset; col++) {
  
        float nom = 0.0f, den = 0.0f;
        for (int c = col-offset; c < col+offset; c++) {
          float w1 = exp(- fabsf((int)left.at<uchar>(row, col) - left.at<uchar>(row, c)) / LAMBDA_C
              // - fabsf(c-col) / LAMBDA_G
              );
          float w2 = exp(- fabsf((int)right.at<uchar>(row, col-d) - right.at<uchar>(row, c-d)) / LAMBDA_C
              // - fabsf(c-col) / LAMBDA_G
              );
          nom += w1 * w2 * fabsf((int)left.at<uchar>(row, c) - right.at<uchar>(row, c-d));
          den += w1 * w2;
        }
        cost_volume[d*left.rows*left.cols + row*left.cols + col] = nom / den;
  
      }
    }
  }

  for (int d = 0; d < max_disparity; d++) {
    for (int i = offset; i < left.rows-offset; i++) {
      for (int j = offset+d; j < left.cols-offset; j++) {
  
        float nom = 0.0f, den = 0.0f;
        for (int c = j-offset; c < j+offset; c++) {
          float w1 = exp(- fabsf((int)left.at<uchar>(i, j) - left.at<uchar>(c, j)) / LAMBDA_C
              // - fabsf(c-i) / LAMBDA_G
              );
          float w2 = exp(- fabsf((int)right.at<uchar>(i, j-d) - right.at<uchar>(c, j-d)) / LAMBDA_C
              // - fabsf(c-j) / LAMBDA_G
              );
          nom += w1 * w2 * cost_volume[d*left.rows*left.cols + i*left.cols + j];
          // cout << d*left.rows*left.cols + c*left.cols + j << endl;
          den += w1 * w2;
        }
        cost_volume[d*left.rows*left.cols + i*left.cols + j] = nom / den;
  
      }
    }
  }

  // for (int d = 0; d < max_disparity; d++) {
  //   for (int j = offset+d; j < left.cols-offset; j++) {
  //     for (int i = offset; i < left.rows-offset; i++) {
  //       float nom = 0.0f, den = 0.0f;
  //       for (int c = j-offset; c < j+offset; c++) {
  //         for (int h = i-offset; h < i+offset; h++) {  
  //           float w1 = exp(
  //               - fabsf((int)left.at<uchar>(i, j) - left.at<uchar>(h, c)) / LAMBDA_C
  //               - sqrt((i-h)*(i-h) + (j-c)*(j-c)) / LAMBDA_G
  //               );
  //           float w2 = exp(
  //               - fabsf((int)right.at<uchar>(i, j-d) - right.at<uchar>(h, c-d)) / LAMBDA_C
  //               - sqrt((i-h)*(i-h) + (j-c)*(j-c)) / LAMBDA_G
  //               );
  //           nom += w1 * w2 * fabsf((int)left.at<uchar>(h, c) - right.at<uchar>(h, c-d));//cost_volume[d*left.rows*left.cols + i*left.cols + j];
  //           den += w1 * w2;
  //         }
  //         cost_volume[d*left.rows*left.cols + i*left.cols + j] = nom / den;
  //       // cout << nom/den << endl;
  //       }
  //     }
  //   }
  // }

  int rows = left.rows, cols = left.cols;
  cv::Mat disparity(rows, cols, CV_16SC1);
    cv::Mat min_costs = cv::Mat::ones(rows, cols, CV_32FC1) * 10000.0f;
    for (int d = 0; d < max_disparity; d++) {
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                if (cost_volume[d*cols*rows + row*cols + col] < min_costs.at<float>(row, col)) {
                    min_costs.at<float>(row, col) = cost_volume[d*cols*rows + row*cols + col];
                    disparity.at<short>(row, col) = d;
                }
            }
        }
    }
    // cout << disparity << endl;
  cv::Mat dis_vis;
  su::convert_to_disparity_visualize(disparity, dis_vis);
  cv::imshow("Disp", dis_vis);
  cv::waitKey(0);
}