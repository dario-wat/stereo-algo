#include "DCBGridStereo.h"

#include <cmath>
#include <ctime>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "st_util.h"
#include "wta.h"

float kernel[] = {0.06136, 0.24477, 0.38774, 0.24477, 0.06136};

// Truncated absolute difference
inline float tad(float pxl_l, float pxl_r, float threshold) {
  return std::min(fabsf(pxl_l-pxl_r), threshold);
}

void DCBGridStereo::tad_matching_cost() {
  for (int d = 0; d < max_disparity; d++) {
    for (int row = 0; row < left.rows; row++) {
      for (int col = d; col < left.cols; col++) {
        cost_volume[d*left.rows*left.cols + row*left.cols + col] =
            tad(left.at<uchar>(row, col), right.at<uchar>(row, col-d), threshold);
      }
    }
  }
}

void DCBGridStereo::create_dcb_grid() {
  for (int d = 0; d < max_disparity; d++) {
    for (int row = 0; row < left.rows; row++) {
      for (int col = d; col < left.cols; col++) {
        int y = round(col / sigma_s), x = round(row / sigma_s);
        int pl = round(left.at<uchar>(row, col) / sigma_r);
        int pr = round(right.at<uchar>(row, col-d) / sigma_r);

        dcb_grid[d*dref + x*xref + y*yref + pl*lref + pr] +=
            cost_volume[d*left.rows*left.cols + row*left.cols + col];
        dcb_counts[d*dref + x*xref + y*yref + pl*lref + pr] += 1;
      }
    }
  }
}

void DCBGridStereo::slice_dcb_grid() {
  for (int d = 0; d < max_disparity; d++) {
    for (int row = 0; row < left.rows; row++) {
      for (int col = 0; col < left.cols; col++) {
        int y = round(col / sigma_s), x = round(row / sigma_s);
        int pl = round(left.at<uchar>(row, col) / sigma_r);
        int pr = round(right.at<uchar>(row, col-d) / sigma_r);

        cost_volume[d*left.rows*left.cols + row*left.cols + col] =
            dcb_grid[d*dref + x*xref + y*yref + pl*lref + pr]
            / dcb_counts[d*dref + x*xref + y*yref + pl*lref + pr];
      }
    }
  }
}

DCBGridStereo::DCBGridStereo(int max_disparity) {
  su::require(max_disparity > 0, "Max disparity must be positive");
  this->max_disparity = max_disparity;
  this->sigma_r = SIGMA_R;
  this->sigma_s = SIGMA_S;
  this->threshold = THRESHOLD;
}

void DCBGridStereo::compute_disparity(const cv::Mat &left, const cv::Mat &right) {
  su::require(left.cols == right.cols && left.rows == right.rows, "Image dimensions must match");
  su::require(left.type() == CV_8UC1 && right.type() == CV_8UC1, "Images must be grayscale");
  this->left = left;
  this->right = right;
  cost_volume = new float[max_disparity*left.cols*left.rows];

  this->xdim = ceil(left.rows/sigma_s) + 1;
  this->ydim = ceil(left.cols/sigma_s) + 1;
  this->ldim = ceil(256/sigma_r) + 1;
  this->rdim = ceil(256/sigma_r) + 1;
  this->dref = xdim*ydim*ldim*rdim;
  this->xref = ydim*ldim*rdim;
  this->yref = ldim*rdim;
  this->lref = rdim;

  dcb_grid = new float[max_disparity*xdim*ydim*ldim*rdim] ();
  dcb_counts = new float[max_disparity*xdim*ydim*ldim*rdim] ();

  tad_matching_cost();
  create_dcb_grid();
  slice_dcb_grid();

  cv::Mat disparity;
  su::wta(disparity, cost_volume, max_disparity, left.rows, left.cols);

  su::convert_to_disparity_visualize(disparity, disparity, true);
  // disparity.convertTo(disparity, CV_8UC1);
  cv::imshow("disp", disparity);
  cv::waitKey(0);

  delete[] cost_volume;
  delete[] dcb_grid;
  delete[] dcb_counts;
}
