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

inline int reflect(int M, int x) {
  if (x < 0) {
    return -x - 1;
  }
  if (x >= M) {
    return 2*M - x - 1;
  }
  return x;
}

// YOLO
void gaussian_smooth(float *data_4d, int d1, int d2, int d3, int d4) {
  float *data_4d_cpy = new float[d1*d2*d3*d4];
  memcpy(data_4d_cpy, data_4d, d1*d2*d3*d4*sizeof(float));
  for (int i = 0; i < d1; i++) {
    for (int j = 0; j < d2; j++) {
      for (int k = 0; k < d3; k++) {
        for (int l = 0; l < d4; l++) {
          float sum = 0.0f;
          for (int p = -2; p <= 2; p++) {
            int i1 = reflect(d1, i-p);
            sum += kernel[p+2] * data_4d_cpy[i1*d2*d3*d4 + j*d3*d4 + k*d4 + l];
          }
          data_4d[i*d2*d3*d4 + j*d3*d4 + k*d4 + l] = sum;
        }
      }
    }
  }

  for (int i = 0; i < d1; i++) {
    for (int j = 0; j < d2; j++) {
      for (int k = 0; k < d3; k++) {
        for (int l = 0; l < d4; l++) {
          float sum = 0.0f;
          for (int p = -2; p <= 2; p++) {
            int j1 = reflect(d2, j-p);
            sum += kernel[p+2] * data_4d_cpy[i*d2*d3*d4 + j1*d3*d4 + k*d4 + l];
          }
          data_4d[i*d2*d3*d4 + j*d3*d4 + k*d4 + l] = sum;
        }
      }
    }
  }

  for (int i = 0; i < d1; i++) {
    for (int j = 0; j < d2; j++) {
      for (int k = 0; k < d3; k++) {
        for (int l = 0; l < d4; l++) {
          float sum = 0.0f;
          for (int p = -2; p <= 2; p++) {
            int k1 = reflect(d3, k-p);
            sum += kernel[p+2] * data_4d_cpy[i*d2*d3*d4 + j*d3*d4 + k1*d4 + l];
          }
          data_4d[i*d2*d3*d4 + j*d3*d4 + k*d4 + l] = sum;
        }
      }
    }
  }

  for (int i = 0; i < d1; i++) {
    for (int j = 0; j < d2; j++) {
      for (int k = 0; k < d3; k++) {
        for (int l = 0; l < d4; l++) {
          float sum = 0.0f;
          for (int p = -2; p <= 2; p++) {
            int l1 = reflect(d4, l-p);
            sum += kernel[p+2] * data_4d_cpy[i*d2*d3*d4 + j*d3*d4 + k*d4 + l1];
          }
          data_4d[i*d2*d3*d4 + j*d3*d4 + k*d4 + l] = sum;
        }
      }
    }
  }

  delete[] data_4d_cpy;
}

void DCBGridStereo::process_dcb_grid() {
  for (int d = 0; d < max_disparity; d++) {
    gaussian_smooth(dcb_grid + d*xdim*ydim*ldim*rdim, xdim, ydim, ldim, rdim);
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
  process_dcb_grid();
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
