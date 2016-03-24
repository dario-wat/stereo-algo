#include "DCBGridStereo.h"

#include <cmath>
#include <ctime>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "st_util.h"
#include "wta.h"

// float kernel[] = {0.06136, 0.24477, 0.38774, 0.24477, 0.06136};
// float kernel[] = {0.2, 0.2, 0.2, 0.2, 0.2};

int n4[16][4] = {
    {0, 0, 0, 0},
    {1, 0, 0, 0},
    {0, 1, 0, 0},
    {1, 1, 0, 0},
    {0, 0, 1, 0},
    {1, 0, 1, 0},
    {0, 1, 1, 0},
    {1, 1, 1, 0},
    {0, 0, 0, 1},
    {1, 0, 0, 1},
    {0, 1, 0, 1},
    {1, 1, 0, 1},
    {0, 0, 1, 1},
    {1, 0, 1, 1},
    {0, 1, 1, 1},
    {1, 1, 1, 1}};

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
        int y = round(col / sigma_s);
        int x = round(row / sigma_s);
        int pl = round(left.at<uchar>(row, col) / sigma_r);
        int pr = round(right.at<uchar>(row, col-d) / sigma_r);

        dcb_grid[d*dref + x*xref + y*yref + pl*lref + pr] +=
            cost_volume[d*left.rows*left.cols + row*left.cols + col];
        dcb_counts[d*dref + x*xref + y*yref + pl*lref + pr] += 1;
      }
    }
  }
}

void create_gaussian_kernel(float *kernel, int n, float sigma) {
  float s = 0;
  for (int i = -n/2; i <= n/2; i++) {
    kernel[i+n/2] = exp(-i*i / (2*sigma*sigma));
    s += kernel[i+n/2];
  }
  for (int i = 0; i < n; i++) {
    kernel[i] /= s;
  }
}

// Reflect pixel if it goes outside of the image boundary
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
// Gaussian smooth in 4d
void gaussian_smooth(float *data_4d, int d1, int d2, int d3, int d4, int n, float sigma) {
  float *data_4d_cpy = new float[d1*d2*d3*d4];
  memcpy(data_4d_cpy, data_4d, d1*d2*d3*d4*sizeof(float));
  float *kernel = new float[n];
  create_gaussian_kernel(kernel, n, sigma);

  for (int i = 0; i < d1; i++) {
    for (int j = 0; j < d2; j++) {
      for (int k = 0; k < d3; k++) {
        for (int l = 0; l < d4; l++) {
          float sum = 0.0f;
          for (int p = -n/2; p <= n/2; p++) {
            int i1 = reflect(d1, i-p);
            sum += kernel[p+n/2] * data_4d_cpy[i1*d2*d3*d4 + j*d3*d4 + k*d4 + l];
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
          for (int p = -n/2; p <= n/2; p++) {
            int j1 = reflect(d2, j-p);
            sum += kernel[p+n/2] * data_4d_cpy[i*d2*d3*d4 + j1*d3*d4 + k*d4 + l];
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
          for (int p = -n/2; p <= n/2; p++) {
            int k1 = reflect(d3, k-p);
            sum += kernel[p+n/2] * data_4d_cpy[i*d2*d3*d4 + j*d3*d4 + k1*d4 + l];
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
          for (int p = -n/2; p <= n/2; p++) {
            int l1 = reflect(d4, l-p);
            sum += kernel[p+n/2] * data_4d_cpy[i*d2*d3*d4 + j*d3*d4 + k*d4 + l1];
          }
          data_4d[i*d2*d3*d4 + j*d3*d4 + k*d4 + l] = sum;
        }
      }
    }
  }

  delete[] data_4d_cpy;
  delete[] kernel;
}

void DCBGridStereo::process_dcb_grid() {
  for (int d = 0; d < max_disparity; d++) {
    gaussian_smooth(dcb_grid + d*xdim*ydim*ldim*rdim, xdim, ydim, ldim, rdim, SIZE, SIGMA);
  }
}

// Looks at these 4 functions, consistency is very important
inline float interpolation_1d(float v1, float v2, float x) {
  return v1 * (1-x)  + x * v2;
}

inline float interpolation_2d(float v[4], float x, float y) {
  float s = interpolation_1d(v[0], v[1], x);
  float t = interpolation_1d(v[2], v[3], x);
  return interpolation_1d(s, t, y);
}

inline float interpolation_3d(float v[8], float x[3]) {
  float s[4];
  s[0] = interpolation_1d(v[0], v[1], x[0]);
  s[1] = interpolation_1d(v[2], v[3], x[0]);
  s[2] = interpolation_1d(v[4], v[5], x[0]);
  s[3] = interpolation_1d(v[6], v[7], x[0]);
  return interpolation_2d(s, x[1], x[2]);
}

inline float interpolation_4d(const float v[16], float x[4]) {
  float s[8];
  for (int i = 0; i < 8; i++) {
    s[i] = interpolation_1d(v[2*i], v[2*i+1], x[0]);
  }
  float y[3] = { x[1], x[2], x[3]};
  return interpolation_3d(s, y);
}

// Taking the values out of the dcb grid and creating the final matching cost volume
void DCBGridStereo::slice_dcb_grid() {
  for (int d = 0; d < max_disparity; d++) {
    for (int row = 0; row < left.rows; row++) {
      for (int col = 0; col < left.cols; col++) {
        int x = row / sigma_s, y = col / sigma_s;
        int pl = left.at<uchar>(row, col) / sigma_r, pr = right.at<uchar>(row, col-d) / sigma_r;

        float coord_diffs[4] = {
            static_cast<float>(row) / sigma_s - x,
            static_cast<float>(col) / sigma_s - y,
            static_cast<float>(left.at<uchar>(row, col)) / sigma_r - pl,
            static_cast<float>(right.at<uchar>(row, col-d)) / sigma_r - pr};

        // Huh?
        float values[16], counts[16];
        for (int i = 0; i < 16; i++) {
          // This complex if-else is so that it does index outside of the grid. If it does, it
          // will be truncated to the lower value
          int dx = x+n4[i][0] >= xdim ? 0 : n4[i][0];
          int dy = y+n4[i][1] >= ydim ? 0 : n4[i][1];
          int dl = pl+n4[i][2] >= ldim ? 0 : n4[i][2];
          int dr = pr+n4[i][3] >= rdim ? 0 : n4[i][3];
          values[i] = dcb_grid[d*dref + (x+dx)*xref + (y+dy)*yref + (pl+dl)*lref + pr+dr];
          counts[i] = dcb_counts[d*dref + (x+dx)*xref + (y+dy)*yref + (pl+dl)*lref + pr+dr];
        }

        cost_volume[d*left.rows*left.cols + row*left.cols + col] =
            interpolation_4d(values, coord_diffs) / interpolation_4d(counts, coord_diffs);
      }
    }
  }
}

DCBGridStereo::DCBGridStereo(int max_disparity, float sigma_s, float sigma_r, float threshold) {
  su::require(max_disparity > 0, "Max disparity must be positive");
  su::require(sigma_r >= 1, "SigmaR must be >= 1");
  su::require(sigma_s >= 1, "SigmaS must be >= 1");
  su::require(threshold > 0, "TAD threshold must be positive");
  this->max_disparity = max_disparity;
  this->sigma_r = sigma_r;
  this->sigma_s = sigma_s;
  this->threshold = sigma_r;
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

  std::cerr << "Dimensions: "
      << max_disparity << " x " << xdim << " x " << ydim << " x " << ldim << " x " << rdim
      << std::endl;
  std::cerr << "Trying to allocate: "
      << static_cast<float>(xdim)*ydim*ldim*rdim*max_disparity*2*sizeof(float)/1024/1204 << " MB"
      << std::endl;

  dcb_grid = new float[max_disparity*xdim*ydim*ldim*rdim] ();
  dcb_counts = new float[max_disparity*xdim*ydim*ldim*rdim] ();

  tad_matching_cost();
  create_dcb_grid();
  process_dcb_grid();
  slice_dcb_grid();

  cv::Mat disparity;
  su::wta(disparity, cost_volume, max_disparity, left.rows, left.cols);

  su::convert_to_disparity_visualize(disparity, disparity, true);
  cv::imshow("disp", disparity);
  cv::waitKey(0);

  delete[] cost_volume;
  delete[] dcb_grid;
  delete[] dcb_counts;
}
