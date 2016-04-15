#include "GuidedImageStereo.h"

#include <cmath>
#include <algorithm>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "AbstractStereoAlgorithm.h"
#include "st_util.h"
#include "wta.h"

GuidedImageStereo::GuidedImageStereo( int min_disparity, int max_disparity, int rows, int cols,
                                      float gamma_c, float gamma_p)
    : AbstractStereoAlgorithm(min_disparity, max_disparity, rows, cols) {
  su::require(gamma_c > 0, "Gamma_c must be positive");
  su::require(gamma_p > 0, "Gamma_p must be positive");
  this->gamma_c = gamma_c;
  this->gamma_p = gamma_p;
  this->cost_volume_r = new float[rows * cols * disparity_range];
  this->cost_volume_l = cost_volume;
}

GuidedImageStereo::~GuidedImageStereo() {
  delete[] cost_volume_r;
}

// Central differences in x direction only
void GuidedImageStereo::gradient(const cv::Mat &src, cv::Mat &dst) {
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

// Using TAD for cost matching
void GuidedImageStereo::initial_cost_volume() {
  cv::Mat dx_left, dx_right;
  gradient(left, dx_left);
  gradient(right, dx_right);

  // wrt left image
  for (int d = min_disparity; d < max_disparity; d++) {
    for (int row = 0; row < rows; row++) {
      for (int col = d; col < cols; col++) {
        float color = std::min(fabsf(left.at<float>(row, col) - right.at<float>(row, col-d)), COLOR_THR);
        float grad = std::min(fabsf(dx_left.at<float>(row, col) - dx_right.at<float>(row, col-d)), GRAD_THR);
        cost_volume_l[(d-min_disparity)*cols*rows + row*cols + col] = GAMMA*color + (1-GAMMA)*grad;
      }
    }
  }

  // wrt right image
  for (int d = min_disparity; d < max_disparity; d++) {
    for (int row = 0; row < rows; row++) {
      for (int col = 0; col < cols-d; col++) {
        float color = std::min(fabsf(left.at<float>(row, col+d) - right.at<float>(row, col)), COLOR_THR);
        float grad = std::min(fabsf(dx_left.at<float>(row, col+d) - dx_right.at<float>(row, col)), GRAD_THR);
        cost_volume_r[(d-min_disparity)*cols*rows + row*cols + col] = GAMMA*color + (1-GAMMA)*grad;
      }
    }
  }
}

// No clue, matlab code told me to do it. Looks very matlaby as well.
void GuidedImageStereo::guided_filter(const cv::Mat &I, cv::Mat &p, int r) {
  cv::Mat mean_I, mean_p, mean_Ip;
  cv::Size size = cv::Size(r, r);
  cv::boxFilter(I, mean_I, -1, size);
  cv::boxFilter(p, mean_p, -1, size);
  cv::boxFilter(I.mul(p), mean_Ip, -1, size);
  cv::Mat cov_Ip = mean_Ip - mean_I.mul(mean_p);

  cv::Mat mean_II;
  cv::boxFilter(I.mul(I), mean_II, -1, size);
  cv::Mat var_I = mean_II - mean_I.mul(mean_I);

  cv::Mat a = cov_Ip / (var_I + EPS);
  cv::Mat b = mean_p - a.mul(mean_I);

  cv::Mat mean_a, mean_b;
  cv::boxFilter(a, mean_a, -1, size);
  cv::boxFilter(b, mean_b, -1, size);

  p = mean_a.mul(I) + mean_b;
}

void GuidedImageStereo::guided_cost_aggregation() {
  for (int d = min_disparity; d < max_disparity; d++) {
    // This solution here is used so that the opencv implementation of the box filter can be used
    // to filter the cost volume. This works because Mat is internally just a pointer to data.
    cv::Mat disp_plane_l = cv::Mat(rows, cols, CV_32FC1, cost_volume_l + (d-min_disparity)*rows*cols);
    guided_filter(left, disp_plane_l, R);
    cv::Mat disp_plane_r = cv::Mat(rows, cols, CV_32FC1, cost_volume_r + (d-min_disparity)*rows*cols);
    guided_filter(right, disp_plane_r, R);
  }
}

// left-right consistancy check. Invalidating pixels and returning them as bad
void GuidedImageStereo::lr_check(cv::Mat &bad, const cv::Mat &disp_left, const cv::Mat &disp_right) {
  bad = cv::Mat::zeros(disp_left.rows, disp_left.cols, CV_8UC1);
  for (int row = 0; row < disp_left.rows; row++) {
    for (int col = 0; col < disp_left.cols; col++) {
      int d = disp_left.at<int>(row, col);
      if (std::abs(d - disp_right.at<int>(row, col-d)) >= 1) {
        bad.at<uchar>(row, col) = 255;
      }
    }
  }
}

// Filling pixels invalidated by lr check using some streak-based filling
void GuidedImageStereo::fill_invalidated(cv::Mat &disp, const cv::Mat &invalidated_mask, int max_d) {
  for (int row = 0; row < disp.rows; row++) {
    for (int col = 0; col < disp.cols; col++) {
      disp.at<int>(row, col) = invalidated_mask.at<uchar>(row, col) > 0 ? -1 : disp.at<int>(row, col);
    }
  }

  // Streak-based filling from the left
  cv::Mat disp_cpy_l = disp.clone();
  for (int row = 0; row < disp.rows; row++) {
    int fill_val = max_d;
    for (int col = 0; col < disp.cols; col++) {
      int curr_val = disp.at<int>(row, col);
      if (curr_val == -1) {
        disp_cpy_l.at<int>(row, col) = fill_val;
      } else {
        disp_cpy_l.at<int>(row, col) = curr_val;
        fill_val = curr_val;
      }
    }
  }

  // Streak-based filling from the right
  cv::Mat disp_cpy_r = disp.clone();
  for (int row = 0; row < disp.rows; row++) {
    int fill_val = max_d;
    for (int col = disp.cols-1; col >= 0; col--) {
      int curr_val = disp.at<int>(row, col);
      if (curr_val == -1) {
        disp_cpy_r.at<int>(row, col) = fill_val;
      } else {
        disp_cpy_r.at<int>(row, col) = curr_val;
        fill_val = curr_val;
      }
    }
  }

  // Take min of these two disparities
  for (int row = 0; row < disp.rows; row++) {
    for (int col = 0; col < disp.cols; col++) {
      disp.at<int>(row, col) = std::min(disp_cpy_l.at<int>(row, col), disp_cpy_r.at<int>(row, col));
    }
  }
}

// Weighted median filter. Additionally correcting pixels that are invalidated in lr check.
void GuidedImageStereo::wmf(const cv::Mat &disp, cv::Mat &disp_out, const cv::Mat &img,
                            const cv::Mat &mask, int window_size, int min_d, int max_d,
                            float gamma_c, float gamma_p) {
  cv::Mat img_filt;
  cv::medianBlur(img, img_filt, 3);
  cv::Mat filtered = cv::Mat::zeros(disp.rows, disp.cols, CV_32SC1);
  int n = window_size / 2;
  float *weights = new float[max_d-min_d+1];

  for (int row = 0; row < disp.rows; row++) {
    for (int col = 0; col < disp.cols; col++) {
      if (mask.at<uchar>(row, col) == 0) {
        continue;
      }

      std::fill_n(weights, max_d-min_d+1, 0);
      float sum = 0;
      for (int i = std::max(row-n, 0); i <= std::min(row+n, disp.rows-1); i++) {
        for (int j = std::max(col-n, 0); j <= std::min(col+n, disp.cols-1); j++) {
          float sdiff = sqrt((row-i)*(row-i) + (col-j)*(col-j));
          float cdiff = fabsf(img_filt.at<uchar>(row, col) - img_filt.at<uchar>(i, j));
          float weight = exp(- cdiff/(gamma_c*gamma_c) - sdiff/(gamma_p*gamma_p));
          weights[disp.at<int>(i, j)-min_d] += weight;
          sum += weight;
        }
      }

      float cum_sum = 0;
      for (int i = 0; i <= max_d-min_d; i++) {
        cum_sum += weights[i];
        if (cum_sum > sum/2) {
          filtered.at<int>(row, col) = i+min_d;
          break;
        }
      }
    }
  }
  delete[] weights;
  disp_out = filtered;
}

void GuidedImageStereo::post_processing(cv::Mat &disparity_l, const cv::Mat &disparity_r) {
  cv::Mat invalidated;
  lr_check(invalidated, disparity_l, disparity_r);
  fill_invalidated(disparity_l, invalidated, max_disparity);

  cv::Mat filt_disp;
  wmf(disparity_l, filt_disp, left, invalidated, R_MEDIAN, min_disparity, max_disparity, gamma_c, gamma_p);
  filt_disp.copyTo(disparity_l, invalidated);
}

cv::Mat GuidedImageStereo::compute_disparity(const cv::Mat &left, const cv::Mat &right) {
  su::require(left.rows == right.rows && left.cols == right.cols, "Image dimension must match");
  su::require(left.type() == CV_8UC1 && right.type() == CV_8UC1, "Images must be grayscale");
  su::require(left.cols == cols && left.rows == rows, "Images must fit the sizes given in the constructor");
  this->rows = left.rows;
  this->cols = left.cols;

  left.convertTo(this->left, CV_32FC1);
  this->left /= 255.0;
  right.convertTo(this->right, CV_32FC1);
  this->right /= 255.0;

  std::fill_n(cost_volume_l, rows*cols*disparity_range, BORDER_THR);
  std::fill_n(cost_volume_r, rows*cols*disparity_range, BORDER_THR);

  // 1. Matching cost computation
  initial_cost_volume();

  // 2. Cost aggregation
  guided_cost_aggregation();

  // 3. Optimization
  cv::Mat disparity_l, disparity_r;
  su::wta(disparity_l, cost_volume_l, min_disparity, max_disparity, rows, cols);
  su::wta(disparity_r, cost_volume_r, min_disparity, max_disparity, rows, cols);

  // 4. Disparity refinement
  post_processing(disparity_l, disparity_r);

  return disparity_l;
}
