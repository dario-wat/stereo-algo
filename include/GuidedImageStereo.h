#ifndef GUIDED_IMAGE_STEREO_H_
#define GUIDED_IMAGE_STEREO_H_

#include <opencv2/core/core.hpp>

// This is an implementation of
//    "Fast Cost-Volume Filtering for Visual Correspondence and Beyond" - Rhemann, et al.
// It is implemented to work on grayscale images only. The parameters are exactly the same as
// used in the MATLAB implementation provided by authors.
// There might be some bugs that I am not aware of.
class GuidedImageStereo {
private:
  // Cost matching constants
  static constexpr float BORDER_THR = 3. / 255;
  static constexpr float COLOR_THR = 7. / 255;    // tau_1
  static constexpr float GRAD_THR = 2. / 255;     // tau_2
  static constexpr float GAMMA = 0.11;            // 1 - alpha_1

  // Post-processing constants
  static constexpr float GAMMA_C = 0.1;           // sigma_c
  static constexpr float GAMMA_P = 9;             // sigma_s
  static const int R_MEDIAN = 19;

  // Box filtering constants
  static const int R = 9;
  static constexpr float EPS = 0.0001;

  // parameters
  int max_disparity;
  float gamma_c, gamma_p;

  // aux variables
  int cols, rows;

  // aux arrays
  float *cost_volume_l;
  float *cost_volume_r;

  // aux matrices
  cv::Mat left, right;

private:
  static void gradient(const cv::Mat &src, cv::Mat &dst);
  void initial_cost_volume();

  static void guided_filter(const cv::Mat &I, cv::Mat &p, int r);
  void guided_cost_aggregation();

  static void lr_check(cv::Mat &bad, const cv::Mat &disp_left, const cv::Mat &disp_right);
  static void fill_invalidated(cv::Mat &disp, const cv::Mat &invalidated_mask, int max_d);
  static void wmf(const cv::Mat &disp, cv::Mat &disp_out, const cv::Mat &img, const cv::Mat &mask,
                  int window_size, int max_d, float gamma_c, float gamma_p);
  void post_processing(cv::Mat &disparity_l, const cv::Mat &disparity_r);

public:
  GuidedImageStereo(int max_disparity, float gamma_c=GAMMA_C, float gamma_p=GAMMA_P);
  cv::Mat compute_disparity(const cv::Mat &left, const cv::Mat &right);
};

#endif
