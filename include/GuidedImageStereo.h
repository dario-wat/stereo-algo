#ifndef GUIDED_IMAGE_STEREO_H_
#define GUIDED_IMAGE_STEREO_H_

#include <opencv2/core/core.hpp>

class GuidedImageStereo {
private:
  static constexpr float BORDER_THR = 3. / 255;
  static constexpr float COLOR_THR = 7. / 255;    // tau_1
  static constexpr float GRAD_THR = 2. / 255;     // tau_2
  static constexpr float GAMMA = 0.11;            // 1 - alpha_1
  static constexpr float GAMMA_C = 0.1;           // sigma_c
  static constexpr float GAMMA_D = 9;             // sigma_s

  int max_disparity;

  int cols, rows;

  // aux arrays
  float *cost_volume_l;
  float *cost_volume_r;

  // aux matrices
  cv::Mat left, right;
  cv::Mat dx_left, dx_right;
private:
  void initial_cost_volume();
public:
  GuidedImageStereo(int max_disparity);
  void compute_disparity(const cv::Mat &left, const cv::Mat &right);
};

#endif
