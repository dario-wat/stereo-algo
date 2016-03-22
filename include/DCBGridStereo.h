#ifndef DCBGRID_STEREO_H_
#define DCBGRID_STEREO_H_

#include <opencv2/core/core.hpp>

class DCBGridStereo {
private:
  static constexpr float GAMMA_C = 5.0f;
  static constexpr float GAMMA_P = 17.5f;
  static constexpr float THRESHOLD = 13.3f;
  static const int WINDOW_SIZE = 35;

  int max_disparity;

public:
  inline static float weight( float pxl_p, float pxl_q, float xp, float yp, float xq, float yq,
                              float gamma_c, float gamma_p);
  inline static float tad(float pxl_l, float pxl_r, float threshold);
  float yoon_kweon_pxl(const cv::Mat &left, const cv::Mat &right, int row, int col, int d, int ws, float *table);
  void yoon_kweon(float *cost_volume, const cv::Mat &left, const cv::Mat &right, int ws);
  DCBGridStereo(int max_disparity);
  void compute_disparity(const cv::Mat &left, const cv::Mat &right);
};

#endif
