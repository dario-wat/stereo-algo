#ifndef IDR_STEREO_H_
#define IDR_STEREO_H_

#include <opencv2/core/core.hpp>

// Iterative Disparity Refinement
class IDRStereo {
private:
  int max_disparity;
  int win_size;
private:
  void cost_aggregation(const cv::Mat &left, const cv::Mat &right, float *cost_volume);
public:
  IDRStereo(int max_disparity, int win_size);
  void compute_disparity(const cv::Mat &left, const cv::Mat &right);
};

#endif