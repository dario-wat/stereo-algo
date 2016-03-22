#ifndef DCBGRID_STEREO_H_
#define DCBGRID_STEREO_H_

#include <opencv2/core/core.hpp>

class DCBGridStereo {
public:
  DCBGridStereo();
  void compute_disparity(const cv::Mat &left, const cv::Mat &right);
};

#endif
