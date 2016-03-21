#ifndef FEATURE_LINK_STEREO_H_
#define FEATURE_LINK_STEREO_H_

#include <opencv2/core/core.hpp>

class FeatureLinkStereo {
public:
  FeatureLinkStereo();
  void compute_disparity(const cv::Mat &left, const cv::Mat &right);
};

#endif