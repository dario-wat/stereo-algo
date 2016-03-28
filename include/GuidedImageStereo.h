#ifndef GUIDED_IMAGE_STEREO_H_
#define GUIDED_IMAGE_STEREO_H_

#include <opencv2/core/core.hpp>

class GuidedImageStereo {
public:
  GuidedImageStereo();
  void compute_disparity(const cv::Mat &left, const cv::Mat &right);
};

#endif
