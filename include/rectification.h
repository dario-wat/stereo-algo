#ifndef RECTIFICATION_H_
#define RECTIFICATION_H_

#include <string>
#include <opencv2/core/core.hpp>

namespace su {
  void rectification_maps(const std::string &param_file,
                          cv::Mat &map1_left, cv::Mat &map2_left,
                          cv::Mat &map1_right, cv::Mat &map2_right,
                          float alpha = -1);
}

#endif
