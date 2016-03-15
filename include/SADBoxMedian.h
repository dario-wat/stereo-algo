#ifndef SAD_BOX_MEDIAN_H_
#define SAD_BOX_MEDIAN_H_

#include <opencv2/core/core.hpp>

class SADBoxMedian {
private:
  int max_disparity;
  int box_size;
  int median_size;
public:
  SADBoxMedian(int max_disparity, int box_size, int median_size);
  cv::Mat compute_disparity(const cv::Mat &left, const cv::Mat &right);
};

#endif