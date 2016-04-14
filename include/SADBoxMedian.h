#ifndef SAD_BOX_MEDIAN_H_
#define SAD_BOX_MEDIAN_H_

#include <opencv2/core/core.hpp>
#include "AbstractStereoAlgorithm.h"

class SADBoxMedian : public AbstractStereoAlgorithm {
private:
  int box_size;
  int median_size;

public:
  SADBoxMedian( int min_disparity, int max_disparity, int rows, int cols,
                int box_size, int median_size);
  cv::Mat compute_disparity(const cv::Mat &left, const cv::Mat &right);
};

#endif
