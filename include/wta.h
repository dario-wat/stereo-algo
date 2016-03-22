#ifndef WTA_H_
#define WTA_H_

#include <opencv2/core/core.hpp>

namespace su {

static const float FL_MAX = 10000.0f;

// Winner-take-all optimization given cost volume. The returning cv::Mat contains
// CV_32SC1 since there is not much time improvement when using unsigned char
void wta(cv::Mat &disparity, const float *cost_volume, int max_d, int rows, int cols);

}

#endif
