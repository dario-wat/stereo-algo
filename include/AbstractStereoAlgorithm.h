#ifndef ABSTRACT_STEREO_ALGORITHM_H_
#define ABSTRACT_STEREO_ALGORITHM_H_

#include "st_util.h"

class AbstractStereoAlgorithm {
protected:
  static constexpr float FL_L = 10000.0f;

  int min_disparity;
  int max_disparity;
  int disparity_range;

  int rows;
  int cols;

  float *cost_volume;

public:
  AbstractStereoAlgorithm(int min_disparity, int max_disparity, int rows, int cols) {
    su::require(min_disparity < max_disparity, "Min disparity must be lower than max disparity");
    su::require(rows > 0 && cols > 0, "Image dimensions must be > 0");
    this->min_disparity = min_disparity;
    this->max_disparity = max_disparity;
    this->disparity_range = max_disparity - min_disparity;
    this->rows = rows;
    this->cols = cols;
    this->cost_volume = new float[rows*cols*disparity_range];
  }

  ~AbstractStereoAlgorithm() {
    delete[] cost_volume;
  }
};

#endif
