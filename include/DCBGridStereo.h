#ifndef DCBGRID_STEREO_H_
#define DCBGRID_STEREO_H_

#include <opencv2/core/core.hpp>

class DCBGridStereo {
private:
  static constexpr float SIGMA_S = 10.0f;
  static constexpr float SIGMA_R = 10.0f;
  static constexpr float THRESHOLD = 13.3f;

  int max_disparity;
  int sigma_s;
  int sigma_r;
  int threshold;

  float *cost_volume;
  cv::Mat left;
  cv::Mat right;

  // dcb grid stuff
  float *dcb_grid;
  float *dcb_counts;
  int xdim, ydim, ldim, rdim;
  int dref, xref, yref, lref;

public:
  void tad_matching_cost();
  void create_dcb_grid();
  void process_dcb_grid();
  void slice_dcb_grid();

  DCBGridStereo(int max_disparity);
  void compute_disparity(const cv::Mat &left, const cv::Mat &right);
};

#endif
