#ifndef DCBGRID_STEREO_H_
#define DCBGRID_STEREO_H_

#include <opencv2/core/core.hpp>

class DCBGridStereo {
private:
  // Subsampling
  static constexpr float SIGMA_S = 10;
  static constexpr float SIGMA_R = 10;

  // TAD threshold
  static constexpr float THRESHOLD = 15;

  // Gaussian kernel
  static const int SIZE = 5;
  static constexpr float SIGMA = 1.0;

  // Parameters
  int max_disparity;
  float sigma_s;
  float sigma_r;
  float threshold;

  // aux stuff
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

  DCBGridStereo(int max_disparity, float sigma_s=SIGMA_S, float sigma_r=SIGMA_S, float threshold=THRESHOLD);
  void compute_disparity(const cv::Mat &left, const cv::Mat &right);
};

#endif
