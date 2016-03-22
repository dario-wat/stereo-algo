#ifndef FEATURE_LINK_STEREO_H_
#define FEATURE_LINK_STEREO_H_

#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

// NOTE discontinued because of the poor description of the algorithms in the original paper.
// I just don't know how to reproduce this.
// Original paper: "Fast Stereo Matching of Feature Links" - Chang-Il Kim, Soon-Yong Park

class FeatureLinkStereo {
private:
  int window_size;
  float init_guess_threshold;
  float threshold;

  int image_height;

  // aux arrays for keypoint indices in each row of the images
  int *left_indices;
  int *right_indices;

  // aux keypoint vectors
  std::vector<cv::KeyPoint> keypoints_left;
  std::vector<cv::KeyPoint> keypoints_right;
  std::vector<int> init_left_matches;
  std::vector<int> init_right_matches;
private:
  void initial_guess(const cv::Mat &left, const cv::Mat &right);
  void feature_link(const cv::Mat &left, const cv::Mat &right, int row);

  // Mean Square Error
  static inline float mse(const cv::Mat &left, const cv::Mat &right,
                          int xl, int yl, int xr, int yr, int n);
  static inline float mse(const cv::Mat &left, const cv::Mat &right,
                          const cv::KeyPoint &kpl, const cv::KeyPoint &kpr, int n);
public:
  FeatureLinkStereo(int window_size, float init_guess_threshold, float threshold);
  void compute_disparity(const cv::Mat &left, const cv::Mat &right);
};

#endif
