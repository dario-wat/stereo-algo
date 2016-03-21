#ifndef FEATURE_LINK_STEREO_H_
#define FEATURE_LINK_STEREO_H_

#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

class FeatureLinkStereo {
private:
  int window_size;

  int image_height;

  // aux arrays for keypoint indices in each row of the images
  int *left_indices;
  int *right_indices;

  // aux keypoint vectors
  std::vector<cv::KeyPoint> keypoints_left;
  std::vector<cv::KeyPoint> keypoints_right;
  std::vector<cv::KeyPoint> init_left_matches;
  std::vector<cv::KeyPoint> init_right_matches;
private:
  void initial_guess(const cv::Mat &left, const cv::Mat &right);
//   inline float mse(const cv::Mat &left, const cv::Mat &right, int x, int y, int d, int n);
public:
  FeatureLinkStereo(int window_size);
  void compute_disparity(const cv::Mat &left, const cv::Mat &right);
};

#endif
