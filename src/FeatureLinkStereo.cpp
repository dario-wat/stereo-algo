#include "FeatureLinkStereo.h"

#include <iostream>
#include <vector>
#include <algorithm>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>

/************* DEBUGGING FUNCTIONS *****************/

void draw_keypoints(cv::Mat &image, const std::vector<cv::KeyPoint> &kps, int size=1) {
  for (auto it = kps.begin(); it != kps.end(); it++) {
    cv::circle(image, it->pt, size, cv::Scalar(0, 0, 255), -1);
  }
}

void print_indices_and_points(const int *indices, const std::vector<cv::KeyPoint> &kps, int height) {
    for (int i = 0; i < height; i++) {
    std::cout << indices[i] << " " << indices[i+1] << std::endl;
    for (int j = indices[i]; j < indices[i+1]; j++) {
      std::cout << kps[j].pt << std::endl;
    }
  }
}

/***************************************************/


// Creates an array in which each element of the array is an index of a set of features in
// the same row in the image, i.e. indices[6] to indices[7] are all features from left to
// right in row 6 in the image
void feature_indices(int *indices, const std::vector<cv::KeyPoint> &kps, int height) {
  std::fill_n(indices, height+1, 0);
  for (auto it = kps.begin(); it != kps.end(); it++) {
    indices[static_cast<int>(it->pt.y)+1]++;
  }
  for (int i = 2; i < height+1; i++) {
    indices[i] += indices[i-1];
  }
}

FeatureLinkStereo::FeatureLinkStereo() {}

void FeatureLinkStereo::compute_disparity(const cv::Mat &left, const cv::Mat &right) {
  std::vector<cv::KeyPoint> keypoints_left, keypoints_right;
  cv::FASTX(left, keypoints_left, 2, true, cv::FastFeatureDetector::TYPE_9_16);
  cv::FASTX(right, keypoints_right, 2, true, cv::FastFeatureDetector::TYPE_9_16);

  
  
  int *left_indices = new int[left.rows+1] ();
  int *right_indices = new int[left.rows+1] ();
  feature_indices(left_indices, keypoints_left, left.rows);
  feature_indices(right_indices, keypoints_right, left.rows);
  // print_indices_and_points(left_indices, keypoints_left, left.rows);
  
  // Drawing stuff for debugging
  cv::Mat draw_left, draw_right;
  cv::cvtColor(left, draw_left, CV_GRAY2BGR);
  draw_keypoints(draw_left, keypoints_left);
  cv::cvtColor(right, draw_right, CV_GRAY2BGR);
  draw_keypoints(draw_right, keypoints_right);
  
  std::cout << keypoints_left.size() << " " << keypoints_right.size() << std::endl;
  cv::imshow("Image left", draw_left);
  cv::imshow("Image right", draw_right);
  cv::waitKey(0);
}
