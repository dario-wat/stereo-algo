#include "FeatureLinkStereo.h"

#include <cmath>
#include <iostream>
#include <vector>
#include <algorithm>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include "st_util.h"

/************* DEBUGGING FUNCTIONS *****************/
// These functions do not perform any checks.

// Makes a deep copy, just to be extra safe
inline cv::Mat convertIfNeeded(const cv::Mat &img) {
  cv::Mat dest;
  if (img.type() != CV_8UC3) {
    cv::cvtColor(img, dest, CV_GRAY2BGR);
  } else {
    dest = img.clone();
  }
  return dest;
}

void draw_keypoints(cv::Mat &image, const cv::Mat &source_img,
                    const std::vector<cv::KeyPoint> &kps, int size=1) {
  image = convertIfNeeded(source_img);
  for (auto it = kps.begin(); it != kps.end(); it++) {
    cv::circle(image, it->pt, size, cv::Scalar(0, 0, 255), -1);
  }
}

void draw_matches(cv::Mat &full_img, const cv::Mat &left, const cv::Mat &right,
                  const std::vector<cv::KeyPoint> kp_left, const std::vector<cv::KeyPoint> &kp_right,
                  int size=1, bool draw_lines=false) {
  full_img = cv::Mat(left.rows, 2*left.cols, CV_8UC3);
  cv::Mat left3 = convertIfNeeded(left);
  cv::Mat right3 = convertIfNeeded(right);
  left3.copyTo(full_img(cv::Rect(0, 0, left.cols, left.rows)));
  right3.copyTo(full_img(cv::Rect(left.cols, 0, left.cols, left.rows)));

  for (int i = 0; i < kp_left.size(); i++) {
    cv::Scalar color(rand()&255, rand()&255, rand()&255);
    cv::circle(full_img, kp_left[i].pt, size, color);
    cv::circle(full_img, kp_right[i].pt + cv::Point2f(left.cols, 0), size, color);
    if (draw_lines) {
      cv::line(full_img, kp_left[i].pt, kp_right[i].pt + cv::Point2f(left.cols, 0), color);
    }
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

FeatureLinkStereo::FeatureLinkStereo(int window_size) {
  su::require(window_size > 0 && window_size % 2, "Window size must be odd and positive");
  this->window_size = window_size;
}

inline float mse(const cv::Mat &left, const cv::Mat &right, int xl, int yl, int xr, int yr, int n) {
  int s = 0;
  for (int dy = -n/2; dy <= n/2; dy++) {
    for (int dx = -n/2; dx <= n/2; dx++) {
      int dist = right.at<uchar>(yr+dy, xr+dx) - left.at<uchar>(yl+dy, xl+dx);
      s += dist * dist;
    }
  }
  return static_cast<float>(s) / (n*n);
}

void FeatureLinkStereo::initial_guess(const cv::Mat &left, const cv::Mat &right) {
  for (int row = 0; row < image_height; row++) {
    for (int i = right_indices[row]; i < right_indices[row+1]; i++) {
      int xr = round(keypoints_right[i].pt.x);
      int yr = round(keypoints_right[i].pt.y);
      for (int j = left_indices[row]; j < left_indices[row+1]; j++) {
        int xl = round(keypoints_left[j].pt.x);
        int yl = round(keypoints_left[j].pt.y);
        float v = mse(left, right, xl, yl, xr, yr, window_size);
        if (v < 5) {
          init_left_matches.push_back(keypoints_left[j]);
          init_right_matches.push_back(keypoints_right[i]);
          goto back_to_reality_goto_lol_I_am_actually_using_GOTO_look_at_me;
        }
        std::cout << v << std::endl;
      }
    }
    back_to_reality_goto_lol_I_am_actually_using_GOTO_look_at_me:;
  }
}

// inline float FeatureLinkStereo::mse(  const cv::Mat &left, const cv::Mat &right,
//                                       int x, int y, int d, int n) {
//   int s = 0;
//   for (int i = y - n/2; i < y + n/2; i++) {
//     for (int j = x - n/2; j < x + n/2; j++) {
//       int dist = right.at<uchar>(i, j) - left.at<uchar>(i, j+d);
//       s += dist * dist;
//     }
//   }
//   return static_cast<float>(s) / (n*n);
// }


void FeatureLinkStereo::compute_disparity(const cv::Mat &left, const cv::Mat &right) {
  this->image_height = left.rows;
  cv::FASTX(left, keypoints_left, 2, true, cv::FastFeatureDetector::TYPE_9_16);
  cv::FASTX(right, keypoints_right, 2, true, cv::FastFeatureDetector::TYPE_9_16);

  left_indices = new int[left.rows+1] ();
  right_indices = new int[left.rows+1] ();
  feature_indices(left_indices, keypoints_left, left.rows);
  feature_indices(right_indices, keypoints_right, left.rows);

  initial_guess(left, right);

  // Drawing stuff for debugging
  cv::Mat draw_left, draw_right, full_img;
  draw_keypoints(draw_left, left, init_left_matches);
  draw_keypoints(draw_right, right, init_right_matches);
  draw_matches(full_img, left, right, init_left_matches, init_right_matches);
  
  std::cout << keypoints_left.size() << " " << keypoints_right.size() << std::endl;
  cv::imshow("Full", full_img);
  // cv::imshow("Image left", draw_left);
  // cv::imshow("Image right", draw_right);
  cv::waitKey(0);
  delete[] left_indices;
  delete[] right_indices;
}
