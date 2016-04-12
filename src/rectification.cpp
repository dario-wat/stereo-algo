#include "rectification.h"

#include <cmath>
#include <string>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

void su::rectification_maps(const std::string &param_file,
                            cv::Mat &map1_left, cv::Mat &map2_left,
                            cv::Mat &map1_right, cv::Mat &map2_right,
                            float alpha) {
  std::ifstream file(param_file);
  if (!file.good()) {
    throw std::invalid_argument("File '" + param_file + "' does not exist");
  }

  // File format:
  //   right camera focal_length (mm)
  //   right camera position_x position_y position_z (mm)
  //   right camera image_width image_height (pxl)
  //   right camera pixel_width pixel_height (mm)
  //   right camera roll (radians)
  //   right camera pitch (radians)
  //   right camera yaw (radians)

  //   left camera focal_length (mm)
  //   left camera position_x position_y position_z (mm)
  //   left camera image_width image_height (pxl)
  //   left camera pixel_width pixel_height (mm)
  //   left camera roll (radians)
  //   left camera pitch (radians)
  //   left camera yaw (radians)

  // Read right
  double focal_length_right;
  double pos_x_right, pos_y_right, pos_z_right;
  int image_width_right, image_height_right;
  double pixel_width_right, pixel_height_right;
  double roll_right, pitch_right, yaw_right;
  file >> focal_length_right;
  file >> pos_x_right >> pos_y_right >> pos_z_right;
  file >> image_width_right >> image_height_right;
  file >> pixel_width_right >> pixel_height_right;
  file >> roll_right >> pitch_right >> yaw_right;

  // Read left
  double focal_length_left;
  double pos_x_left, pos_y_left, pos_z_left;
  int image_width_left, image_height_left;
  double pixel_width_left, pixel_height_left;
  double roll_left, pitch_left, yaw_left;
  file >> focal_length_left;
  file >> pos_x_left >> pos_y_left >> pos_z_left;
  file >> image_width_left >> image_height_left;
  file >> pixel_width_left >> pixel_height_left;
  file >> roll_left >> pitch_left >> yaw_left;
  file.close();

  // Create right intrinsic
  double fx_r = focal_length_right / pixel_width_right;
  double fy_r = focal_length_right / pixel_height_right;
  double x0_r = image_width_right / 2;
  double y0_r = image_height_right / 2;
  cv::Mat camera_matrix_right = (cv::Mat_<double>(3, 3) <<
      fx_r,    0, x0_r,
         0, fy_r, y0_r,
         0,    0,    1 );

  // Create left intrinsic
  double fx_l = focal_length_left / pixel_width_left;
  double fy_l = focal_length_left / pixel_height_left;
  double x0_l = image_width_left / 2;
  double y0_l = image_height_left / 2;
  cv::Mat camera_matrix_left = (cv::Mat_<double>(3, 3) <<
      fx_l,    0, x0_l,
         0, fy_l, y0_l,
         0,    0,    1 );

  // Assume no distortions
  cv::Mat dist_right = (cv::Mat_<double>(5, 1) << 0., 0., 0., 0., 0.);
  cv::Mat dist_left = (cv::Mat_<double>(5, 1) << 0., 0., 0., 0., 0.);

  // Right rotation matrix
  cv::Mat rotation_pitch_right = (cv::Mat_<double>(3, 3) <<
      1,                0,                 0,
      0, cos(pitch_right), -sin(pitch_right),
      0, sin(pitch_right),  cos(pitch_right) );
  cv::Mat rotation_yaw_right = (cv::Mat_<double>(3, 3) <<
       cos(yaw_right), 0, sin(yaw_right),
                    0, 1,              0,
      -sin(yaw_right), 0, cos(yaw_right));
  cv::Mat rotation_roll_right = (cv::Mat_<double>(3, 3) <<
      cos(roll_right), -sin(roll_right), 0,
      sin(roll_right),  cos(roll_right), 0,
                    0,                0, 1 );
  cv::Mat rotation_right = rotation_pitch_right * rotation_yaw_right * rotation_roll_right;

  // Left rotation matrix
  cv::Mat rotation_pitch_left = (cv::Mat_<double>(3, 3) <<
      1,               0,                0,
      0, cos(pitch_left), -sin(pitch_left),
      0, sin(pitch_left),  cos(pitch_left) );
  cv::Mat rotation_yaw_left = (cv::Mat_<double>(3, 3) <<
       cos(yaw_left), 0, sin(yaw_left),
                   0, 1,             0,
      -sin(yaw_left), 0, cos(yaw_left) );
  cv::Mat rotation_roll_left = (cv::Mat_<double>(3, 3) <<
      cos(roll_left), -sin(roll_left), 0,
      sin(roll_left),  cos(roll_left), 0,
                   0,               0, 1 );
  cv::Mat rotation_left = rotation_pitch_left * rotation_yaw_left * rotation_roll_left;

  // Stereo extrinsic matrices
  cv::Mat translation_right = (cv::Mat_<double>(3, 1) << pos_x_right, pos_y_right, pos_z_right) / 1000.0;
  cv::Mat right_extrinsic = cv::Mat::zeros(4, 4, CV_64FC1);
  rotation_right.copyTo(right_extrinsic(cv::Rect(0, 0, 3, 3)));
  translation_right.copyTo(right_extrinsic(cv::Rect(3, 0, 1, 3)));
  right_extrinsic.at<double>(3, 3) = 1.0;

  cv::Mat translation_left = (cv::Mat_<double>(3, 1) << pos_x_left, pos_y_left, pos_z_left) / 1000.0;
  cv::Mat left_extrinsic = cv::Mat::zeros(4, 4, CV_64FC1);
  rotation_left.copyTo(left_extrinsic(cv::Rect(0, 0, 3, 3)));
  translation_left.copyTo(left_extrinsic(cv::Rect(3, 0, 1, 3)));
  left_extrinsic.at<double>(3, 3) = 1.0;

  // Creating stereo matrices
  cv::Mat stereo_extrinsic = left_extrinsic.inv() * right_extrinsic;
  cv::Mat R = stereo_extrinsic(cv::Rect(0, 0, 3, 3));
  cv::Mat T = stereo_extrinsic(cv::Rect(3, 0, 1, 3));

  // Rectification matrices
  cv::Mat Rl, Rr, Pl, Pr;
  cv::stereoRectify(camera_matrix_left, dist_left,
                    camera_matrix_right, dist_right,
                    cv::Size(image_width_left, image_height_left),
                    R, T, Rl, Rr, Pl, Pr, cv::noArray(),
                    CV_CALIB_ZERO_DISPARITY, alpha);

  cv::initUndistortRectifyMap(camera_matrix_left, dist_left, Rl, Pl,
                              cv::Size(image_width_left, image_height_left), CV_32FC1,
                              map1_left, map2_left);

  cv::initUndistortRectifyMap(camera_matrix_right, dist_right, Rr, Pr,
                              cv::Size(image_width_left, image_height_left), CV_32FC1,
                              map1_right, map2_right);
}
