#include <iostream>
#include <cstdio>
#include <iomanip>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <ctime>
#include "FiveRegionStereo.h"
#include "DisparityPropagationStereo.h"
#include "IDRStereo.h"
#include "SADBoxMedian.h"
#include "FeatureLinkStereo.h"
#include "DCBGridStereo.h"
#include "GuidedImageStereo.h"
#include "st_util.h"
#include "rectification.h"

#include "boost/filesystem/operations.hpp"
#include "boost/filesystem/path.hpp"

namespace fs = boost::filesystem;

using std::cout;
using std::endl;
using std::cerr;

void rectify(const std::string &left_dir, const std::string &right_dir) {
    fs::path path_left = fs::system_complete(fs::path(left_dir));
    fs::path path_right = fs::system_complete(fs::path(right_dir));

    cv::Mat map1_left, map2_left, map1_right, map2_right;
    su::rectification_maps("camera_params.txt", map1_left, map2_left, map1_right, map2_right);

    fs::directory_iterator end_iter;
    for (fs::directory_iterator dir_itr(path_left); dir_itr != end_iter; dir_itr++) {
        std::string filename = dir_itr->path().filename().string();
        cv::Mat img_left = cv::imread(path_left.string() + "/" + filename, CV_LOAD_IMAGE_GRAYSCALE);
        cv::remap(img_left, img_left, map1_left, map2_left, CV_INTER_LINEAR);

        cv::Mat scale_025, scale_0125;
        cv::resize(img_left, scale_025, cv::Size(), 0.25, 0.25);
        cv::resize(img_left, scale_0125, cv::Size(), 0.125, 0.125);

        cv::imwrite("rect_left_0125/" + filename, scale_0125);
        cv::imwrite("rect_left_025/" + filename, scale_025);
        cv::imwrite("rect_left/" + filename, img_left);
    }

    for (fs::directory_iterator dir_itr(path_right); dir_itr != end_iter; dir_itr++) {
        std::string filename = dir_itr->path().filename().string();
        cv::Mat img_right = cv::imread(path_right.string() + "/" + filename, CV_LOAD_IMAGE_GRAYSCALE);
        cv::remap(img_right, img_right, map1_right, map2_right, CV_INTER_LINEAR);

        cv::Mat scale_025, scale_0125;
        cv::resize(img_right, scale_025, cv::Size(), 0.25, 0.25);
        cv::resize(img_right, scale_0125, cv::Size(), 0.125, 0.125);

        cv::imwrite("rect_right_0125/" + filename, scale_0125);
        cv::imwrite("rect_right_025/" + filename, scale_025);
        cv::imwrite("rect_right/" + filename, img_right);
    }
}

int main(int argc, char **argv) {
    // if (argc != 6) {
    //     cerr    << "Usage: ./program <left_image> <right_image> <max_disparity> <gamma_c> <gamma_p>"
    //             << endl;
    //     exit(1);
    // }

    cv::Mat img_left_g;
    cv::cvtColor(cv::imread(argv[1]), img_left_g, CV_BGR2GRAY);
    cv::Mat img_right_g;
    cv::cvtColor(cv::imread(argv[2]), img_right_g, CV_BGR2GRAY);

    cv::Mat map1_left, map2_left, map1_right, map2_right;
    su::rectification_maps("camera_params.txt", map1_left, map2_left, map1_right, map2_right);

    cv::Mat cleft, cright;
    cv::cvtColor(img_left_g, cleft, CV_GRAY2BGR);
    cv::cvtColor(img_right_g, cright, CV_GRAY2BGR);
    su::draw_horiz_lines(cleft, 50, 3);
    su::draw_horiz_lines(cright, 50, 3);

    cv::remap(img_left_g, img_left_g, map1_left, map2_left, CV_INTER_LINEAR);
    cv::remap(img_right_g, img_right_g, map1_right, map2_right, CV_INTER_LINEAR);

    cv::remap(cleft, cleft, map1_left, map2_left, CV_INTER_LINEAR);
    cv::remap(cright, cright, map1_right, map2_right, CV_INTER_LINEAR);
    cv::resize(cleft, cleft, cv::Size(), 0.25, 0.25);
    cv::resize(cright, cright, cv::Size(), 0.25, 0.25);
    cv::imshow("cleft", cleft);
    cv::imshow("cright", cright);

    double scale = 0.25;
    cv::resize(img_left_g, img_left_g, cv::Size(), scale, scale);
    cv::resize(img_right_g, img_right_g, cv::Size(), scale, scale);

    // int max_disp = atoi(argv[3]);
    // float gamma_c = atof(argv[4]);
    // float gamma_p = atof(argv[5]);

    clock_t begin = clock();
    SADBoxMedian idr = SADBoxMedian(400, 25, 5);
    cv::Mat disp = idr.compute_disparity(img_left_g, img_right_g);
    // FiveRegionStereo frs = FiveRegionStereo(0, 255, 3, 3, 25, 6, 0.0);
    // cv::Mat disp = frs.compute_disparity(img_left_g, img_right_g);
    // DisparityPropagationStereo dps = DisparityPropagationStereo(256, 1, 1);
    // cv::Mat disp = dps.compute_disparity(img_left_g, img_right_g);
    // FeatureLinkStereo fls = FeatureLinkStereo(3, 5.0, 10.0);
    // fls.compute_disparity(img_left_g, img_right_g);
    // DCBGridStereo dcb = DCBGridStereo(256, 10, 10);
    // cv::Mat disp = dcb.compute_disparity(img_left_g, img_right_g);
    // GuidedImageStereo gis = GuidedImageStereo(256, 10, 10);
    // cv::Mat disp = gis.compute_disparity(img_left_g, img_right_g);

    // cv::Mat disp;
    // cv::StereoSGBM sgbm(0, 256, 21);
    // sgbm(img_left_g, img_right_g, disp);

    cerr << "Full time: " << double(clock()-begin) / CLOCKS_PER_SEC << endl;

    cv::Mat disp_vis;
    su::convert_to_disparity_visualize(disp, disp_vis, 30, 200, true);
    // cout << disp << endl;
    // disp.convertTo(disp_vis, CV_8UC1);
    cv::imshow("Disparity", disp_vis);
    // cv::imshow("Left", img_left_g);
    // cv::imshow("Right", img_right_g);

    // cv::imwrite("disp.png", disp_vis);
    cv::waitKey(0);

    // disp.convertTo(disp, CV_16SC1);
    // su::print_mat<int>(disp);
    return 0;
}
