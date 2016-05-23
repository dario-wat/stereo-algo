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

        std::string tmpname = dir_itr->path().stem().string();
        std::string filename_out = "left" + tmpname.substr(tmpname.size()-5) + ".pgm";
        cv::imwrite("rect_left_0125/" + filename_out, scale_0125);
        cv::imwrite("rect_left_025/" + filename_out, scale_025);
        cv::imwrite("rect_left/" + filename_out, img_left);
    }

    for (fs::directory_iterator dir_itr(path_right); dir_itr != end_iter; dir_itr++) {
        std::string filename = dir_itr->path().filename().string();
        cv::Mat img_right = cv::imread(path_right.string() + "/" + filename, CV_LOAD_IMAGE_GRAYSCALE);
        cv::remap(img_right, img_right, map1_right, map2_right, CV_INTER_LINEAR);

        cv::Mat scale_025, scale_0125;
        cv::resize(img_right, scale_025, cv::Size(), 0.25, 0.25);
        cv::resize(img_right, scale_0125, cv::Size(), 0.125, 0.125);

        std::string tmpname = dir_itr->path().stem().string();
        std::string filename_out = "right" + tmpname.substr(tmpname.size()-5) + ".pgm";
        cv::imwrite("rect_right_0125/" + filename_out, scale_0125);
        cv::imwrite("rect_right_025/" + filename_out, scale_025);
        cv::imwrite("rect_right/" + filename_out, img_right);
    }
}

void rectify_one(const std::string &left, const std::string &right) {
    cv::Mat img_left_g;
    cv::cvtColor(cv::imread(left), img_left_g, CV_BGR2GRAY);
    cv::Mat img_right_g;
    cv::cvtColor(cv::imread(right), img_right_g, CV_BGR2GRAY);

    cv::Mat map1_left, map2_left, map1_right, map2_right;
    su::rectification_maps("camera_params.txt", map1_left, map2_left, map1_right, map2_right);

    cv::Mat cleft, cright;
    cv::cvtColor(img_left_g, cleft, CV_GRAY2BGR);
    cv::cvtColor(img_right_g, cright, CV_GRAY2BGR);

    cv::remap(img_left_g, img_left_g, map1_left, map2_left, CV_INTER_LINEAR);
    cv::remap(img_right_g, img_right_g, map1_right, map2_right, CV_INTER_LINEAR);

    cv::Mat mapx = cv::Mat::zeros(cleft.rows, cleft.cols, CV_32FC1);
    cv::Mat mapy = cv::Mat::zeros(cleft.rows, cleft.cols, CV_32FC1);

    for (int i = 0; i < cleft.rows; i++) {
        for (int j = 0; j < cleft.cols; j++) {
            float x = map1_left.at<float>(i,j);
            float y = map2_left.at<float>(i,j);
            if (x >= 0 && y >= 0 && x < cleft.cols && y < cleft.rows) {
                mapx.at<float>(y,x) = j;
                mapy.at<float>(y,x) = i;
            }
        }
    }

    cv::Mat mapxr = cv::Mat::zeros(cleft.rows, cleft.cols, CV_32FC1);
    cv::Mat mapyr = cv::Mat::zeros(cleft.rows, cleft.cols, CV_32FC1);
    for (int i = 0; i < cleft.rows; i++) {
        for (int j = 0; j < cleft.cols; j++) {
            float x = map1_right.at<float>(i,j);
            float y = map2_right.at<float>(i,j);
            if (x >= 0 && y >= 0 && x < cleft.cols && y < cleft.rows) {
                mapxr.at<float>(y,x) = j;
                mapyr.at<float>(y,x) = i;
            }
        }
    }

    cv::Scalar color = cv::Scalar(0, 0, 255);

    cv::remap(cleft, cleft, map1_left, map2_left, CV_INTER_LINEAR);
    cv::remap(cright, cright, map1_right, map2_right, CV_INTER_LINEAR);
    su::draw_horiz_lines(cleft, 200, 12, color);
    su::draw_horiz_lines(cright, 200, 12, color);

    cv::Mat combined1 = cv::Mat::zeros(cleft.rows, cleft.cols + cright.cols + 40, CV_8UC3);
    combined1 = cv::Scalar(255, 255, 255);
    cleft.copyTo(combined1(cv::Rect(0, 0, cleft.cols, cleft.rows)));
    cright.copyTo(combined1(cv::Rect(cleft.cols+40, 0, cleft.cols, cleft.rows)));
    su::draw_horiz_lines(combined1, 200, 12, color);

    cv::Mat combined = cv::Mat::zeros(cleft.rows + cleft.rows + 80, cleft.cols + cright.cols + 40, CV_8UC3);
    combined = cv::Scalar(255, 255, 255);
    combined1.copyTo(combined(cv::Rect(0, cleft.rows+80, cleft.cols + cleft.cols + 40, cleft.rows)));    

    cv::Mat inverted, invertedr;
    cv::remap(cleft, inverted, mapx, mapy, CV_INTER_LINEAR);
    cv::remap(cright, invertedr, mapxr, mapyr, CV_INTER_LINEAR);

    inverted.copyTo(combined(cv::Rect(0, 0, cleft.cols, cleft.rows)));
    invertedr.copyTo(combined(cv::Rect(cleft.cols+40, 0, cleft.cols, cleft.rows)));

    cv::resize(inverted, inverted, cv::Size(), 0.25, 0.25);
    cv::resize(invertedr, invertedr, cv::Size(), 0.25, 0.25);

    cv::resize(cleft, cleft, cv::Size(), 0.25, 0.25);
    cv::resize(cright, cright, cv::Size(), 0.25, 0.25);
    cv::resize(combined, combined, cv::Size(), 0.25, 0.25);

    cv::imwrite("combined.png", combined);
    // cv::imshow("combined", combined);
    // cv::imwrite("invleft.png", inverted);
    // cv::imwrite("invright.png", invertedr);
    // cv::waitKey(0);
}

void stream_eval(const std::string &left_dir, const std::string &right_dir) {
    const int N = 493;
    int min_d =  56, max_d =  88, rows =  256, cols =  324;         // Small images
    // int min_d = 112, max_d = 176, rows =  512, cols =  648;         // Medium images
    // int min_d = 448, max_d = 704, rows = 2048, cols = 2592;         // Large images

    clock_t begin = clock();

    // std::vector<int> counts(max_d-min_d, 0);
    // cv::StereoSGBM sgbm(min_d, max_d-min_d, 25);
    // SADBoxMedian idr = SADBoxMedian(min_d, max_d, rows, cols, 51, 5);
    DCBGridStereo dcb = DCBGridStereo(min_d, max_d, rows, cols, 6, 16);
    // GuidedImageStereo gis = GuidedImageStereo(min_d, max_d, rows, cols, 2, 2);
    for (int i = 1; i <= N; i++) {
        clock_t ibeg = clock();
        std::string idx = su::str(i);
        cv::Mat left = cv::imread(  left_dir + "/left_0" + std::string(3-idx.size(), '0') + idx + ".pgm",
                                    CV_LOAD_IMAGE_GRAYSCALE);
        cv::Mat right = cv::imread( right_dir + "/right_0" + std::string(3-idx.size(), '0') + idx + ".pgm",
                                    CV_LOAD_IMAGE_GRAYSCALE);

        // cv::Mat disp = idr.compute_disparity(left, right);
        cv::Mat disp = dcb.compute_disparity(left, right);
        // cv::Mat disp = gis.compute_disparity(left, right);

        // cv::Mat disp;
        // sgbm(left, right, disp);
        // disp /= 16;
        // disp += 1;

        cerr << i << " time: " << double(clock()-ibeg) / CLOCKS_PER_SEC << endl;

        // cv::Mat dispint;
        // disp.convertTo(dispint, CV_32SC1);
        // su::count_disparities(dispint, counts, min_d, max_d);
        // su::print_counts(counts, min_d, max_d);

        cv::Mat disp_vis;
        su::convert_to_disparity_visualize(disp, disp_vis, min_d, max_d, true);
        // cv::imshow("Disparity", disp_vis);
        // cv::imshow("Left", left);
        cv::imwrite("dispmaps/disp_0" + std::string(3-idx.size(), '0') + idx + ".pgm", disp_vis);
        // cv::waitKey(1);
    }
    cerr << "Full time: " << double(clock()-begin) / CLOCKS_PER_SEC << endl;
}

void eval_one(const std::string &left_f, const std::string &right_f, int min_d, int max_d) {
    cv::Mat left = cv::imread(left_f, CV_LOAD_IMAGE_GRAYSCALE);
    cv::Mat right = cv::imread(right_f, CV_LOAD_IMAGE_GRAYSCALE);

    cv::StereoSGBM sgbm(min_d, max_d-min_d, 21);
    SADBoxMedian sbm = SADBoxMedian(min_d, max_d, left.rows, left.cols, 21, 5);
    DCBGridStereo dcb = DCBGridStereo(min_d, max_d, left.rows, left.cols, 10, 10);
    cv::Mat disp = dcb.compute_disparity(left, right);
    // sgbm(left, right, disp);
    // disp /= 16;
    // disp += 1;
    // cout << disp << endl;
    // disp.convertTo(disp, CV_8UC1);
    cv::Mat disp_vis;
    su::convert_to_disparity_visualize(disp, disp_vis, min_d, max_d, true);
    cv::imshow("Disparity", disp_vis);
    cv::waitKey(0);
}

int main(int argc, char **argv) {
    // if (argc != 6) {
    //     cerr    << "Usage: ./program <left_image> <right_image> <max_disparity> <gamma_c> <gamma_p>"
    //             << endl;
    //     exit(1);
    // }

    rectify_one(argv[1], argv[2]);
    // double scale = 0.25;
    // cv::resize(img_left_g, img_left_g, cv::Size(), scale, scale);
    // cv::resize(img_right_g, img_right_g, cv::Size(), scale, scale);

    // rectify(argv[1], argv[2]);
    // eval_one(argv[1], argv[2], 0, 256);
    // stream_eval(argv[1], argv[2]);

    // cout << disp << endl;
    // disp.convertTo(disp_vis, CV_8UC1);
    
    // cv::imshow("Left", img_left_g);
    // cv::imshow("Right", img_right_g);

    // cv::imwrite("disp.png", disp_vis);

    // disp.convertTo(disp, CV_16SC1);
    // su::print_mat<int>(disp);
    return 0;
}
