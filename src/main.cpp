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

    // cv::Mat img_left_g;
    // cv::cvtColor(cv::imread(argv[1]), img_left_g, CV_BGR2GRAY);
    // cv::Mat img_right_g;
    // cv::cvtColor(cv::imread(argv[2]), img_right_g, CV_BGR2GRAY);

    // cv::Mat map1_left, map2_left, map1_right, map2_right;
    // su::rectification_maps("camera_params.txt", map1_left, map2_left, map1_right, map2_right);

    // cv::Mat cleft, cright;
    // cv::cvtColor(img_left_g, cleft, CV_GRAY2BGR);
    // cv::cvtColor(img_right_g, cright, CV_GRAY2BGR);
    // su::draw_horiz_lines(cleft, 50, 3);
    // su::draw_horiz_lines(cright, 50, 3);

    // cv::remap(img_left_g, img_left_g, map1_left, map2_left, CV_INTER_LINEAR);
    // cv::remap(img_right_g, img_right_g, map1_right, map2_right, CV_INTER_LINEAR);

    // cv::remap(cleft, cleft, map1_left, map2_left, CV_INTER_LINEAR);
    // cv::remap(cright, cright, map1_right, map2_right, CV_INTER_LINEAR);
    // cv::resize(cleft, cleft, cv::Size(), 0.25, 0.25);
    // cv::resize(cright, cright, cv::Size(), 0.25, 0.25);
    // cv::imshow("cleft", cleft);
    // cv::imshow("cright", cright);

    // double scale = 0.25;
    // cv::resize(img_left_g, img_left_g, cv::Size(), scale, scale);
    // cv::resize(img_right_g, img_right_g, cv::Size(), scale, scale);

    // int min_d = 112;
    // int max_d = 176;
    // int rows = 512;
    // int cols = 648;

    // int min_d = 120*4;
    // int max_d = 168*4;
    // int rows = 2048;
    // int cols = 2592;

    int min_d = 56;
    int max_d = 88;
    int rows = 256;
    int cols = 324;

    // int min_d = 0;
    // int max_d = 88;

    // int max_disp = atoi(argv[3]);
    // float gamma_c = atof(argv[4]);
    // float gamma_p = atof(argv[5]);
    std::string left_dir = argv[1];
    std::string right_dir = argv[2];

    int N = 493;

    clock_t begin = clock();
    
    std::vector<int> counts(max_d-min_d, 0);
    cv::StereoSGBM sgbm(min_d, max_d-min_d, 15);
    SADBoxMedian idr = SADBoxMedian(min_d, max_d, rows, cols, 11, 3);
    for (int i = 1; i <= N; i++) {
        clock_t ibeg = clock();
        std::string idx = su::str(i);
        cv::Mat left = cv::imread(  left_dir + "/left_0" + std::string(3-idx.size(), '0') + idx + ".pgm",
                                    CV_LOAD_IMAGE_GRAYSCALE);
        cv::Mat right = cv::imread( right_dir + "/right_0" + std::string(3-idx.size(), '0') + idx + ".pgm",
                                    CV_LOAD_IMAGE_GRAYSCALE);

        cv::Mat disp = idr.compute_disparity(left, right);
        // DCBGridStereo dcb = DCBGridStereo(max_d, 10, 10);
        // cv::Mat disp = dcb.compute_disparity(left, right);

        // GuidedImageStereo gis = GuidedImageStereo(174, 10, 10);
        // cv::Mat disp = gis.compute_disparity(left, right);
        // FiveRegionStereo frs = FiveRegionStereo(0, 170, 3, 3, 25, 6, 0.0);
        // cv::Mat disp = frs.compute_disparity(left, right);

        // cv::Mat disp;
        // sgbm(left, right, disp);
        // disp /= 16;
        // disp += 1;

        // cv::medianBlur(disp, disp, 5);

        // std::cout << disp << std::endl;
        
        cerr << "time: " << double(clock()-ibeg) / CLOCKS_PER_SEC << endl;


        // cv::Mat dispint;
        // disp.convertTo(dispint, CV_32SC1);
        // su::count_disparities(dispint, counts, min_d, max_d);
        // su::print_counts(counts, min_d, max_d);

        cv::Mat disp_vis;
        su::convert_to_disparity_visualize(disp, disp_vis, min_d, max_d, true);
        // cerr << "time: " << double(clock()-ibeg) / CLOCKS_PER_SEC << endl;
        cv::imshow("Disparity", disp_vis);
        // cv::imshow("Left", left);
        // cv::imwrite("large_disp/disp_0" + std::string(3-idx.size(), '0') + idx + ".pgm", disp_vis);
        cv::waitKey(1);
    }
    
    
    
    
    // DisparityPropagationStereo dps = DisparityPropagationStereo(256, 1, 1);
    // cv::Mat disp = dps.compute_disparity(img_left_g, img_right_g);
    // FeatureLinkStereo fls = FeatureLinkStereo(3, 5.0, 10.0);
    // fls.compute_disparity(img_left_g, img_right_g);
    

  

    cerr << "Full time: " << double(clock()-begin) / CLOCKS_PER_SEC << endl;


    // cout << disp << endl;
    // disp.convertTo(disp_vis, CV_8UC1);
    
    // cv::imshow("Left", img_left_g);
    // cv::imshow("Right", img_right_g);

    // cv::imwrite("disp.png", disp_vis);
    

    // disp.convertTo(disp, CV_16SC1);
    // su::print_mat<int>(disp);
    return 0;
}
