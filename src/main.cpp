#include <iostream>
#include <cstdio>
#include <iomanip>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <ctime>
#include "FiveRegionStereo.h"
#include "DisparityPropagationStereo.h"
#include "st_util.h"

using std::cout;
using std::endl;
using std::cerr;

int main(int argc, char **argv) {
    if (argc != 5) {
        cerr << "Usage: ./program <left_image> <right_image> <max_disparity> <region_size>" << endl;
        exit(1);
    }

    cv::Mat img_left_g;
    cv::cvtColor(cv::imread(argv[1]), img_left_g, CV_BGR2GRAY);
    cv::Mat img_right_g;
    cv::cvtColor(cv::imread(argv[2]), img_right_g, CV_BGR2GRAY);
    int max_disp = atoi(argv[3]);
    int region_size = atoi(argv[4]);

    clock_t begin = clock();
    // FiveRegionStereo frs = FiveRegionStereo(0, max_disp, region_size, region_size, 25, 6, 0.0);
    // cv::Mat disparity = frs.compute_disparity(img_left_g, img_right_g);
    DisparityPropagationStereo dps = DisparityPropagationStereo(max_disp);
    dps.compute_disparity(img_left_g, img_right_g);
    
    cerr << "Full time: " << double(clock()-begin) / CLOCKS_PER_SEC << endl;
    // su::print_mat_float(disparity);
    // su::convert_to_disparity_visualize(disparity, disparity);
    // cv::imshow("Disparity", disparity);
    // cv::waitKey(0);
    return 0;
}