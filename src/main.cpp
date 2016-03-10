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
    if (argc != 6) {
        cerr    << "Usage: ./program <left_image> <right_image> <max_disparity> <sigma_s> <sigma_r>"
                << endl;
        exit(1);
    }

    cv::Mat img_left_g;
    cv::cvtColor(cv::imread(argv[1]), img_left_g, CV_BGR2GRAY);
    cv::Mat img_right_g;
    cv::cvtColor(cv::imread(argv[2]), img_right_g, CV_BGR2GRAY);

    int max_disp = atoi(argv[3]);
    float sigma_s = atof(argv[4]);
    float sigma_r = atof(argv[5]);

    clock_t begin = clock();
    DisparityPropagationStereo dps = DisparityPropagationStereo(max_disp, sigma_s, sigma_r);
    cv::Mat disparity_map = dps.compute_disparity(img_left_g, img_right_g);
    
    cerr << "Full time: " << double(clock()-begin) / CLOCKS_PER_SEC << endl;
    su::print_mat<short>(disparity_map);
    return 0;
}