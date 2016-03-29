#include <iostream>
#include <cstdio>
#include <iomanip>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <ctime>
#include "FiveRegionStereo.h"
#include "DisparityPropagationStereo.h"
#include "IDRStereo.h"
#include "SADBoxMedian.h"
#include "FeatureLinkStereo.h"
#include "DCBGridStereo.h"
#include "GuidedImageStereo.h"
#include "st_util.h"

using std::cout;
using std::endl;
using std::cerr;

int main(int argc, char **argv) {
    if (argc != 6) {
        cerr    << "Usage: ./program <left_image> <right_image> <max_disparity> <gamma_c> <gamma_p>"
                << endl;
        exit(1);
    }

    cv::Mat img_left_g;
    cv::cvtColor(cv::imread(argv[1]), img_left_g, CV_BGR2GRAY);
    cv::Mat img_right_g;
    cv::cvtColor(cv::imread(argv[2]), img_right_g, CV_BGR2GRAY);

    int max_disp = atoi(argv[3]);
    float gamma_c = atof(argv[4]);
    float gamma_p = atof(argv[5]);

    clock_t begin = clock();
    // SADBoxMedian idr = SADBoxMedian(max_disp, box_size, median_size);
    // cv::Mat disp = idr.compute_disparity(img_left_g, img_right_g);
    // FiveRegionStereo frs = FiveRegionStereo(0, 48, 3, 3, 25, 6, 0.0);
    // cv::Mat disp = frs.compute_disparity(img_left_g, img_right_g);
    // DisparityPropagationStereo dps = DisparityPropagationStereo(max_disp, box_size, median_size);
    // cv::Mat disp = dps.compute_disparity(img_left_g, img_right_g);
    // FeatureLinkStereo fls = FeatureLinkStereo(3, 5.0, 10.0);
    // fls.compute_disparity(img_left_g, img_right_g);
    // DCBGridStereo dcb = DCBGridStereo(max_disp, sigma_s, sigma_r);
    // cv::Mat disp = dcb.compute_disparity(img_left_g, img_right_g);
    GuidedImageStereo gis = GuidedImageStereo(max_disp, gamma_c, gamma_p);
    cv::Mat disp = gis.compute_disparity(img_left_g, img_right_g);

    cerr << "Full time: " << double(clock()-begin) / CLOCKS_PER_SEC << endl;
    // cerr << disp.rows << ' ' << disp.cols << ' ' << disp.channels() << endl;
    // cv::Mat disp_vis;
    // su::convert_to_disparity_visualize(disp, disp_vis, true);
    // cv::imshow("Disparity", disp_vis);
    // cv::waitKey(0);

    // disp.convertTo(disp, CV_16SC1);
    su::print_mat<int>(disp);
    return 0;
}
