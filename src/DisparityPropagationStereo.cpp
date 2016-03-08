#include "DisparityPropagationStereo.h"

#include <ctime>
#include <iostream>
#include <algorithm>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "st_util.h"

/**** JUST FOR DEBUGGING, WILL BE REMOVED ****/
#define CLOCKS clock_t start = clock();
#define CLOCKP std::cout << (clock()-start) / float(CLOCKS_PER_SEC) << std::endl;
using std::cout;
using std::cerr;
using std::endl;
/************************/

static const float LAMBDA = 0.5f;

DisparityPropagationStereo::DisparityPropagationStereo(int max_disparity) {
    su::require(max_disparity > 0, "Max disparity must be positive");
    this->max_disparity = max_disparity;
}

//TODO lambda to float
inline void DisparityPropagationStereo::tad(const short *dx_left, const uchar *left,
                                            const short *dx_right, const uchar *right, float *cost, 
                                            int width, int d, int tg, uchar tc, double lambda) {
    // TODO vectorize this
    for (int col = d; col < width; col++) {
        // cout << dx_left[col] << endl;
        cost[col] = lambda * std::min<int>(std::abs((int)left[col] - right[col-d]), tc) +
            (1 - lambda) * std::min<int>(std::abs(dx_left[col] - dx_right[col-d]), tg);
    }
}

inline void DisparityPropagationStereo::tad_r(  const short *dx_left, const uchar *left,
                                                const short *dx_right, const uchar *right, float *cost, 
                                            int width, int d, int tg, uchar tc, double lambda) {
    // TODO vectorize this
    for (int col = 0; col < width-d; col++) {
        // cout << dx_left[col+d] - dx_right[col] << endl;
        cost[col] = lambda * std::min<int>(std::abs((int)left[col+d] - right[col]), tc) +
            (1 - lambda) * std::min<int>(std::abs(dx_left[col+d] - dx_right[col]), tg);
    }
}

cv::Mat lr_check(const cv::Mat &disparity_left, const cv::Mat &disparity_right) {
    cv::Mat stable = cv::Mat(disparity_left.rows, disparity_left.cols, CV_8UC1);
    cout << disparity_left.cols << endl;
    for (int row = 0; row < disparity_left.rows; row++) {
        for (int col = 0; col < disparity_left.cols; col++) {
            short d = disparity_left.at<short>(row, col);
            // cout << d << ' ';
            if (col - (int)d < 0 || d != disparity_right.at<short>(row, col-d)) {
                // cout << col - (int)d << endl;
                stable.at<uchar>(row, col) = 0;
            } else {
                stable.at<uchar>(row, col) = 255;
            }
        }
    }
    return stable;
}

// TODO this needs love
void DisparityPropagationStereo::preprocess() {
    cv::Mat dx_left, dx_right;
    cv::Scharr(left, dx_left, CV_16SC1, 1, 0);
    cv::Scharr(right, dx_right, CV_16SC1, 1, 0);
    // cout << dx_left << endl;

    // Cost volume has (disparity, height, width) ordering so that it matches Mat and can be
    // used for box filtering
    float *cost_volume = new float[max_disparity*left.cols*left.rows];
    std::fill_n(cost_volume, max_disparity*left.cols*left.rows, 100000.0);  // TODO
    for (int d = 0; d < max_disparity; d++) {
        float *disparity_level = cost_volume + d*left.cols*left.rows;
        for (int row = 0; row < left.rows; row++) {
            // TODO all this shit into a function
            tad(    dx_left.ptr<short>(row), left.ptr<uchar>(row), dx_right.ptr<short>(row),
                    right.ptr<uchar>(row), disparity_level + row*left.cols, left.cols, d,
                    100, 100, LAMBDA);
        }
        cv::Mat cost_slice = cv::Mat(left.rows, left.cols, CV_32FC1, disparity_level);
        // TODO might be faster if I implement it
        cv::boxFilter(cost_slice, cost_slice, -1, cv::Size(5, 5));
    }

    //TODO this can be done with extra memory to achieve cache locality
    cv::Mat disparity_left = cv::Mat(left.rows, left.cols, CV_16SC1);
    for (int row = 0; row < left.rows; row++) {
        for (int col = 0; col < left.cols; col++) {
            float minv = 10000.0;
            int cd = -1;
            for (int d = 0; d < max_disparity; d++) {
                if (cost_volume[d*left.cols*left.rows + row*left.cols + col] < minv) {
                    minv = cost_volume[d*left.cols*left.rows + row*left.cols + col];
                    cd = d;
                }
            }
            // something is wrong when I use Mat#at, cannot figure out what
            disparity_left.at<short>(row, col) = cd;
        }
    }

    // cout << dx_left.size() << " " << left.size() << endl;    
    // disparity.convertTo(disparity, CV_8UC1m);
    cv::Mat lvis;
    su::convert_to_disparity_visualize(disparity_left, lvis, true);
    cv::imshow("Scharr", lvis);
    // cv::waitKey(0);

    std::fill_n(cost_volume, max_disparity*left.cols*left.rows, 100000.0);  // TODO
    for (int d = 0; d < max_disparity; d++) {
        float *disparity_level = cost_volume + d*left.cols*left.rows;
        for (int row = 0; row < left.rows; row++) {
            // TODO all this shit into a function
            tad_r(    dx_left.ptr<short>(row), left.ptr<uchar>(row), dx_right.ptr<short>(row),
                    right.ptr<uchar>(row), disparity_level + row*left.cols, left.cols, d,
                    100, 100, LAMBDA);
        }
        cv::Mat cost_slice = cv::Mat(left.rows, left.cols, CV_32FC1, disparity_level);
        // TODO might be faster if I implement it
        cv::boxFilter(cost_slice, cost_slice, -1, cv::Size(5, 5));
    }

    //TODO this can be done with extra memory to achieve cache locality
    cv::Mat disparity_right = cv::Mat(left.rows, left.cols, CV_16SC1);
    for (int row = 0; row < left.rows; row++) {
        for (int col = 0; col < left.cols; col++) {
            float minv = 10000.0;
            int cd = -1;
            for (int d = 0; d < max_disparity; d++) {
                if (cost_volume[d*left.cols*left.rows + row*left.cols + col] < minv) {
                    minv = cost_volume[d*left.cols*left.rows + row*left.cols + col];
                    cd = d;
                }
            }
            // something is wrong when I use Mat#at, cannot figure out what
            disparity_right.at<short>(row, col) = cd;
        }
    }

    // cout << dx_left.size() << " " << left.size() << endl;    
    // disparity_right.convertTo(disparity_right, CV_8UC1); 
    cv::Mat rvis;
    su::convert_to_disparity_visualize(disparity_right, rvis);
    cv::imshow("Scharr2", rvis);

    cv::Mat stable = lr_check(disparity_left, disparity_right);
    cv::imshow("Stable", stable);

    cv::waitKey(0);

    delete[] cost_volume;
    cout << "Exit" << endl;
}

void DisparityPropagationStereo::compute_disparity(const cv::Mat &left, const cv::Mat &right) {
    su::require(left.cols == right.cols && left.rows == right.rows, "Shape of the images must match");
    su::require(left.channels() == 1 && right.channels() == 1, "Images must be grayscale");
    // TODO require lambda 0..1
    this->left = left;
    this->right = right;
    preprocess();
    cout << "Full exit" << endl;
}