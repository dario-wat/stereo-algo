#include "DisparityPropagationStereo.h"

#include <ctime>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "st_util.h"

#define DC 3        // Changing this might break the algorithm

/**** JUST FOR DEBUGGING, WILL BE REMOVED ****/
#define CLOCKS clock_t start = clock();
#define CLOCKP std::cerr << (clock()-start) / float(CLOCKS_PER_SEC) << std::endl;
#define CLOCKF std::cerr << (clock()-start) << std::endl;

#define CPUS uint64_t start = su::rdtsc();
#define CPUE std::cerr << su::rdtsc() - start << endl;
using std::cout;
using std::cerr;
using std::endl;
/************************/

// Using this as max value for float, for some reason, numeric_limits worsens results
static const float FL_MAX = 10000.0f;

static const float LAMBDA = 0.5f;
static const float LC = 0.2f;
static const short TRUNC_INT = 100;
static const short TRUNC_GRAD = 100;

inline void sub_shift_vectorize(const short *a, const short *b, float *c, int n, int d) {
    for (int i = d; i < n; i++) {
        c[i] = (float) (a[i] - b[i-d]);
    }
}

inline void abs_shift_vectorize(float *c, int n, int d) {
    for (int i = d; i < n; i++) {
        c[i] = fabsf(c[i]);
    }
}

inline void min_trunc_shift_vectorize(float *c, float t, int n, int d) {
    for (int i = d; i < n; i++) {
        c[i] = std::min<float>(c[i], t);
    }
}


DisparityPropagationStereo::DisparityPropagationStereo(int max_disparity) {
    su::require(max_disparity > 0, "Max disparity must be positive");
    this->max_disparity = max_disparity;
}

//TODO lambda to float
inline void DisparityPropagationStereo::tad(const cv::Mat &dx_left, 
                                            const cv::Mat &dx_right, float *cost, int row,
                                            int width, int d, float lambda) {
    for (int col = d; col < width; col++) {
        short int_diff = std::abs((short)left.at<uchar>(row, col) - right.at<uchar>(row, col-d));
        short c_intensity = std::min<short>(int_diff, TRUNC_INT);
        short grad_diff = std::abs(dx_left.at<short>(row, col) - dx_right.at<short>(row, col-d));
        short c_grad = std::min<short>(grad_diff, TRUNC_GRAD);
        cost[col] = lambda * c_intensity + (1.0f - lambda) * c_grad;
    }
}

inline void DisparityPropagationStereo::tad_r(  const short *dx_left, const uchar *left,
                                                const short *dx_right, const uchar *right, float *cost, 
                                            int width, int d, short tg, short tc, float lambda) {
    // TODO vectorize this
    for (int col = 0; col < width-d; col++) {
        // cout << dx_left[col+d] - dx_right[col] << endl;
        cost[col] = lambda * std::min<int>(std::abs((int)left[col+d] - right[col]), tc) +
            (1 - lambda) * std::min<int>(std::abs(dx_left[col+d] - dx_right[col]), tg);
    }
}

cv::Mat lr_check(const cv::Mat &disparity_left, const cv::Mat &disparity_right) {
    cv::Mat stable = cv::Mat(disparity_left.rows, disparity_left.cols, CV_8UC1);
    for (int row = 0; row < disparity_left.rows; row++) {
        for (int col = 0; col < disparity_left.cols; col++) {
            short d = disparity_left.at<short>(row, col);
            if (col - (int)d < 0 || d != disparity_right.at<short>(row, col-d)) {
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
    std::fill_n(cost_volume, max_disparity*left.cols*left.rows, FL_MAX);  // TODO

    for (int d = 0; d < max_disparity; d++) {
        float *disparity_level = cost_volume + d*left.cols*left.rows;
        // CLOCKS
        for (int row = 0; row < left.rows; row++) {
            // TODO all this shit into a function
            tad(dx_left, dx_right, disparity_level + row*left.cols, row, left.cols, d, LAMBDA);
        }
        cv::Mat cost_slice = cv::Mat(left.rows, left.cols, CV_32FC1, disparity_level);
        cv::boxFilter(cost_slice, cost_slice, -1, cv::Size(5, 5));
    }

    min3_disps = new short[left.cols*left.rows*DC];
    std::fill_n(min3_disps, left.cols*left.rows*DC, -1);
    // TODO maybe swap loops for cache locality    
    // Also minimize the number of operations
    for (int row = 0; row < left.rows; row++) {
        for (int col = 0; col < left.cols; col++) {
            float minv1 = 10000.0, minv2 = 10000.0, minv3 = 10000.0;
            for (int d = 0; d < max_disparity; d++) {
                float cost = cost_volume[d*left.cols*left.rows + row*left.cols + col];
                if (cost < minv1) {
                    minv3 = minv2;
                    minv2 = minv1;
                    minv1 = cost;
                    min3_disps[row*DC*left.cols + col*DC + 2] = min3_disps[row*DC*left.cols + col*DC + 1];
                    min3_disps[row*DC*left.cols + col*DC + 1] = min3_disps[row*DC*left.cols + col*DC];
                    min3_disps[row*DC*left.cols + col*DC] = d;
                } else if (cost < minv2) {
                    minv3 = minv2;
                    minv2 = cost;
                    min3_disps[row*DC*left.cols + col*DC + 2] = min3_disps[row*DC*left.cols + col*DC + 1];
                    min3_disps[row*DC*left.cols + col*DC + 1] = d;
                } else if (cost < minv3) {
                    minv3 = cost;
                    min3_disps[row*DC*left.cols + col*DC + 2] = cost;
                }
            }
        }
    }

    
    disparity_left = cv::Mat(left.rows, left.cols, CV_16SC1);

    CLOCKS
    cv::Mat min_costs = cv::Mat::ones(left.rows, left.cols, CV_32FC1) * FL_MAX;
    for (int d = 0; d < max_disparity; d++) {
        for (int row = 0; row < left.rows; row++) {
            for (int col = 0; col < left.cols; col++) {
                if (cost_volume[d*left.cols*left.rows + row*left.cols + col] < min_costs.at<float>(row, col)) {
                    min_costs.at<float>(row, col) = cost_volume[d*left.cols*left.rows + row*left.cols + col];
                    disparity_left.at<short>(row, col) = d;
                }
            }
        }
    }
    // for (int row = 0; row < left.rows; row++) {
    //     for (int col = 0; col < left.cols; col++) {
    //         float minv = FL_MAX;
    //         int cd = -1;
    //         for (int d = 0; d < max_disparity; d++) {
    //             if (cost_volume[d*left.cols*left.rows + row*left.cols + col] < minv) {
    //                 minv = cost_volume[d*left.cols*left.rows + row*left.cols + col];
    //                 cd = d;
    //             }
    //         }
    //         // something is wrong when I use Mat#at, cannot figure out what
    //         disparity_left.at<short>(row, col) = cd;
    //     }
    // }
    CLOCKF

    // cout << dx_left.size() << " " << left.size() << endl;    
    // disparity.convertTo(disparity, CV_8UC1m);
    cv::Mat lvis;
    su::convert_to_disparity_visualize(disparity_left, lvis, true);
    // cv::imshow("Scharr", lvis);
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
    disparity_right = cv::Mat(left.rows, left.cols, CV_16SC1);
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
    // cv::imshow("Scharr2", rvis);

    stable = lr_check(disparity_left, disparity_right);
    // cv::imshow("Stable", stable);

    // cv::waitKey(0);

    delete[] cost_volume;
    // cout << "Exit" << endl;
}

static const float SIGMA_S = 42.5f / 6;
static const float SIGMA_R = 22.5f / 6;

float alpha(uchar p, uchar pl) {
    return exp(-1/SIGMA_S - std::abs((int)p - pl) / SIGMA_R);
}

void DisparityPropagationStereo::disparity_propagation() {
    float *cost_volume = new float[max_disparity*left.rows*left.cols];
    for (int d = 0; d < max_disparity; d++) {
        for (int row = 0; row < left.rows; row++) {
            for (int col = 0; col < left.cols; col++) {
                if (stable.at<uchar>(row, col)) {
                    float r = 0;
                    short d1 = min3_disps[row*left.cols*DC + col*DC];
                    short d2 = min3_disps[row*left.cols*DC + col*DC + 1];
                    short d3 = min3_disps[row*left.cols*DC + col*DC + 2];
                    if (std::fabs(d1-d) > 1.0f) {
                        r += 2*LC;
                    } else {
                        r += LC * std::fabs(d1-d) * std::fabs(d1-d);
                    }
                    if (std::fabs(d2-d) > 1.0f) {
                        r += 2*LC;
                    } else {
                        r += LC * std::fabs(d2-d) * std::fabs(d2-d);
                    }
                    if (std::fabs(d3-d) > 1.0f) {
                        r += 2*LC;
                    } else {
                        r += LC * std::fabs(d3-d) * std::fabs(d3-d);
                    }
                    cost_volume[d*left.cols*left.rows + row*left.cols + col] =
                        (d - (int)disparity_left.at<short>(row, col))
                        * (d - (int)disparity_left.at<short>(row, col))
                        + r;
                } else {
                    cost_volume[d*left.cols*left.rows + row*left.cols + col] = 0;
                }
            }
        }
    }

    for (int d = 0; d < max_disparity; d++) {
        for (int row = 0; row < left.rows; row++) {
            for (int col = 1; col < left.cols; col++) {
                // cout << alpha(left.at<uchar>(row, col), left.at<uchar>(row, col-1)) << ' ';
                cost_volume[d*left.cols*left.rows + row*left.cols + col] +=
                    alpha(left.at<uchar>(row, col), left.at<uchar>(row, col-1))
                    * cost_volume[d*left.cols*left.rows + row*left.cols + col - 1];
            }
        }
    }

    for (int d = 0; d < max_disparity; d++) {
        for (int row = 0; row < left.rows; row++) {
            for (int col = left.cols-2; col >= 0; col--) {
                float a = alpha(left.at<uchar>(row, col), left.at<uchar>(row, col+1));
                cost_volume[d*left.cols*left.rows + row*left.cols + col] =
                    (1 - a*a) * cost_volume[d*left.cols*left.rows + row*left.cols + col]
                    + a * cost_volume[d*left.cols*left.rows + row*left.cols + col+1];
            }
        }
    }

    for (int d = 0; d < max_disparity; d++) {
        for (int col = 0; col < left.cols; col++) {
            for (int row = 1; row < left.rows; row++) {
                cost_volume[d*left.cols*left.rows + row*left.cols + col] +=
                    alpha(left.at<uchar>(row, col), left.at<uchar>(row-1, col))
                    * cost_volume[d*left.cols*left.rows + (row-1)*left.cols + col];
            }
        }
    }

    for (int d = 0; d < max_disparity; d++) {
        for (int col = 0; col < left.cols; col++) {
            for (int row = left.rows-2; row >= 0; row--) {
                float a = alpha(left.at<uchar>(row, col), left.at<uchar>(row+1, col));
                cost_volume[d*left.cols*left.rows + row*left.cols + col] =
                    (1 - a*a) * cost_volume[d*left.cols*left.rows + row*left.cols + col]
                    + a * cost_volume[d*left.cols*left.rows + (row+1)*left.cols + col];
            }
        }
    }

    cv::Mat disparity_map = cv::Mat(left.rows, left.cols, CV_16SC1);
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
            disparity_map.at<short>(row, col) = cd;
        }
    }
    // cout << disparity_map << endl;
    // su::print_mat<short>(disparity_map);
    disparity = disparity_map;
    // su::convert_to_disparity_visualize(disparity_map, disparity_map, true);
    // cv::imshow("Final disparity", disparity_map);
    // cv::waitKey(0);
}
 
cv::Mat DisparityPropagationStereo::compute_disparity(const cv::Mat &left, const cv::Mat &right) {
    su::require(left.cols == right.cols && left.rows == right.rows, "Shape of the images must match");
    su::require(left.channels() == 1 && right.channels() == 1, "Images must be grayscale");
    // TODO require lambda 0..1
    // TODO require other parameters
    this->left = left;
    this->right = right;
    preprocess();
    disparity_propagation();
    // cout << "Full exit" << endl;
    return disparity;
}