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

// Everything is short to speed up (cache locality)

// Using this as max value for float, for some reason, numeric_limits worsens results
static const float FL_MAX = 10000.0f;

static const float LAMBDA = 0.5f;
static const float LC = 0.2f;
static const short TRUNC_INT = 100;
static const short TRUNC_GRAD = 100;


static const float SIGMA_S = 42.5f / 6;
static const float SIGMA_R = 22.5f / 6;


DisparityPropagationStereo::DisparityPropagationStereo(int max_disparity) {
    su::require(max_disparity > 0, "Max disparity must be positive");
    this->max_disparity = max_disparity;
}

//TODO statictize this function
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

// Left-right consistency check to create stability map. A pixel is stable if
// D_left(x,y) == D_right( (x,y) - (D_left(x,y),0) )
cv::Mat DisparityPropagationStereo::lr_check(   const cv::Mat &disparity_left,
                                                const cv::Mat &disparity_right) {
    cv::Mat stable = cv::Mat(disparity_left.rows, disparity_left.cols, CV_8UC1);
    for (int row = 0; row < disparity_left.rows; row++) {
        for (int col = 0; col < disparity_left.cols; col++) {
            short d_left = disparity_left.at<short>(row, col);
            short d_right = disparity_right.at<short>(row, col-d_left);
            stable.at<uchar>(row, col) = col - d_left < 0 || d_left != d_right ? 0 : 255;
        }
    }
    return stable;
}

// Performs winner-take-all optimization using auxilliray array to deal with cache locality
void DisparityPropagationStereo::wta(   cv::Mat &disparity, const float *cost_volume, int max_d,
                                        int rows, int cols) {
    if (disparity.rows != rows || disparity.cols != cols || disparity.type() != CV_16SC1) {
        disparity = cv::Mat(rows, cols, CV_16SC1);
    }
    cv::Mat min_costs = cv::Mat::ones(rows, cols, CV_32FC1) * FL_MAX;
    for (int d = 0; d < max_d; d++) {
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                if (cost_volume[d*cols*rows + row*cols + col] < min_costs.at<float>(row, col)) {
                    min_costs.at<float>(row, col) = cost_volume[d*cols*rows + row*cols + col];
                    disparity.at<short>(row, col) = d;
                }
            }
        }
    }
}

// Additional array is used to deal with cache locality and setup loops in proper ordering.
// Construction of that array is about 1% of runtime of this function, not sure if it is
// worth destroying the code more just to squeeze that out.
// Reordering or reimplementing the min3 logic could help, but the compiler seems too smart.
void DisparityPropagationStereo::min3_disparities(  short *min3_disps, const float *cost_volume,
                                                    int max_d, int rows, int cols) {
    float *aux_costs = new float[cols*rows*DC];
    std::fill_n(aux_costs, cols*rows*DC, FL_MAX);
    for (int d = 0; d < max_d; d++) {
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                float cost = cost_volume[d*cols*rows + row*cols + col];
                int i = row*DC*cols + col*DC;
                if (cost < aux_costs[i]) {
                    aux_costs[i+2] = aux_costs[i+1];
                    aux_costs[i+1] = aux_costs[i];
                    aux_costs[i] = cost;
                    min3_disps[i+2] = min3_disps[i+1];
                    min3_disps[i+1] = min3_disps[i];
                    min3_disps[i] = d;
                } else if (cost < aux_costs[i+1]) {
                    aux_costs[i+2] = aux_costs[i+1];
                    aux_costs[i+1] = cost;
                    min3_disps[i+2] = min3_disps[i+1];
                    min3_disps[i+1] = d;
                } else if (cost < aux_costs[i+2]) {
                    aux_costs[i+2] = cost;
                    min3_disps[i+2] = d;
                }
            }
        }
    }
    delete[] aux_costs;
}

//TODO get rid of these global constants
// Geodesic filter as explained in the paper
void DisparityPropagationStereo::geodesic_filter(float *cost_volume, const cv::Mat &img, int max_d) {
    int rows = img.rows, cols = img.cols;
    // Instead of calculating alphas, this way I can create lookup table and get the values much
    // faster. There is no need to do this in the constructor since the overhead of creating
    // this table compared to the rest of the filter is negligible.
    float alpha[512];
    for (int diff = -255; diff < 256; diff++) {
        alpha[diff+255] = exp(-1/SIGMA_S - std::abs(diff) / SIGMA_R);
    }

    // For some reason in the first two loops it runs faster if row and col loops are placed
    // in this order even though it increases cache misses. When trying to improve cache locality
    // loops run slower.

    // Horizontal left to right pass
    for (int d = 0; d < max_d; d++) {
        for (int col = 1; col < cols; col++) {
            for (int row = 0; row < rows; row++) {
                float a = alpha[255 + img.at<uchar>(row, col) - img.at<uchar>(row, col-1)];
                cost_volume[d*cols*rows + row*img.cols + col] +=
                    a * cost_volume[d*cols*rows + row*cols + col-1];
            }
        }
    }

    // Horizontal right to left pass
    for (int d = 0; d < max_d; d++) {
        for (int col = cols-2; col >= 0; col--) {
            for (int row = 0; row < rows; row++) {
                float a = alpha[255 + img.at<uchar>(row, col) - img.at<uchar>(row, col+1)];
                cost_volume[d*cols*rows + row*cols + col] =
                    (1 - a*a) * cost_volume[d*cols*rows + row*cols + col]
                    + a * cost_volume[d*cols*rows + row*cols + col+1];
            }
        }
    }

//TODO maybe vectorize these 2 loops
    // Vertical top to bottom pass
    for (int d = 0; d < max_d; d++) {
        for (int row = 1; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                float a = alpha[255 + img.at<uchar>(row, col) - img.at<uchar>(row-1, col)];
                cost_volume[d*cols*rows + row*cols + col] +=
                    a * cost_volume[d*cols*rows + (row-1)*cols + col];
            }
        }
    }

    // Vertical bottom to top pass
    for (int d = 0; d < max_d; d++) {
        for (int row = rows-2; row >= 0; row--) {
            for (int col = 0; col < cols; col++) {
                float a = alpha[255 + img.at<uchar>(row, col) - img.at<uchar>(row+1, col)];
                cost_volume[d*cols*rows + row*cols + col] =
                    (1 - a*a) * cost_volume[d*cols*rows + row*cols + col]
                    + a * cost_volume[d*cols*rows + (row+1)*cols + col];
            }
        }
    }
}

// Matching cost computation. Creates a table for penalty term R to make the calculations
// faster. This still seems to be the slowest part of the code.
void DisparityPropagationStereo::matching_cost( float *cost_volume, const short *min3_disps,
                                                const cv::Mat &disparity, const cv::Mat &stable,
                                                int max_d, int rows, int cols) {
    // Lookup table for penalty term R    
    float penalty[512];
    for (int i = 0; i < 512; i++) {
        penalty[i] = 2*LC;
    }
    penalty[256] = LC;
    penalty[254] = LC;
    penalty[255] = 0;

    // Matching cost computation, same one from the paper
    for (int d = 0; d < max_d; d++) {
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                if (stable.at<uchar>(row, col)) {
                    short d1 = min3_disps[row*cols*DC + col*DC];
                    short d2 = min3_disps[row*cols*DC + col*DC + 1];
                    short d3 = min3_disps[row*cols*DC + col*DC + 2];
                    
                    float r = penalty[255+d1-d];
                    r += penalty[255+d2-d];
                    r += penalty[255+d3-d];

                    int diff = (d - disparity.at<short>(row, col));
                    cost_volume[d*cols*rows + row*cols + col] = diff * diff + r;
                } else {
                    cost_volume[d*cols*rows + row*cols + col] = 0;
                }
            }
        }
    }
}

// TODO this needs love
void DisparityPropagationStereo::preprocess() {
    // Edges (derivatives)
    cv::Mat dx_left, dx_right;
    cv::Scharr(left, dx_left, CV_16SC1, 1, 0);
    cv::Scharr(right, dx_right, CV_16SC1, 1, 0);

    // Cost volume has (disparity, height, width) ordering so that it matches Mat and can be
    // used for box filtering
    float *cost_volume = new float[max_disparity*left.cols*left.rows];
    std::fill_n(cost_volume, max_disparity*left.cols*left.rows, FL_MAX);

    // TAD matching cost computation and box filter aggregation for left image
    for (int d = 0; d < max_disparity; d++) {
        float *disparity_level = cost_volume + d*left.cols*left.rows;
        for (int row = 0; row < left.rows; row++) {
            tad(dx_left, dx_right, disparity_level + row*left.cols, row, left.cols, d, LAMBDA);
        }
        cv::Mat cost_slice = cv::Mat(left.rows, left.cols, CV_32FC1, disparity_level);
        cv::boxFilter(cost_slice, cost_slice, -1, cv::Size(5, 5));
    }

    // TODO maybe move this at the top of disparity propagation
    min3_disps = new short[left.cols*left.rows*DC];
    std::fill_n(min3_disps, left.cols*left.rows*DC, -1);
    min3_disparities(min3_disps, cost_volume, max_disparity, left.rows, left.cols);

    // Left image disparity
    disparity_left = cv::Mat(left.rows, left.cols, CV_16SC1);
    wta(disparity_left, cost_volume, max_disparity, left.rows, left.cols);
    

    // TAD matching cost computation and box filter aggregation for right image
    std::fill_n(cost_volume, max_disparity*left.cols*left.rows, FL_MAX);
    for (int d = 0; d < max_disparity; d++) {
        float *disparity_level = cost_volume + d*left.cols*left.rows;
        for (int row = 0; row < left.rows; row++) {
            tad_r(  dx_left.ptr<short>(row), left.ptr<uchar>(row), dx_right.ptr<short>(row),
                    right.ptr<uchar>(row), disparity_level + row*left.cols, left.cols, d,
                    100, 100, LAMBDA);
        }
        cv::Mat cost_slice = cv::Mat(left.rows, left.cols, CV_32FC1, disparity_level);
        cv::boxFilter(cost_slice, cost_slice, -1, cv::Size(5, 5));
    }

    // Right image disparity
    disparity_right = cv::Mat(left.rows, left.cols, CV_16SC1);
    wta(disparity_right, cost_volume, max_disparity, left.rows, left.cols);

    // Left right consistency check to find stable pixels
    stable = lr_check(disparity_left, disparity_right);

    // TODO add other deletes
    delete[] cost_volume;
}

void DisparityPropagationStereo::disparity_propagation() {
    float *cost_volume = new float[max_disparity*left.rows*left.cols];
    matching_cost(cost_volume, min3_disps, disparity_left, stable, max_disparity, left.rows, left.cols);

    geodesic_filter(cost_volume, left, max_disparity);

    cv::Mat disparity_map = cv::Mat(left.rows, left.cols, CV_16SC1);
    wta(disparity_map, cost_volume, max_disparity, left.rows, left.cols);
    disparity = disparity_map;
    delete[] cost_volume;
    delete[] min3_disps;        // TODO dont like this
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