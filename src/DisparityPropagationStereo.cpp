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

// Everything has type short to speed up (cache locality)

// Using this as max value for float, for some reason, numeric_limits worsens results
static const float FL_MAX = 10000.0f;

DisparityPropagationStereo::DisparityPropagationStereo( int max_disparity, float sigma_s,
                                                        float sigma_r) {
    su::require(max_disparity > 3, "Max disparity must be grater than 3 (due to R penalty term)");
    su::require(sigma_s > 0.0, "Sigma_s must be positive");
    su::require(sigma_r > 0.0, "Sigma_r must be positive");
    this->max_disparity = max_disparity;
    this->sigma_s = sigma_s;
    this->sigma_r = sigma_r;
}

// Using fabsf to vectorize. g++ does not vectorize integral type abs functions
inline void DisparityPropagationStereo::tad(const short *dx_left, const uchar *left,
                                            const short *dx_right, const uchar *right, float *cost,
                                            int width, int d, short tg, short tc, float lambda) {
    for (int col = d; col < width; col++) {
        cost[col] = lambda * std::min<int>(fabsf((int)left[col] - right[col-d]), tc)
            + (1 - lambda) * std::min<int>(fabsf(dx_left[col] - dx_right[col-d]), tg);
    }
}

// Also using fabsf to vectorize
inline void DisparityPropagationStereo::tad_r(  const short *dx_left, const uchar *left,
                                                const short *dx_right, const uchar *right, float *cost,
                                                int width, int d, short tg, short tc, float lambda) {
    for (int col = 0; col < width-d; col++) {
        cost[col] = lambda * std::min<int>(fabsf((int)left[col+d] - right[col]), tc)
            + (1 - lambda) * std::min<int>(fabsf(dx_left[col+d] - dx_right[col]), tg);
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

// Geodesic filter as explained in the paper
void DisparityPropagationStereo::geodesic_filter(float *cost_volume, const cv::Mat &img, int max_d,
        float sigma_s, float sigma_r) {
    int rows = img.rows, cols = img.cols;
    // Instead of calculating alphas, this way I can create lookup table and get the values much
    // faster. There is no need to do this in the constructor since the overhead of creating
    // this table compared to the rest of the filter is negligible.
    float alpha[512];
    for (int diff = -255; diff < 256; diff++) {
        alpha[diff+255] = exp(-1/sigma_s - std::abs(diff) / sigma_r);
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
                                                int max_d, int rows, int cols, float lc) {
    // Lookup table for penalty term R    
    float penalty[512];
    for (int i = 0; i < 512; i++) {
        penalty[i] = 2*lc;
    }
    penalty[256] = lc;
    penalty[254] = lc;
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

void DisparityPropagationStereo::preprocess() {
    // Edges (derivatives)
    cv::Mat dx_left, dx_right;
    cv::Scharr(left, dx_left, CV_16SC1, 1, 0);
    cv::Scharr(right, dx_right, CV_16SC1, 1, 0);

    // Cost volume has (disparity, height, width) ordering so that it matches Mat and can be
    // used for box filtering
    std::fill_n(cost_volume, max_disparity*left.cols*left.rows, FL_MAX);

    // TAD matching cost computation and box filter aggregation for left image
    for (int d = 0; d < max_disparity; d++) {
        float *disparity_level = cost_volume + d*left.cols*left.rows;
        for (int row = 0; row < left.rows; row++) {
            tad(dx_left.ptr<short>(row), left.ptr<uchar>(row), dx_right.ptr<short>(row),
                right.ptr<uchar>(row), disparity_level + row*left.cols, left.cols, d,
                TRUNC_INT, TRUNC_GRAD, LAMBDA);
        }
        cv::Mat cost_slice = cv::Mat(left.rows, left.cols, CV_32FC1, disparity_level);
        cv::boxFilter(cost_slice, cost_slice, -1, cv::Size(5, 5));
    }

    // Calculate min 3 disparities for each pixel
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
                    TRUNC_INT, TRUNC_GRAD, LAMBDA);
        }
        cv::Mat cost_slice = cv::Mat(left.rows, left.cols, CV_32FC1, disparity_level);
        cv::boxFilter(cost_slice, cost_slice, -1, cv::Size(5, 5));
    }

    // Right image disparity
    disparity_right = cv::Mat(left.rows, left.cols, CV_16SC1);
    wta(disparity_right, cost_volume, max_disparity, left.rows, left.cols);

    // Left right consistency check to find stable pixels
    stable = lr_check(disparity_left, disparity_right);
}

void DisparityPropagationStereo::disparity_propagation() {
    matching_cost(  cost_volume, min3_disps, disparity_left, stable, max_disparity,
                    left.rows, left.cols, LC);
    geodesic_filter(cost_volume, left, max_disparity, sigma_s, sigma_r);

    disparity = cv::Mat(left.rows, left.cols, CV_16SC1);
    wta(disparity, cost_volume, max_disparity, left.rows, left.cols);
    cv::medianBlur(disparity, disparity, 5);
    cv::medianBlur(disparity, disparity, 5);
    cv::medianBlur(disparity, disparity, 5);
}
 
cv::Mat DisparityPropagationStereo::compute_disparity(const cv::Mat &left, const cv::Mat &right) {
    su::require(left.cols == right.cols && left.rows == right.rows, "Shape of the images must match");
    su::require(left.channels() == 1 && right.channels() == 1, "Images must be grayscale");
    this->left = left;
    this->right = right;
    
    cost_volume = new float[max_disparity*left.cols*left.rows];
    min3_disps = new short[left.cols*left.rows*DC];
    
    preprocess();
    disparity_propagation();

    delete[] cost_volume;
    delete[] min3_disps;
    return disparity;
}
