#ifndef FIVE_REGION_STEREO_H_
#define FIVE_REGION_STEREO_H_

#include <opencv2/core/core.hpp>

// This is a C++ implementation of FIVE_RECT algorithm from BoofCV. That algorithm is based on
// Heiko Hirschmuller, Peter R. Innocent, and Jon Garibaldi. "Real-Time Correlation-Based Stereo
// Vision with Reduced Border Errors." Int. J. Comput. Vision 47, 1-3 2002

// For reference check:
// https://github.com/lessthanoptimal/BoofCV/blob/master/main/feature/src
//                                      /boofcv/alg/feature/disparity/DisparityScoreWindowFive.java
// Hopefully this implementation runs faster. Most of the comments are also c/p from BoofCV.
// It computes dense disparity map with subpixel accuracy.
class FiveRegionStereo {
private:
    static const int DISCRETIZER = 10000;

    int min_disparity, max_disparity;
    int region_width, region_height;
    int radiusX, radiusY;
    int max_error, validate_RtoL;
    double texture;
    
    int invalid_disparity;
    int length_horizontal;
    int image_width;

    int active_vertical_score;

    // aux arrays
    int *element_score;
    int *five_score;
    int *column_score;

    // aux matrices
    int *horizontal_score;
    int *vertical_score;

    // images
    cv::Mat left;
    cv::Mat right;
    // cv::Mat disparity;

private:
    void configure(const cv::Mat &left, const cv::Mat &right);
    void compute_score_row_sad(int element_max, int index_left, int index_right);
    void compute_score_row(int row, int *scores);
    void compute_first_row();
    void compute_score_five(int *top, int *middle, int *bottom, int *score);
    int max_disparity_at_column_L2R(int col);
    int select_right_to_left(int col, int *scores, int region_width);
    //TODO rename
    void process(int row, int* scores, cv::Mat &image_disparity, int radiusX, int region_width);
    void compute_remaining_rows(cv::Mat &disparity);
public:
    // max_per_pixel_error - The maximum allowed error.  Note this is per pixel error.
    //                       Try 10.
    // validate_RtoL - Tolerance for how difference the left to right associated
    //                 values can be.  Try 6
    // texture - Tolerance for how similar optimal region is to other region.
    //           Disable with a value < 0. Closer to zero is more tolerant. Try 0.1
    FiveRegionStereo(   int min_disparity, int max_disparity, int radiusX, int radiusY,
                        int max_per_pixel_error, int validate_RtoL, double texture);
    ~FiveRegionStereo();
    cv::Mat compute_disparity(const cv::Mat &left, const cv::Mat &right);
};

#endif