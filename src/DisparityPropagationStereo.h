#ifndef DISPARITY_PROPAGATION_STEREO_H
#define DISPARITY_PROPAGATION_STEREO_H

#include <opencv2/core/core.hpp>

class DisparityPropagationStereo {
private:
    int max_disparity;

    // aux arrays
    short *min3_disps;

    cv::Mat left, right;
    int height, width;

    //TODO rename these disparities to raw
    cv::Mat stable, disparity_left, disparity_right;
    cv::Mat disparity;

private:
    // Winner-take-all
    static void wta(cv::Mat &disparity, const float *cost_volume, int max_d, int rows, int cols);
    // Left-right consistenct check
    static cv::Mat lr_check(const cv::Mat &disparity_left, const cv::Mat &disparity_right);
    static void geodesic_filter(float *cost_volume, const cv::Mat &img, int max_d);
    // Truncated Absolute Difference
    inline void tad( const cv::Mat &dx_left, 
                                            const cv::Mat &dx_right, float *cost, int row,
                                            int width, int d, float lambda);
    static inline void tad_r( const short *dx_left, const uchar *left, const short *dx_right,
                            const uchar *right, float *cost, int width, int d, short tg, short tc,
                            float lambda);
    void preprocess();
    void disparity_propagation();
public:
    DisparityPropagationStereo(int max_disparity);
    cv::Mat compute_disparity(const cv::Mat &left, const cv::Mat &right);
};

#endif