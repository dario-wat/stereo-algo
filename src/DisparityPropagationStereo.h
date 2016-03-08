#ifndef DISPARITY_PROPAGATION_STEREO_H
#define DISPARITY_PROPAGATION_STEREO_H

#include <opencv2/core/core.hpp>

class DisparityPropagationStereo {
private:
    int max_disparity;

    cv::Mat left, right;
    int height, width;

private:
    // Truncated Absolute Difference
    static inline void tad( const short *dx_left, const uchar *left, const short *dx_right,
                            const uchar *right, float *cost, int width, int d, int tg, uchar tc,
                            double lambda);
    static inline void tad_r( const short *dx_left, const uchar *left, const short *dx_right,
                            const uchar *right, float *cost, int width, int d, int tg, uchar tc,
                            double lambda);
    void preprocess();
public:
    DisparityPropagationStereo(int max_disparity);
    void compute_disparity(const cv::Mat &left, const cv::Mat &right);
};

#endif