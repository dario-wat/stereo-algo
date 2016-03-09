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
    // Truncated Absolute Difference
    static inline void tad( const short *dx_left, const uchar *left, const short *dx_right,
                            const uchar *right, float *cost, int width, int d, int tg, uchar tc,
                            double lambda);
    static inline void tad_r( const short *dx_left, const uchar *left, const short *dx_right,
                            const uchar *right, float *cost, int width, int d, int tg, uchar tc,
                            double lambda);
    void preprocess();
    void disparity_propagation();
public:
    DisparityPropagationStereo(int max_disparity);
    cv::Mat compute_disparity(const cv::Mat &left, const cv::Mat &right);
};

#endif