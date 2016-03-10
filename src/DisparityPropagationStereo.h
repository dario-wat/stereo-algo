#ifndef DISPARITY_PROPAGATION_STEREO_H
#define DISPARITY_PROPAGATION_STEREO_H

#include <opencv2/core/core.hpp>

// This algorithms is an implementation of
// "Real-time local stereo via edge-aware propagation" - Sun, et al.
// DC parameter (number of minimum disparities to use in penalty term R) is fixed to 3. Some
// reimplementation is needed to change that.
// The algorithm is also made to work only on grayscale images.
// Some parameters are hardcoded and are different than what is mentioned in the paper.
// The algorithm might have some bugs that I am unaware of.
class DisparityPropagationStereo {
private:
    // Parameters
    static constexpr float LAMBDA = 0.2f;
    static const short TRUNC_INT = 100;
    static const short TRUNC_GRAD = 100;
    static constexpr float LC = 4.0f;

    // static const float SIGMA_S = 42.5f / 6;
    // static const float SIGMA_R = 22.5f / 6;
    
    int max_disparity;
    float sigma_s, sigma_r;
    cv::Mat left, right;

    // aux arrays
    short *min3_disps;
    float *cost_volume;

    // aux matrices
    cv::Mat stable, disparity_left, disparity_right;
    cv::Mat disparity;

private:
    // Winner-take-all
    static void wta(cv::Mat &disparity, const float *cost_volume, int max_d, int rows, int cols);
    // Left-right consistenct check
    static cv::Mat lr_check(const cv::Mat &disparity_left, const cv::Mat &disparity_right);
    // Geodesic filter from paper
    static void geodesic_filter(float *cost_volume, const cv::Mat &img, int max_d,
                                float sigma_s, float sigma_r);
    // Find 3 disparities with minimum matching cost at each pixel. Used for R.
    static void min3_disparities(   short *min_disps, const float *cost_volume, int max_d,
                                    int rows, int cols);
    
    // Truncated Absolute Difference (for left image)
    static inline void tad( const short *dx_left, const uchar *left,
                            const short *dx_right, const uchar *right, float *cost,
                            int width, int d, short tg, short tc, float lambda);
    // Truncated Aboslute Difference (for right image)
    static inline void tad_r(   const short *dx_left, const uchar *left,
                                const short *dx_right, const uchar *right, float *cost,
                                int width, int d, short tg, short tc, float lambda);
    
    // Second (real) matching cost computation
    static void matching_cost(  float *cost_volume, const short *min3_disps,
                                const cv::Mat &disparity, const cv::Mat &stable,
                                int max_d, int rows, int cols, float lc);

    void preprocess();
    void disparity_propagation();

public:
    DisparityPropagationStereo(int max_disparity, float sigma_s, float sigma_r);
    cv::Mat compute_disparity(const cv::Mat &left, const cv::Mat &right);
};

#endif