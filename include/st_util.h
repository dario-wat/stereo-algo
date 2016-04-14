#ifndef STEREO_ALGO_UTIL_H_
#define STEREO_ALGO_UTIL_H_

#include <string>
#include <vector>
#include <opencv2/core/core.hpp>

namespace su {
    void require(bool condition, const std::string &msg);
    template <typename T> std::string str(T t);

    template <typename T> void print_mat(const cv::Mat &m);
    void convert_to_disparity_visualize(const cv::Mat &source, cv::Mat &dest, bool color=false);
    void convert_to_disparity_visualize(const cv::Mat &source, cv::Mat &dest,
                                        double minv, double maxv, bool color=false);

    void draw_horiz_lines(cv::Mat &img, int step, int thickness);

    // Prints opencv Mat that contains floats with 1 decimal digit precision
    void print_mat_float(const cv::Mat &m);
    void count_disparities(const cv::Mat &disparity, std::vector<int> &counts, int min_d, int max_d);
    void print_counts(const std::vector<int> &counts, int min_d, int max_d);

    uint64_t rdtsc();
}

#endif
