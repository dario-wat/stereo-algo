#include "st_util.h"

#include <string>
#include <sstream>
#include <stdexcept>
#include <cstdio>
#include <iostream>
#include <opencv2/core/core.hpp>

void su::require(bool condition, const std::string &msg) {
    if (!condition) {
        throw std::invalid_argument(msg);
    }
}

template <typename T>
std::string su::str(T t) {
    std::stringstream ss;
    ss << t;
    return ss.str();
}

// Explicit instantiation for su::str function
template std::string su::str<int>(int t);

template <typename T>
void su::print_mat(const cv::Mat &m) {
    for (int i = 0; i < m.rows; i++) {
        for (int j = 0; j < m.cols; j++) {
            std::cout << m.at<T>(i,j) << ' ';
        }
        std::cout << std::endl;
    }
}

// Explicits instantiation for su::print_mat
template void su::print_mat<float>(const cv::Mat &m);

void su::convert_to_disparity_visualize(const cv::Mat &source, cv::Mat &dest) {
    double minv, maxv;
    cv::minMaxLoc(source, &minv, &maxv);
    dest = (source - minv) * 255.0 / (maxv - minv);
    dest.convertTo(dest, CV_8UC1);
}

void su::print_mat_float(const cv::Mat &m) {
    for (int i = 0; i < m.rows; i++) {
        for (int j = 0; j < m.cols; j++) {
            printf("%.1f ", m.at<float>(i, j));
        }
        printf("\n");
    }
}
