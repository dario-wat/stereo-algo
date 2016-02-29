#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

int main(int argc, char **argv) {

    cv::Mat img = cv::imread(argv[1]);
    cv::imshow("Image", img);
    cv::waitKey(0);
    return 0;
}