#include <iostream>
#include <cstdio>
#include <iomanip>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <ctime>
#include "FiveRegionStereo.h"

using std::cout;
using std::endl;
using std::cerr;

// void print_array(int *a, int n) {
//     for (int i = 0; i < n; i++) {
//         std::cout << a[i] << ' ';
//     }
//     std::cout << std::endl;
// }

// void print_mat(cv::Mat m) {
//     for (int i = 0; i < m.rows; i++) {
//         for (int j = 0; j < m.cols; j++) {
//             printf("%.1f ", m.at<float>(i, j));
//         }
//         cout << endl;
//     }
// }

// /**
//      * compute the score for each element all at once to encourage the JVM to optimize and
//      * encourage the JVM to optimize this section of code.
//      *
//      * Was original inline, but was actually slightly slower by about 3% consistently,  It
//      * is in its own function so that it can be overridden and have different cost functions
//      * inserted easily.
//      */
// // TODO fix this function to use Mat calls instead
// void compute_score_row_sad(cv::Mat left, cv::Mat right, int element_max, int index_left,
//         int index_right, int *element_score) {
//     for (int r_col = 0; r_col < element_max; r_col++) {
//         int diff = (left.data[index_left++] & 0xFF) - (right.data[index_right++] & 0xFF);
//         element_score[r_col] = abs(diff);
//     }
// }

// void compute_score_row(cv::Mat left, cv::Mat right, int row, int *scores, int minDisparity,
//         int maxDisparity, int regionWidth, int *element_score) {
//     // disparity as the outer loop to maximize common elements in inner loops, reducing redundant calculations
//     for (int d = minDisparity; d < maxDisparity; d++) {
//         int disp_from_min = d - minDisparity;

//         // number of individual columns the error is computed in
//         int col_max = left.cols - d;
//         // number of regions that a score/error is computed in
//         int score_max = col_max - regionWidth;

//         // indices that data is read to/from for different data structures
//         int index_score = left.cols*disp_from_min + disp_from_min;
//         int index_left = 0 + left.cols*row + d;
//         int index_right = 0 + right.cols*row;

//         // Fill elementScore with scores for individual elements for this row at disparity d
//         compute_score_row_sad(left, right, col_max, index_left, index_right, element_score);

//         // score at the first column
//         int score = 0;
//         for (int i = 0; i < regionWidth; i++) {
//             score += element_score[i];
//         }

//         scores[index_score++] = score;

//         for( int col = 0; col < score_max; col++ , index_score++ ) {
//             scores[index_score] = score += element_score[col+regionWidth] - element_score[col];
//         }
//     }
// }

// /**
//      * Compute the final score by sampling the 5 regions.  Four regions are sampled around the center
//      * region.  Out of those four only the two with the smallest score are used.
//      */
// void compute_score_five(int *top, int *middle, int *bottom, int *score, int width,
//         int minDisparity, int maxDisparity, int radiusX) {
//     // disparity as the outer loop to maximize common elements in inner loops, reducing redundant calculations
//     for (int d = minDisparity; d < maxDisparity; d++) {

//         // take in account the different in image border between the sub-regions and the effective region
//         int index_src = (d-minDisparity)*width + (d-minDisparity) + radiusX;
//         int index_dst = (d-minDisparity)*width + (d-minDisparity);
//         int end = index_src + (width-d-4*radiusX);
//         while (index_src < end) {
//             int s = 0;

//             // sample four outer regions at the corners around the center region
//             int val0 = top[index_src-radiusX];
//             int val1 = top[index_src+radiusX];
//             int val2 = bottom[index_src-radiusX];
//             int val3 = bottom[index_src+radiusX];

//             // select the two best scores from outer for regions
//             if( val1 < val0 ) {
//                 int temp = val0;
//                 val0 = val1;
//                 val1 = temp;
//             }

//             if( val3 < val2 ) {
//                 int temp = val2;
//                 val2 = val3;
//                 val3 = temp;
//             }

//             if( val3 < val0 ) {
//                 s += val2;
//                 s += val3;
//             } else if( val2 < val1 ) {
//                 s += val2;
//                 s += val0;
//             } else {
//                 s += val0;
//                 s += val1;
//             }

//             score[index_dst++] = s + middle[index_src++];
//         }
//     }
// }

// int max_disparity_at_column_L2R(int col, int minDisparity, int maxDisparity) {
//     return 1+col-minDisparity-std::max(0,col-maxDisparity+1);
// }

// int select_right_to_left(int col, int *scores, int minDisparity, int maxDisparity,
//         int image_width, int region_width) {
//     // see how far it can search
//     int local_max = std::min(image_width-region_width, col+maxDisparity) - col-minDisparity;

//     int index_best = 0;
//     int index_score = col;
//     int score_best = scores[col];
//     index_score += image_width+1;

//     for (int i = 1; i < local_max; i++, index_score += image_width+1) {
//         int s = scores[index_score];
//         if (s < score_best) {
//             score_best = s;
//             index_best = i;

//         }
//     }
//     return index_best;
// }

// void process(int row, int* scores, cv::Mat image_disparity, int radiusX, int minDisparity,
//         int maxDisparity, int image_width, int region_width, int *column_score, int max_error,
//         int right_to_left_tolerance, double texture, int invalid_disparity) {
    
//     int index_disparity = 0 + row*image_disparity.cols + radiusX + minDisparity;
//     // cerr << 0 << ' ' << row << ' ' << image_disparity.cols << ' ' << radiusX << ' ' << minDisparity << endl;
//     for (int col = minDisparity; col <= image_width-region_width; col++) {
//         // Determine the number of disparities that can be considered at this column
//         // make sure the disparity search doesn't go outside the image border
//         int local_max = max_disparity_at_column_L2R(col, minDisparity, maxDisparity);

//         // index of the element being examined in the score array
//         int index_score = col - minDisparity;

//         // select the best diparity
//         int best_disparity = 0;
//         int score_best = column_score[0] = scores[index_score];
//         index_score += image_width;

//         for (int i = 1; i < local_max; i++, index_score += image_width) {
//             int s = scores[index_score];
//             column_score[i] = s;
//             if (s < score_best) {
//                 score_best = s;
//                 best_disparity = i;
//             }
//         }

//         // detect bad matches
//         if (score_best > max_error) {
//             // make sure the error isnt too large
//             best_disparity = invalid_disparity;
//         } else if (right_to_left_tolerance >= 0) {
//             // if the associate is different going the other direction it is probably noise
//             int disparity_RtoL = select_right_to_left(col-best_disparity-minDisparity, scores,
//                     minDisparity, maxDisparity, image_width, region_width);
//             if (abs(disparity_RtoL-best_disparity) > right_to_left_tolerance) {
//                 best_disparity = invalid_disparity;
//             }
//         }

//         // test to see if the region lacks sufficient texture if:
//         // 1) not already eliminated 2) sufficient disparities to check, 3) it's activated
//         int texture_threshold = texture*10000;
//         if (texture_threshold > 0 && best_disparity != invalid_disparity && local_max >= 3) {
//             // find the second best disparity value and exclude its neighbors
//             int second_best = 10000000; // integer max
//             for (int i = 0; i < best_disparity-1; i++) {
//                 if (column_score[i] < second_best) {
//                     second_best = column_score[i];
//                 }
//             }
//             for (int i = best_disparity+2; i < local_max; i++) {
//                 if (column_score[i] < second_best) {
//                     second_best = column_score[i];
//                 }
//             }

//             // similar scores indicate lack of texture
//             // C = (C2-C1)/C1
//             // TODO this looks fishy
//             if (10000*(second_best-score_best) <= texture_threshold*score_best) {
//                 best_disparity = invalid_disparity;
//             }
//         }

//         // S32_F32#setDisparity
//         if( best_disparity <= 0 || best_disparity >= local_max-1) {
//             ((float*)image_disparity.data)[index_disparity] = best_disparity;
//             // image_disparity.at<float>(row, col) = best_disparity;
//         } else {
//             int c0 = column_score[best_disparity-1];
//             int c1 = column_score[best_disparity];
//             int c2 = column_score[best_disparity+1];

//             float offset = (float)(c0-c2)/(float)(2*(c0-2*c1+c2));

//             ((float*)image_disparity.data)[index_disparity] = best_disparity + offset;
//             // image_disparity.at<float>(row, col) = best_disparity + offset;
//         }
//         index_disparity++;
//     }
// }

// void func(cv::Mat left, cv::Mat right, int minDisparity, int maxDisparity, int radiusX,
//         int radiusY, double maxPerPixelError, int validateRtoL, double texture) {

//     if (maxDisparity > left.cols-2*radiusX) {
//         std::cout << "Runtime error here" << std::endl;
//     }

//     int regionHeight = radiusY*2+1;
//     int regionWidth = radiusX*2+1;
//     int max_error = regionHeight*regionWidth*maxPerPixelError*3;

//     int length_horizontal = left.cols * (maxDisparity - minDisparity);

//     // in SelectRectStandard it is image_disparity
//     cv::Mat disparity = cv::Mat::ones(left.rows, left.cols, CV_32FC1) * (maxDisparity+1);

//     // ImplDisparityScoreSadRectFive_U8#_process
//     cv::Mat horizontal_score = cv::Mat(regionHeight, length_horizontal, CV_32SC1);
//     cv::Mat vertical_score = cv::Mat(regionHeight, length_horizontal, CV_32SC1);
//     int *element_score = new int[left.cols];    // TODO I would prefer not to dynamically allocate
//     int *five_score = new int[length_horizontal];


//     // Call to ImplSelectRectStandardBase_S32#configure
//     // Here comes call to SelectRectStandard#configure
//     int invalid_disparity = maxDisparity+1;
//     int *column_score = new int[maxDisparity];
//     int image_width = disparity.cols;






//     // ImplDisparityScoreSadRectFive_U8#computeFirstRow
//     /**
//      * Initializes disparity calculation by finding the scores for the initial block of horizontal
//      * rows.
//      */
//      int *first_row = vertical_score.ptr<int>(0);
//     int active_vertical_score = 1;
//     // compute horizontal scores for first row block
//     for (int row = 0; row < regionHeight; row++) {
//         int *scores = horizontal_score.ptr<int>(row);
//         compute_score_row(left, right, row, scores, minDisparity, maxDisparity, regionWidth, element_score);
//         // print_array(scores, length_horizontal);
//         // exit(1);
//     }
    

//     // compute score for the top possible row
//     for (int i = 0; i < length_horizontal; i++) {
//         int sum = 0;
//         for (int row = 0; row < regionHeight; row++) {
//             sum += horizontal_score.at<int>(row, i);
//         }
//         first_row[i] = sum;
//     }





//     // ImplDisparityScoreSadRectFive_U8#computeRemainingRows
//     // efficiently compute rest of the rows using previous results to avoid repeat computations
//     /**
//      * Using previously computed results it efficiently finds the disparity in the remaining rows.
//      * When a new block is processes the last row/column is subtracted and the new row/column is
//      * added.
//      */
//     for (int row = regionHeight; row < left.rows; row++, active_vertical_score++) {
//         int old_row = row % regionHeight;
//         int *previous = vertical_score.ptr<int>((active_vertical_score-1) % regionHeight);
//         int *active = vertical_score.ptr<int>((active_vertical_score) % regionHeight);

//         // subtract first row from vertical score
//         int *scores = horizontal_score.ptr<int>(old_row);
//         for (int i = 0; i < length_horizontal; i++) {
//             active[i] = previous[i] - scores[i];
//         }

//         compute_score_row(left, right, row, scores, minDisparity, maxDisparity, regionWidth, element_score);

//         // add the new score
//         for (int i = 0; i < length_horizontal; i++) {
//             active[i] += scores[i];
//         }

//         if (active_vertical_score >= regionHeight-1) {
//             int *top = vertical_score.ptr<int>((active_vertical_score - 2*radiusY) % regionHeight);
//             int *middle = vertical_score.ptr<int>((active_vertical_score - radiusY) % regionHeight);
//             int *bottom = vertical_score.ptr<int>(active_vertical_score % regionHeight);

//             compute_score_five(top, middle, bottom, five_score, left.cols, minDisparity, maxDisparity, radiusX);

//             // ImplSelectRectStandardBase_S32#process
//             process(row - (1 + 4*radiusY) + 2*radiusY+1, five_score, disparity, radiusX*2, minDisparity,
//                 maxDisparity, image_width, radiusX*4+1, column_score, max_error,
//                 validateRtoL, texture, invalid_disparity);
//         }
//     }
//     // std::cout << horizontal_score << endl;

//     print_mat(disparity);
//     disparity.convertTo(disparity, CV_8UC1);
//     // cv::imshow("dd", disparity);
//     // cv::waitKey(0);
// }

int main(int argc, char **argv) {

    cv::Mat img_left_g;
    cv::cvtColor(cv::imread(argv[1]), img_left_g, CV_BGR2GRAY);
    cv::Mat img_right_g;
    cv::cvtColor(cv::imread(argv[2]), img_right_g, CV_BGR2GRAY);
    int max_disp = atoi(argv[3]);
    int region_size = atoi(argv[4]);

    FiveRegionStereo frs = FiveRegionStereo(0, max_disp, region_size, region_size, 25, 6, 0.2);
    frs.compute_disparity(img_left_g, img_right_g);
    cerr << "Outside" << endl;
    clock_t begin = clock();
    // func(img_left_g, img_right_g, 0, max_disp, region_size, region_size, 25, 6, 0.2);
    cerr << double(clock()-begin) / CLOCKS_PER_SEC;
    return 0;
}