#include "FiveRegionStereo.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "st_util.h"
#include <iostream>

// Some functions that optimize the execution by pointing the compiler to vectorize the loops.

inline void vectorize_add(int *a, int *b, int n) {
    for (int i = 0; i < n; i++) {
        a[i] += b[i];
    }
}

inline void vectorize_subtract_mem(int *a, int *b, int *c, int n) {
    for (int i = 0; i < n; i++) {
        a[i] = b[i] - c[i];
    }
}

inline void vectorize_sub_3idx(unsigned char *a, unsigned char *b, int *c, int n, int il, int ir)  {
    for (int i = 0; i < n; i++, il++, ir++) {
        c[i] = (int)a[il] - b[ir];
    }
}

// using fabsf since abs doesn't work. I get about 5-10% speedup with this vectorization
inline void vectorize_abs(int *a, int n) {
    for (int i = 0; i < n; i++) {
        a[i] = fabsf(a[i]);
    }
}

FiveRegionStereo::FiveRegionStereo( int min_disparity, int max_disparity, int radiusX, int radiusY,
                                    int max_per_pixel_error, int validate_RtoL, double texture) {
    // Some requirements for parameters
    su::require(max_disparity > 0, "Max disparity must be greater than 0");
    su::require(min_disparity >= 0 && min_disparity < max_disparity,
                "Min disparity must be >=0 and < than max disparity");
    su::require(radiusX > 0, "radiusX must be greater than 0");
    su::require(radiusY > 0, "radiusY must be greater than 0");

    // Set all the parameters
    this->min_disparity = min_disparity;
    this->max_disparity = max_disparity;
    this->radiusX = radiusX;
    this->radiusY = radiusY;
    this->validate_RtoL = validate_RtoL;
    this->texture = texture;
    this->region_height = radiusY*2 + 1;
    this->region_width = radiusX*2 + 1;
    this->max_error = region_height * region_width * max_per_pixel_error * 3;
    this->invalid_disparity = max_disparity - min_disparity + 1;
    
    // Aux arrays
    this->column_score = new int[max_disparity];
    this->element_score = NULL;
    this->five_score = NULL;
    this->vertical_score = NULL;
    this->horizontal_score = NULL;
}

FiveRegionStereo::~FiveRegionStereo() {
    delete column_score;
    if (element_score != NULL) {
        delete element_score;
    }
    if (five_score != NULL) {
        delete five_score;
    }
}

void FiveRegionStereo::configure(int width) {
    if (width == image_width) {
        return;     // everything is already setup properly
    }
    
    su::require(max_disparity <= width - 2*radiusX,
                "The maximum disparity is too large for this image size: max size "
                    + su::str(width - 2*radiusX));
    
    this->length_horizontal = width * (max_disparity - min_disparity);
    this->image_width = width;

    // Reinitialize matrices
    if (horizontal_score != NULL) {
        delete horizontal_score;
    }
    this->horizontal_score = new int[region_height*length_horizontal];
    if (vertical_score != NULL) {
        delete vertical_score;
    }
    this->vertical_score = new int[region_height*length_horizontal];
    
    // Reinitialize arrays with different sizes
    if (element_score != NULL) {        // TODO i dont like this
        delete element_score;
    }
    this->element_score = new int[width];
    if (five_score != NULL) {
        delete five_score;
    }
    this->five_score = new int[length_horizontal];
}



inline void FiveRegionStereo::compute_score_row_sad(int element_max, int index_left,
                                                    int index_right) {
    vectorize_sub_3idx(left.data, right.data, element_score, element_max, index_left, index_right);
    vectorize_abs(element_score, element_max);
}

void FiveRegionStereo::compute_score_row(int row, int *scores) {
    // disparity as the outer loop to maximize common elements in inner loops,
    // reducing redundant calculations
    for (int d = min_disparity; d < max_disparity; d++) {
        int disp_from_min = d - min_disparity;

        // number of individual columns the error is computed in
        int col_max = image_width - d;
        // number of regions that a score/error is computed in
        int score_max = col_max - region_width;

        // indices that data is read to/from for different data structures
        int index_score = image_width*disp_from_min + disp_from_min;
        int index_left = image_width*row + d;
        int index_right = image_width*row;

        // Fill elementScore with scores for individual elements for this row at disparity d
        compute_score_row_sad(col_max, index_left, index_right);

        // score at the first column
        // TODO maybe extract into a function
        int score = 0;
        for (int i = 0; i < region_width; i++) {
            score += element_score[i];
        }
 
        scores[index_score] = score;
        index_score++;

        for (int col = 0; col < score_max; col++, index_score++) {
            score += element_score[col+region_width] - element_score[col];
            scores[index_score] = score;
        }
    }
}

void FiveRegionStereo::compute_first_row() {
    int *first_row = vertical_score;
    // std::cerr << first_row << std::endl;
    // std::cerr << vertical_score.ptr<int>(1) << std::endl;
    // std::cerr << vertical_score.ptr<int>(2) << std::endl;
    std::cerr << first_row << std::endl;
    std::cerr << vertical_score + image_width * 1 << std::endl;
    std::cerr << vertical_score + image_width * 2 << std::endl;
    active_vertical_score = 1;
    
    // compute horizontal scores for first row block
    for (int row = 0; row < region_height; row++) {
        int *scores = horizontal_score + length_horizontal * (row);
        compute_score_row(row, scores);
    }
    
    // compute score for the top possible row
    for (int i = 0; i < length_horizontal; i++) {
        int sum = 0;
        for (int row = 0; row < region_height; row++) {
            sum += horizontal_score[row*length_horizontal + i];
        }
        first_row[i] = sum;
    }
}


// TODO make this nicer
/**
     * Compute the final score by sampling the 5 regions.  Four regions are sampled around the center
     * region.  Out of those four only the two with the smallest score are used.
     */
void FiveRegionStereo::compute_score_five(int *top, int *middle, int *bottom, int *score) {
    // disparity as the outer loop to maximize common elements in inner loops, reducing redundant calculations
    for (int d = min_disparity; d < max_disparity; d++) {

        // take in account the different in image border between the sub-regions and the effective region
        int index_src = (d-min_disparity)*image_width + (d-min_disparity) + radiusX;
        int index_dst = (d-min_disparity)*image_width + (d-min_disparity);
        int end = index_src + (image_width-d-4*radiusX);
        while (index_src < end) {
            int s = 0;

            // sample four outer regions at the corners around the center region
            int val0 = top[index_src-radiusX];
            int val1 = top[index_src+radiusX];
            int val2 = bottom[index_src-radiusX];
            int val3 = bottom[index_src+radiusX];

            // select the two best scores from outer for regions
            if (val1 < val0) {
                int temp = val0;
                val0 = val1;
                val1 = temp;
            }

            if (val3 < val2) {
                int temp = val2;
                val2 = val3;
                val3 = temp;
            }

            if (val3 < val0) {
                s += val2;
                s += val3;
            } else if (val2 < val1) {
                s += val2;
                s += val0;
            } else {
                s += val0;
                s += val1;
            }

            score[index_dst++] = s + middle[index_src++];
        }
    }
}

int FiveRegionStereo::max_disparity_at_column_L2R(int col) {
    return 1+col-min_disparity-std::max(0,col-max_disparity+1);
}

int FiveRegionStereo::select_right_to_left(int col, int *scores, int region_width) {
    // see how far it can search
    int local_max = std::min(image_width-region_width, col+max_disparity) - col-min_disparity;

    int index_best = 0;
    int index_score = col;
    int score_best = scores[col];
    index_score += image_width+1;

    for (int i = 1; i < local_max; i++, index_score += image_width+1) {
        int s = scores[index_score];
        if (s < score_best) {
            score_best = s;
            index_best = i;

        }
    }
    return index_best;
}


void FiveRegionStereo::process(int row, int* scores, cv::Mat image_disparity, int radiusX, int region_width) {
    
    int index_disparity = 0 + row*image_disparity.cols + radiusX + min_disparity;
    // cerr << 0 << ' ' << row << ' ' << image_disparity.cols << ' ' << radiusX << ' ' << min_disparity << endl;
    for (int col = min_disparity; col <= image_width-region_width; col++) {
        // Determine the number of disparities that can be considered at this column
        // make sure the disparity search doesn't go outside the image border
        int local_max = max_disparity_at_column_L2R(col);

        // index of the element being examined in the score array
        int index_score = col - min_disparity;

        // select the best diparity
        int best_disparity = 0;
        int score_best = column_score[0] = scores[index_score];
        index_score += image_width;

        for (int i = 1; i < local_max; i++, index_score += image_width) {
            int s = scores[index_score];
            column_score[i] = s;
            if (s < score_best) {
                score_best = s;
                best_disparity = i;
            }
        }

        // detect bad matches
        if (score_best > max_error) {
            // make sure the error isnt too large
            best_disparity = invalid_disparity;
        } else if (validate_RtoL >= 0) {
            // if the associate is different going the other direction it is probably noise
            int disparity_RtoL = select_right_to_left(col-best_disparity-min_disparity, scores, region_width);
            if (abs(disparity_RtoL-best_disparity) > validate_RtoL) {
                best_disparity = invalid_disparity;
            }
        }

        // test to see if the region lacks sufficient texture if:
        // 1) not already eliminated 2) sufficient disparities to check, 3) it's activated
        int texture_threshold = texture*10000;
        if (texture_threshold > 0 && best_disparity != invalid_disparity && local_max >= 3) {
            // find the second best disparity value and exclude its neighbors
            int second_best = 10000000; // integer max
            for (int i = 0; i < best_disparity-1; i++) {
                if (column_score[i] < second_best) {
                    second_best = column_score[i];
                }
            }
            for (int i = best_disparity+2; i < local_max; i++) {
                if (column_score[i] < second_best) {
                    second_best = column_score[i];
                }
            }

            // similar scores indicate lack of texture
            // C = (C2-C1)/C1
            // TODO this looks fishy
            if (10000*(second_best-score_best) <= texture_threshold*score_best) {
                best_disparity = invalid_disparity;
            }
        }

        // S32_F32#setDisparity
        if( best_disparity <= 0 || best_disparity >= local_max-1) {
            ((float*)image_disparity.data)[index_disparity] = best_disparity;
            // image_disparity.at<float>(row, col) = best_disparity;
        } else {
            int c0 = column_score[best_disparity-1];
            int c1 = column_score[best_disparity];
            int c2 = column_score[best_disparity+1];

            float offset = (float)(c0-c2)/(float)(2*(c0-2*c1+c2));

            ((float*)image_disparity.data)[index_disparity] = best_disparity + offset;
            // image_disparity.at<float>(row, col) = best_disparity + offset;
        }
        index_disparity++;
    }
}


#include <cstdio>
void print_mat(cv::Mat m) {
    for (int i = 0; i < m.rows; i++) {
        for (int j = 0; j < m.cols; j++) {
            printf("%.1f ", m.at<float>(i, j));
        }
        std::cout << std::endl;
    }
}



// ImplDisparityScoreSadRectFive_U8#computeRemainingRows
    // efficiently compute rest of the rows using previous results to avoid repeat computations
    /**
     * Using previously computed results it efficiently finds the disparity in the remaining rows.
     * When a new block is processes the last row/column is subtracted and the new row/column is
     * added.
     */
void FiveRegionStereo::compute_remaining_rows() {
    cv::Mat disparity = cv::Mat::ones(left.rows, left.cols, CV_32FC1) * (max_disparity+1);
    for (int row = region_height; row < left.rows; row++, active_vertical_score++) {
        int old_row = row % region_height;
        int *previous = vertical_score + length_horizontal *((active_vertical_score-1) % region_height);
        int *active = vertical_score + length_horizontal *((active_vertical_score) % region_height);

        // subtract first row from vertical score
        int *scores = horizontal_score + length_horizontal * (old_row);
        vectorize_subtract_mem(active, previous, scores, length_horizontal);

        compute_score_row(row, scores);

        // add the new score
        vectorize_add(active, scores, length_horizontal);

        if (active_vertical_score >= region_height-1) {
            int *top = vertical_score + length_horizontal * ((active_vertical_score - 2*radiusY) % region_height);
            int *middle = vertical_score + length_horizontal * ((active_vertical_score - radiusY) % region_height);
            int *bottom = vertical_score + length_horizontal * (active_vertical_score % region_height);

            compute_score_five(top, middle, bottom, five_score);

            // ImplSelectRectStandardBase_S32#process
            process(row - (1 + 4*radiusY) + 2*radiusY+1, five_score, disparity, radiusX*2, radiusX*4+1);
        }
    }

    print_mat(disparity);
    // disparity.convertTo(disparity, CV_8UC1);
    // cv::imshow("wind", disparity);
    // cv::waitKey(0);
}

void FiveRegionStereo::compute_disparity(const cv::Mat &left, const cv::Mat &right) {
    //TODO check shape
    this->left = left;
    this->right = right;
    std::cerr << 'A' << std::endl;
    configure(left.cols);
    std::cerr << 'A' << std::endl;
    compute_first_row();
    std::cerr << 'A' << std::endl;
    compute_remaining_rows();
    std::cerr << 'A' << std::endl;
}