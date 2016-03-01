#ifndef STEREO_ALGO_UTIL_H_
#define STEREO_ALGO_UTIL_H_

#include <string>

namespace su {
    void require(bool condition, const std::string &msg);
    template <typename T> std::string str(T t);
}

#endif