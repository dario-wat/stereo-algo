#include "st_util.h"

#include <string>
#include <sstream>
#include <stdexcept>

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