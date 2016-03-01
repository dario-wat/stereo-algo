cmake_minimum_required(VERSION 2.8)
project(algos)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3")

find_package(OpenCV REQUIRED)
add_executable(algost main.cpp)
target_link_libraries(algost ${OpenCV_LIBS})
