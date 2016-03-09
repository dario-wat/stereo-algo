cmake_minimum_required(VERSION 2.8)
project(algos)

set(USE_PROFILER "-fprofile-arcs -ftest-coverage -pg")
#set(USE_PROFILER "")
#g++ *.cpp -std=c++11 -O3 -fprofile-arcs -ftest-coverage `pkg-config --cflags --libs opencv`
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3")
find_package(OpenCV REQUIRED)

add_executable(algost
  main.cpp
  st_util.cpp
  FiveRegionStereo.cpp
  DisparityPropagationStereo.cpp
)

target_link_libraries(algost ${OpenCV_LIBS})
