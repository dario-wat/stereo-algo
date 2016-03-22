cmake_minimum_required(VERSION 2.8)
project(algos)

# Compile manually with this for debugging and coverage
#g++ *.cpp -std=c++11 -O3 -pg -fprofile-arcs -ftest-coverage `pkg-config --cflags --libs opencv`

# Compile with vectorizing shown
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3 -ftree-vectorizer-verbose=1")

# Normal compile
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3")

find_package(OpenCV REQUIRED)

include_directories(include)

add_executable(algost
  src/main.cpp
  src/st_util.cpp
  src/wta.cpp
  #src/FiveRegionStereo.cpp
  #src/DisparityPropagationStereo.cpp
  #src/IDRStereo.cpp
  src/SADBoxMedian.cpp
  #src/FeatureLinkStereo.cpp
)

target_link_libraries(algost ${OpenCV_LIBS})
