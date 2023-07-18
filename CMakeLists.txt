cmake_minimum_required(VERSION 3.10)
project(demo)

set(CMAKE_CXX_STANDARD 14)
SET(CMAKE_BUILD_TYPE Release)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3 -march=native ")
find_package(OpenCV REQUIRED)

#set(ncnn_DIR "/home/c211/lyc/ncnn/alexnet_demo/ncnn/build/install/lib/cmake/ncnn" CACHE PATH "Directory that contains ncnnConfig.cmake")
#set(ncnn_DIR "/home/moi/APP/include/ncnn/lib/cmake/ncnn"
#        CACHE PATH "Directory that contains ncnnConfig.cmake")


set(MNN "/home/moi/APP/include/mnn" CACHE STRING "MNN library path")
include_directories(${MNN}/include)
link_directories(${MNN}/lib)

add_executable(demo main2.cpp tracking.cpp net.cpp common.cpp)
target_link_libraries(demo
        MNN
        MNN_Express
        ${OpenCV_LIBS}
        -fopenmp
        )
