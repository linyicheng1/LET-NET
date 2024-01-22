cmake_minimum_required(VERSION 3.10)
project(demo)

set(CMAKE_CXX_STANDARD 14)
SET(CMAKE_BUILD_TYPE Release)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3 -march=native ")

find_package(OpenCV REQUIRED)
find_package(glog)
set(MNN "/home/moi/APP/include/mnn2" CACHE STRING "MNN library path")
include_directories(${MNN}/include)
link_directories(${MNN}/lib)

add_executable(demo main.cpp net.cpp)
target_link_libraries(demo
        MNN
        MNN_Express
        ${OpenCV_LIBS}
        -fopenmp
        )
add_executable(demo2 main2.cpp net.cpp)
target_link_libraries(demo2
    MNN
    MNN_Express
    ${OpenCV_LIBS}
    -fopenmp
)
