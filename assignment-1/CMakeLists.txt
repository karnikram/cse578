cmake_minimum_required(VERSION 3.11)

project(calibration)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Debug)

find_package(OpenCV REQUIRED)

add_executable(main src/main.cpp src/utils.cpp src/calibrator.cpp)
add_executable(opencv_imagelist_creator src/opencv_imagelist_creator.cpp)
add_executable(opencv_calib src/opencv_calibration.cpp)

target_link_libraries(main ${OpenCV_LIBS})
target_link_libraries(opencv_imagelist_creator ${OpenCV_LIBS})
target_link_libraries(opencv_calib ${OpenCV_LIBS})
