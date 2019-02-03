cmake_minimum_required(VERSION 3.11)

project(calibration)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Debug)

find_package(OpenCV REQUIRED)

add_executable(run_dlt src/run_dlt.cpp src/utils.cpp src/calibrator.cpp)
add_executable(run_ransac_dlt src/run_ransac_dlt.cpp src/utils.cpp src/calibrator.cpp)
add_executable(run_undistorted_dlt src/run_undistorted_dlt.cpp src/utils.cpp src/calibrator.cpp)

target_link_libraries(run_dlt ${OpenCV_LIBS})
target_link_libraries(run_ransac_dlt ${OpenCV_LIBS})
target_link_libraries(run_undistorted_dlt ${OpenCV_LIBS})
