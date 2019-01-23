cmake_minimum_required(VERSION 3.11)

project(calibration)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

add_executable(main src/main.cpp)

target_link_libraries(main ${OpenCV_LIBS})
