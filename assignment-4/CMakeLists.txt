cmake_minimum_required(VERSION 3.5.11)
project(grab-cut)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Debug)

find_package(OpenCV REQUIRED)

add_executable(main src/main.cpp src/grabcut.cpp src/graph.cpp src/maxflow.cpp)

target_link_libraries(main ${OpenCV_LIBS})
