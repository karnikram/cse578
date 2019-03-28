cmake_minimum_required(VERSION 3.5.11)
project(grab-cut)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Debug)

find_package(OpenCV REQUIRED)
find_package(Boost 1.45.0 COMPONENTS filesystem)

include_directories(${Boost_INCLUDE_DIRS})
add_executable(run src/run.cpp src/grabcut.cpp src/graph.cpp src/maxflow.cpp)

target_link_libraries(run ${OpenCV_LIBS} ${Boost_LIBRARIES})
