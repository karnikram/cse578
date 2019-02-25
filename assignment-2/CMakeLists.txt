cmake_minimum_required(VERSION 3.11)

project(Panaroma)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Debug)

find_package(OpenCV REQUIRED)

add_executable(run src/run.cpp src/panaroma.cpp src/utils.cpp)

target_link_libraries(run ${OpenCV_LIBS})
