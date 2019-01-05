cmake_minimum_required(VERSION 3.5.1)
project(assignment-0)

set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_CXX_FLAGS "-lstdc++fs")

find_package(OpenCV REQUIRED)

add_executable(video_to_images video_to_images.cpp)
add_executable(images_to_video images_to_video.cpp)
#add_executable(capture_images capture_images.cpp)
#add_executable(chroma_keying chroma_keying.cpp)

target_link_libraries(video_to_images ${OpenCV_LIBS})
target_link_libraries(images_to_video ${OpenCV_LIBS} stdc++fs)
#target_link_libraries(capture_images ${OpenCV_LIBS})
#target_link_libraries(chroma_keying ${OpenCV_LIBS})
