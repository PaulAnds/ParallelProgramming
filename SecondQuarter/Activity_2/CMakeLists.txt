cmake_minimum_required(VERSION 3.26)
project(ImageProcesser)

set(CMAKE_CXX_STANDARD 17)

add_executable(ImageProcesser main.cpp)

find_package(OpenCV REQUIRED)
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
target_link_libraries(ImageProcesser ${OpenCV_LIBS})