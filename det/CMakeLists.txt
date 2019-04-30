cmake_minimum_required(VERSION 3.4)
project(MTCNN)
add_definitions(-DCPU_ONLY)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV)

# minicaffe
#include_directories(/home/wencc/anaconda3/include)
include_directories(/home/wencc/JDProject/MTCNN/mini-caffe/build/include)
include_directories(/home/wencc/JDProject/MTCNN/mini-caffe/include)
#link_directories(/homew/wencc/ananconda3/lib)
link_directories(/home/wencc/JDProject/MTCNN/mini-caffe/build)

set(SOURCE_FILES main.cpp MTCNN.cpp MTCNN.h)
add_executable(MTCNN ${SOURCE_FILES})

target_link_libraries(MTCNN ${OpenCV_LIBS} )
target_link_libraries(MTCNN caffe)
