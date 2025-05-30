cmake_minimum_required(VERSION 3.10)
project(OPENCV_CPP)
find_package(OpenCV REQUIRED)
add_executable(opencv_cpp 
main.cpp
imageDetection.cpp
yolo.cpp
videoDetection.cpp
cameraDetection.cpp
)
target_link_libraries(opencv_cpp ${OpenCV_LIBS})
