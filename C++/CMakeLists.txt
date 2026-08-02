cmake_minimum_required(VERSION 3.10)
project(robocup_infer)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(OpenCV_DIR "D:/Tools/Env/opencv/opencv/mingw-build/install/x64/mingw/lib")
find_package(OpenCV REQUIRED)
find_package(ncnn REQUIRED)
find_package(Threads REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(robocup_infer main.cpp)

target_link_libraries(robocup_infer
    ncnn
    ${OpenCV_LIBS}
    Threads::Threads
)

