cmake_minimum_required(VERSION 3.10)
project(robocup_infer)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Pass OpenCV_DIR or CMAKE_PREFIX_PATH on the command line when the libraries
# are installed in a non-standard location. Do not hard-code a development PC.
find_package(OpenCV REQUIRED)
find_package(ncnn REQUIRED)
find_package(Threads REQUIRED)

add_executable(robocup_infer main.cpp)
target_include_directories(robocup_infer PRIVATE ${OpenCV_INCLUDE_DIRS})

target_link_libraries(robocup_infer
    ncnn
    ${OpenCV_LIBS}
    Threads::Threads
)
