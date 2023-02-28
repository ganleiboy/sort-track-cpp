cmake_minimum_required(VERSION 3.0.2)
project(sort_cpp)

## 设定编译参数
set(CMAKE_BUILD_TYPE "Debug")  # Debug or Release
set(CMAKE_CXX_FLAGS "-std=c++14")

# OpenCV
find_package(OpenCV REQUIRED)
if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV not found")
endif()

# 设定头文件路径
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# 设定源码列表,查找指定目录(都放在./src/中)中的所有源文件,并将名称保存到 DIR_SRCS 变量中
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/src/ DIR_SRC)

## Declare a C++ executable
add_executable(main ${DIR_SRC})
target_link_libraries(main ${OpenCV_LIBS})