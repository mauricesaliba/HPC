cmake_minimum_required(VERSION 3.7)
project(npp_01)

set(CMAKE_CXX_STANDARD 11)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fopenmp")

set(SOURCE_FILES nbody.cpp vector2.h)
add_executable(npp_01 ${SOURCE_FILES})