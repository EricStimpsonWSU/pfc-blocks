cmake_minimum_required(VERSION 3.25)

# Project name and language
project(helloffft LANGUAGES CXX)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add the executable
add_executable(helloffft main.cpp)

# Link the FFTW library
target_include_directories(helloffft PRIVATE ${CMAKE_SOURCE_DIR})
target_link_directories(helloffft PRIVATE ${CMAKE_SOURCE_DIR}/libs)
target_link_libraries(helloffft PRIVATE fftw3)