# CMake version
cmake_minimum_required(VERSION 3.10)

# Declare project
project(GI-chip)

# Set c++ standard
set(CMAKE_CXX_STANDARD 14)

# Set output directories
set(DEFAULT_CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)

# 3rd parties
include_directories(Libraries/random/include)
include_directories(Libraries/eigen)

# Includes
include_directories(Includes)

# Project module
add_subdirectory(Examples)
