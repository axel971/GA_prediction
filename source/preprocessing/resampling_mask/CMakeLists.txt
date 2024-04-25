

cmake_minimum_required(VERSION 2.5)

project(resampling_mask)

# Find ITK
find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

add_executable(resampling_mask ./source/resample_mask.cpp)

target_link_libraries(resampling_mask ${ITK_LIBRARIES})