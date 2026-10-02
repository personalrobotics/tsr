# SPDX-License-Identifier: BSD-2-Clause
# Authors: Siddhartha Srinivasa and contributors to TSR
#
# Consumes the package produced by `cmake --install` of cpp/. Driven by smoke.sh.
cmake_minimum_required(VERSION 3.16)
project(sstsr_cpp_consumer LANGUAGES CXX)

find_package(sstsr_cpp CONFIG REQUIRED)

add_executable(consumer main.cpp)
target_link_libraries(consumer PRIVATE sstsr::sstsr_cpp)
