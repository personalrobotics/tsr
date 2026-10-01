# SPDX-License-Identifier: BSD-2-Clause
# Authors: Siddhartha Srinivasa and contributors to TSR
#
# Consumes the CMake package carried inside an installed sstsr wheel, found the way a
# planner finds it: by asking the interpreter. Driven by smoke.sh.
cmake_minimum_required(VERSION 3.16)
project(sstsr_cpp_wheel_consumer LANGUAGES CXX)

find_package(Python REQUIRED COMPONENTS Interpreter)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import tsr; print(tsr.get_cmake_dir(), end='')"
  OUTPUT_VARIABLE SSTSR_CMAKE_DIR
  RESULT_VARIABLE _rc
)
if(NOT _rc EQUAL 0 OR NOT SSTSR_CMAKE_DIR)
  message(FATAL_ERROR "the installed sstsr does not carry a CMake package")
endif()
list(APPEND CMAKE_PREFIX_PATH "${SSTSR_CMAKE_DIR}")

# Asking for a version exercises the rendered ConfigVersion file, which is the half of the
# wheel package that nothing else reads.
if(NOT SSTSR_CPP_VERSION)
  message(FATAL_ERROR "pass -DSSTSR_CPP_VERSION=<X.Y.Z>; smoke.sh reads it from the install")
endif()
find_package(sstsr_cpp ${SSTSR_CPP_VERSION} CONFIG REQUIRED)

add_executable(wheel_consumer ../consumer/main.cpp)
target_link_libraries(wheel_consumer PRIVATE sstsr::sstsr_cpp)
