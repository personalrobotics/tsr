# SPDX-License-Identifier: BSD-2-Clause
# Authors: Siddhartha Srinivasa and contributors to TSR
#
# The C++ pose region, and the CMake package a planner links against. See docs/CPP.md.
#
# Standard library only, on purpose: a C++ consumer that wants a TSR should not inherit a
# linear-algebra dependency to hold a 4x4. That is also what lets the exported package
# config carry no find_dependency at all.
cmake_minimum_required(VERSION 3.16)
project(sstsr_cpp VERSION 3.2.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(SSTSR_CPP_BUILD_TESTS "Build the C++ tests" ON)
option(SSTSR_CPP_INSTALL "Install the headers and the CMake package" ON)

include(GNUInstallDirs)

# The warning set every target here is built with. -Werror is deliberate: this is a
# reference implementation and a warning is a defect.
#
# The one exception is GCC's -Warray-bounds. GCC 12 and 13 report a false positive inside
# std::vector's copy, inlined at -O2, when the element is a 48-byte std::array<double, 6> --
# which is exactly `ChainCoords`, one XyzRpy per link. The diagnostic points at
# __builtin_memmove in <bits/stl_algobase.h> rather than at any of our code, and names an
# offset at the end of a correctly sized element. It is scoped to GCC rather than silenced
# outright, so the macOS CI job still compiles every one of these files with -Warray-bounds
# active and the check keeps its teeth on the project.
set(SSTSR_CPP_WARNINGS -Wall -Wextra -Wpedantic -Werror)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  list(APPEND SSTSR_CPP_WARNINGS -Wno-array-bounds)
endif()
if(MSVC)
  set(SSTSR_CPP_WARNINGS "")
endif()

add_library(sstsr_cpp
  src/transform.cpp
  src/tsr.cpp
  src/tsr_chain.cpp
)
add_library(sstsr::sstsr_cpp ALIAS sstsr_cpp)
set_target_properties(sstsr_cpp PROPERTIES EXPORT_NAME sstsr_cpp POSITION_INDEPENDENT_CODE ON)
target_compile_features(sstsr_cpp PUBLIC cxx_std_20)
target_include_directories(sstsr_cpp PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_compile_options(sstsr_cpp PRIVATE ${SSTSR_CPP_WARNINGS})

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

if(SSTSR_CPP_BUILD_TESTS)
  enable_testing()

  # The corpus is the Python's recorded answers. A build tree that cannot see it still
  # builds and runs every other test; only this one is skipped, and it says so.
  set(SSTSR_CONFORMANCE_CORPUS "${CMAKE_CURRENT_SOURCE_DIR}/../tests/reference/tsr_conformance.json"
      CACHE FILEPATH "The conformance corpus the C++ is checked against")

  foreach(case_name tsr tsr_chain conformance)
    add_executable(test_${case_name} tests/test_${case_name}.cpp)
    target_link_libraries(test_${case_name} PRIVATE sstsr_cpp)
    target_include_directories(test_${case_name} PRIVATE tests)
    target_compile_options(test_${case_name} PRIVATE ${SSTSR_CPP_WARNINGS})
    add_test(NAME ${case_name} COMMAND test_${case_name})
  endforeach()

  target_compile_definitions(test_conformance PRIVATE
    SSTSR_CONFORMANCE_CORPUS="${SSTSR_CONFORMANCE_CORPUS}")
endif()

# ---------------------------------------------------------------------------
# The installed CMake package
# ---------------------------------------------------------------------------
#
# Two packagings of the same headers, and they must stay in step:
#   * `cmake --install` of this directory, exporting a real target;
#   * the wheel, where hatch_build.py renders a relocatable sstsr_cppTargets.cmake
#     beside a copy of the headers.
# Both are consumed through the one cmake/sstsr_cppConfig.cmake.in below, and both are
# exercised by cpp/examples/{consumer,wheel_consumer}.

if(SSTSR_CPP_INSTALL)
  include(CMakePackageConfigHelpers)

  install(DIRECTORY include/sstsr DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
  install(TARGETS sstsr_cpp EXPORT sstsr_cppTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  )
  install(EXPORT sstsr_cppTargets
    NAMESPACE sstsr::
    FILE sstsr_cppTargets.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/sstsr_cpp
  )
  configure_package_config_file(cmake/sstsr_cppConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/sstsr_cppConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/sstsr_cpp
  )
  write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/sstsr_cppConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
  )
  install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/sstsr_cppConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/sstsr_cppConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/sstsr_cpp
  )
endif()
