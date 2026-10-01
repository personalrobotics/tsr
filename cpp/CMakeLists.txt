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

add_library(sstsr_cpp
  src/transform.cpp
  src/tsr.cpp
)
add_library(sstsr::sstsr_cpp ALIAS sstsr_cpp)
set_target_properties(sstsr_cpp PROPERTIES EXPORT_NAME sstsr_cpp POSITION_INDEPENDENT_CODE ON)
target_compile_features(sstsr_cpp PUBLIC cxx_std_20)
target_include_directories(sstsr_cpp PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
if(NOT MSVC)
  target_compile_options(sstsr_cpp PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

if(SSTSR_CPP_BUILD_TESTS)
  enable_testing()

  # The corpus is the Python's recorded answers. A build tree that cannot see it still
  # builds and runs every other test; only this one is skipped, and it says so.
  set(SSTSR_CONFORMANCE_CORPUS "${CMAKE_CURRENT_SOURCE_DIR}/../tests/reference/tsr_conformance.json"
      CACHE FILEPATH "The conformance corpus the C++ is checked against")

  foreach(case_name tsr conformance)
    add_executable(test_${case_name} tests/test_${case_name}.cpp)
    target_link_libraries(test_${case_name} PRIVATE sstsr_cpp)
    target_include_directories(test_${case_name} PRIVATE tests)
    if(NOT MSVC)
      target_compile_options(test_${case_name} PRIVATE -Wall -Wextra -Wpedantic -Werror)
    endif()
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
