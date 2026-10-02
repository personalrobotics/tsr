# SPDX-License-Identifier: BSD-2-Clause
# Authors: Siddhartha Srinivasa and contributors to TSR
#
# The wheel's targets file, hand-written rather than exported, because a wheel is
# relocatable: site-packages is wherever the venv happens to be, so every path here is
# resolved relative to this file. hatch_build.py copies it in as sstsr_cppTargets.cmake.
#
# The wheel ships headers and sources, not a compiled library -- a pure-Python wheel
# cannot carry a platform binary -- so the target compiles them into the consumer. That
# is also why it is an INTERFACE target here and a real one in the source install.

if(TARGET sstsr::sstsr_cpp)
  return()
endif()

get_filename_component(_sstsr_cpp_root "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
set(_sstsr_cpp_include "${_sstsr_cpp_root}/include")
set(_sstsr_cpp_src "${_sstsr_cpp_root}/src")

if(NOT EXISTS "${_sstsr_cpp_include}/sstsr/tsr.hpp")
  message(FATAL_ERROR
    "sstsr_cpp: headers are missing from ${_sstsr_cpp_include}. "
    "The wheel is incomplete or was built from a tree without cpp/; reinstall sstsr.")
endif()

add_library(sstsr::sstsr_cpp INTERFACE IMPORTED)
set_target_properties(sstsr::sstsr_cpp PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES "${_sstsr_cpp_include}"
  INTERFACE_SOURCES "${_sstsr_cpp_src}/transform.cpp;${_sstsr_cpp_src}/tsr.cpp;${_sstsr_cpp_src}/tsr_chain.cpp"
  INTERFACE_COMPILE_FEATURES "cxx_std_20"
)
