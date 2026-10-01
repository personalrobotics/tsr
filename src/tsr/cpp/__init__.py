# SPDX-License-Identifier: BSD-2-Clause
# Authors: Siddhartha Srinivasa and contributors to TSR

"""Where the C++ core's headers and CMake package are, for a C++ consumer to find.

A planner that wants TSRs in C++ asks the installed Python package, rather than being
told a path::

    find_package(Python COMPONENTS Interpreter)
    execute_process(COMMAND "${Python_EXECUTABLE}" -c
                    "import tsr; print(tsr.get_cmake_dir(), end='')"
                    OUTPUT_VARIABLE TSR_CMAKE_DIR)
    list(APPEND CMAKE_PREFIX_PATH "${TSR_CMAKE_DIR}")
    find_package(sstsr_cpp CONFIG REQUIRED)
    target_link_libraries(my_planner PRIVATE sstsr::sstsr_cpp)

See ``docs/CPP.md``. The rules the C++ implements are the Python's, held to them by the
conformance corpus; the one deliberate difference is the sampling engine.
"""

from __future__ import annotations

from pathlib import Path

_HERE = Path(__file__).resolve().parent

#: The header that proves a tree really holds the C++ core rather than an empty directory.
_PROBE = Path("sstsr") / "tsr.hpp"

__all__ = ["get_cmake_dir", "get_include"]


def get_include() -> str:
    """The include directory holding ``sstsr/*.hpp``.

    Works from a wheel (the headers sit beside this file) and from a source checkout
    (they sit in ``cpp/include``), because a developer building a C++ consumer against
    an editable install is the common case and should not need a different path.
    """
    candidates = [_HERE / "include", _HERE.parents[2] / "cpp" / "include"]
    for candidate in candidates:
        if (candidate / _PROBE).is_file():
            return str(candidate)
    searched = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(f"the sstsr C++ headers are not installed. Looked for {_PROBE} in:\n  {searched}")


def get_cmake_dir() -> str:
    """The directory holding ``sstsr_cppConfig.cmake``.

    Only a built wheel carries the CMake package: it is rendered at build time, because a
    relocatable targets file has to resolve paths relative to wherever the wheel landed.
    From an editable install this raises, and a consumer should either install a wheel or
    point CMake at ``cpp/`` directly — which is what ``cpp/examples/consumer`` does.
    """
    get_include()  # the config is useless without the headers beside it
    directory = _HERE / "cmake"
    if not (directory / "sstsr_cppConfig.cmake").is_file():
        raise FileNotFoundError(
            f"no CMake package at {directory}. Only a built wheel carries one; from a source "
            "checkout, configure cpp/ directly or install a wheel."
        )
    return str(directory)
