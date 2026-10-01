# SPDX-License-Identifier: BSD-2-Clause
# Authors: Siddhartha Srinivasa and contributors to TSR

"""Put the C++ core's CMake package into the wheel.

The wheel stays ``py3-none-any``. It carries the C++ **sources and headers**, not a
compiled library, so a planner that links ``sstsr::sstsr_cpp`` compiles them into its own
build. That keeps one artifact, one release pipeline and one install story, while a C++
consumer still gets a real CMake package — the model ``ssik_cpp`` uses.

The CMake package cannot simply be copied in. An exported targets file from
``cmake --install`` holds absolute paths from the machine that built it, and a wheel lands
wherever the virtualenv is, so the wheel's targets file has to resolve its paths relative
to itself. ``cpp/cmake/sstsr_cppWheelTargets.cmake`` is that hand-written relocatable
version; this hook renders it, the shared config template, and a version file, and maps
them plus ``cpp/include`` and ``cpp/src`` into ``tsr/cpp/`` in the wheel.

Nothing is copied into ``src/``. Everything goes through ``force_include``, which maps a
path into the wheel without touching the working tree. That matters beyond tidiness: a
staged copy under ``src/tsr/cpp/include`` would **shadow** ``cpp/include`` for
:func:`tsr.get_include` in an editable install, so a developer would silently get the
headers as they were at the last wheel build.

Both packagings are exercised: ``cpp/examples/consumer`` builds against an installed
source prefix, ``cpp/examples/wheel_consumer`` against a built wheel. The wheel one is the
only thing that reads the file this hook renders.
"""

from __future__ import annotations

import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

#: Source layout.
CPP_INCLUDE = "cpp/include"
CPP_SRC = "cpp/src"
CPP_CMAKE = "cpp/cmake"

#: Where it all lands inside the wheel, beside ``tsr/cpp/__init__.py``'s locators.
WHEEL_CPP_DIR = "tsr/cpp"

#: ``configure_package_config_file`` normally expands this. Doing it by hand means the
#: wheel build needs no CMake — the sdist and wheel build on a machine without it.
#: Everything the real ``@PACKAGE_INIT@`` provides beyond this is for relocating an
#: install prefix, which the relocatable targets file handles itself.
_PACKAGE_INIT = """
# --- rendered by hatch_build.py in place of @PACKAGE_INIT@ -------------------
# The wheel's package is relocatable: sstsr_cppTargets.cmake resolves every path
# relative to itself, so there is no install prefix to reconstruct here.
macro(check_required_components _NAME)
  foreach(comp ${${_NAME}_FIND_COMPONENTS})
    if(NOT ${_NAME}_${comp}_FOUND)
      if(${_NAME}_FIND_REQUIRED_${comp})
        set(${_NAME}_FOUND FALSE)
      endif()
    endif()
  endforeach()
endmacro()
# ----------------------------------------------------------------------------
"""

#: A SameMajorVersion check, rendered rather than produced by
#: ``write_basic_package_version_file`` for the same reason.
_CONFIG_VERSION = """\
# Rendered by hatch_build.py. SameMajorVersion, matching cpp/CMakeLists.txt.
set(PACKAGE_VERSION "{version}")
if(PACKAGE_VERSION VERSION_LESS PACKAGE_FIND_VERSION)
  set(PACKAGE_VERSION_COMPATIBLE FALSE)
elseif(PACKAGE_FIND_VERSION_MAJOR STREQUAL "{major}")
  set(PACKAGE_VERSION_COMPATIBLE TRUE)
  if(PACKAGE_FIND_VERSION STREQUAL PACKAGE_VERSION)
    set(PACKAGE_VERSION_EXACT TRUE)
  endif()
else()
  set(PACKAGE_VERSION_COMPATIBLE FALSE)
endif()
"""


def _cmake_version(release: str) -> str:
    """The leading ``X.Y.Z`` of a PEP 440 version.

    CMake compares versions numerically and has no notion of ``rc1`` or ``.dev3``, so a
    pre-release has to present as its release number. ``3.3.0rc1`` becomes ``3.3.0``.
    """
    match = re.match(r"^(\d+(?:\.\d+)*)", release)
    return match.group(1) if match else "0"


class CppPackageHook(BuildHookInterface):
    """Maps the C++ core and its CMake package into ``tsr/cpp/`` in the wheel."""

    PLUGIN_NAME = "sstsr-cpp"

    _rendered: Optional[str] = None

    def initialize(self, version: str, build_data: Dict[str, Any]) -> None:
        if self.target_name != "wheel":
            return  # the sdist ships cpp/ as it stands; nothing to render

        root = Path(self.root)
        for required in (CPP_INCLUDE, CPP_SRC, CPP_CMAKE):
            if not (root / required).is_dir():
                raise RuntimeError(f"hatch_build.py: {required} is missing; cannot build the C++ package")

        # The three generated files go to a scratch directory, never into src/.
        self._rendered = tempfile.mkdtemp(prefix="sstsr-cpp-cmake-")
        rendered = Path(self._rendered)

        template = (root / CPP_CMAKE / "sstsr_cppConfig.cmake.in").read_text()
        (rendered / "sstsr_cppConfig.cmake").write_text(template.replace("@PACKAGE_INIT@", _PACKAGE_INIT))
        shutil.copyfile(root / CPP_CMAKE / "sstsr_cppWheelTargets.cmake", rendered / "sstsr_cppTargets.cmake")
        numeric = _cmake_version(self.metadata.version)
        (rendered / "sstsr_cppConfigVersion.cmake").write_text(
            _CONFIG_VERSION.format(version=numeric, major=numeric.split(".")[0])
        )

        force_include = build_data.setdefault("force_include", {})
        force_include[str(root / CPP_INCLUDE)] = f"{WHEEL_CPP_DIR}/include"
        force_include[str(root / CPP_SRC)] = f"{WHEEL_CPP_DIR}/src"
        for name in ("sstsr_cppConfig.cmake", "sstsr_cppTargets.cmake", "sstsr_cppConfigVersion.cmake"):
            force_include[str(rendered / name)] = f"{WHEEL_CPP_DIR}/cmake/{name}"

    def finalize(self, version: str, build_data: Dict[str, Any], artifact_path: str) -> None:
        if self._rendered:
            shutil.rmtree(self._rendered, ignore_errors=True)
            self._rendered = None
