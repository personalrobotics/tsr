#!/usr/bin/env bash
# Build a wheel, install it into a scratch venv, and build a C++ consumer against the
# CMake package the wheel carries.
#
# This is the only check of the relocatable targets file and the rendered version file:
# an exported targets file from `cmake --install` holds absolute build-machine paths, so
# the wheel needs its own, and nothing but this exercises it.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "${here}/../../.." && pwd)"
work="$(mktemp -d)"
trap 'rm -rf "${work}"' EXIT

uv build --wheel -o "${work}/dist" "${repo}" >/dev/null
wheel="$(ls "${work}"/dist/*.whl)"

uv venv "${work}/venv" >/dev/null
VIRTUAL_ENV="${work}/venv" uv pip install --quiet "${wheel}"
python="${work}/venv/bin/python"

version="$("${python}" -c "import tsr; print(tsr.__version__)")"
cmake_version="$("${python}" -c "
import re, tsr
print(re.match(r'^(\d+(?:\.\d+)*)', tsr.__version__).group(1))
")"
echo "installed sstsr ${version} (CMake sees ${cmake_version})"
echo "cmake dir: $("${python}" -c 'import tsr; print(tsr.get_cmake_dir())')"

cmake -S "${here}" -B "${work}/build" -DCMAKE_BUILD_TYPE=Release \
      -DPython_EXECUTABLE="${python}" -DSSTSR_CPP_VERSION="${cmake_version}" >/dev/null
cmake --build "${work}/build" -j >/dev/null
"${work}/build/wheel_consumer"
echo "wheel-package consumer: ok"
