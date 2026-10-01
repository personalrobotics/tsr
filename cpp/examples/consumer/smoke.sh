#!/usr/bin/env bash
# Install cpp/ to a scratch prefix and build this consumer against the exported package.
# This is the only check that the installed CMake package is usable from outside the
# build tree, which is what a planner actually does.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cpp_root="$(cd "${here}/../.." && pwd)"
work="$(mktemp -d)"
trap 'rm -rf "${work}"' EXIT

cmake -S "${cpp_root}" -B "${work}/build" -DCMAKE_BUILD_TYPE=Release \
      -DSSTSR_CPP_BUILD_TESTS=OFF -DCMAKE_INSTALL_PREFIX="${work}/prefix" >/dev/null
cmake --build "${work}/build" -j >/dev/null
cmake --install "${work}/build" >/dev/null

cmake -S "${here}" -B "${work}/consumer" -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_PREFIX_PATH="${work}/prefix" >/dev/null
cmake --build "${work}/consumer" -j >/dev/null
"${work}/consumer/consumer"
echo "installed-package consumer: ok"
