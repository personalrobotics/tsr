// SPDX-License-Identifier: BSD-2-Clause
// Authors: Siddhartha Srinivasa and contributors to TSR
#pragma once

#include <cstddef>
#include <random>

namespace sstsr {

// The generator a seeded sample is drawn from. Owned by the caller and passed by
// reference, so a solve that threads one generator through is repeatable.
using Rng = std::mt19937_64;

// A draw in [0, 1) that does not go through the standard library's distributions,
// whose implementations differ between libstdc++ and libc++. Taking the top 53 bits
// of the engine's output directly keeps a seeded sample identical across both.
//
// This is the one place the C++ deliberately differs from the Python: numpy's
// Generator is a different engine, so a seeded sample is reproducible *within* an
// implementation rather than across the two. Everything else in the contract is
// shared -- see docs/CPP.md.
inline double unit(Rng& rng) { return static_cast<double>(rng() >> 11) * 0x1.0p-53; }

}  // namespace sstsr
