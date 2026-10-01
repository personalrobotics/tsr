// SPDX-License-Identifier: BSD-2-Clause
// Authors: Siddhartha Srinivasa and contributors to TSR
//
// What a planner does with a region: build it, sample it, ask whether a pose is in it.
// If this compiles and runs against the installed package, the package is usable.
#include <cstdio>
#include <stdexcept>

#include "sstsr/rng.hpp"
#include "sstsr/tsr.hpp"

int main() {
  using namespace sstsr;

  Bounds6 Bw{};
  Bw.rows[0] = {-0.05, 0.05};
  Bw.rows[1] = {-0.05, 0.05};
  Bw.rows[5] = {-3.14159265358979323846, 3.14159265358979323846};
  const TSR region(Transform::identity(), Transform::identity(), Bw);

  Rng rng(7);
  const Transform sampled = region.sample(rng);
  if (!region.contains(sampled)) {
    std::fprintf(stderr, "a sampled pose was not contained\n");
    return 1;
  }

  Transform far = Transform::identity();
  far.at(0, 3) = 5.0;
  if (region.contains(far)) {
    std::fprintf(stderr, "a pose 5 m away was contained\n");
    return 1;
  }
  if (region.distance(far) <= 0.0) {
    std::fprintf(stderr, "a pose outside the region has no distance to it\n");
    return 1;
  }

  // The construction contract is part of the package's behaviour, not just its headers.
  bool rejected = false;
  try {
    Transform reflected = Transform::identity();
    reflected.at(0, 0) = -1.0;
    (void)TSR(reflected, Transform::identity(), Bw);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  if (!rejected) {
    std::fprintf(stderr, "a reflected frame was accepted\n");
    return 1;
  }

  std::printf("sstsr_cpp consumer ok: volume %.6f, distance to a far pose %.6f\n", region.volume(),
              region.distance(far));
  return 0;
}
