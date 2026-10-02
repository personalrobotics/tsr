// SPDX-License-Identifier: BSD-2-Clause
// Authors: Siddhartha Srinivasa and contributors to TSR
//
// What a planner does with a region: build it, sample it, ask whether a pose is in it.
// If this compiles and runs against the installed package, the package is usable.
#include <cstdio>
#include <stdexcept>
#include <vector>

#include "sstsr/rng.hpp"
#include "sstsr/tsr.hpp"
#include "sstsr/tsr_chain.hpp"

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

  // What a planner does with a CHAIN: sample it, keep the witness, and check membership from
  // that witness without an optimiser. This also makes the example link against tsr_chain.cpp,
  // which is the only way a source file missing from the wheel's INTERFACE_SOURCES shows up --
  // the wheel ships the file either way, so nothing else would notice.
  Bounds6 hinge{};
  hinge.rows[5] = {-1.5707963267948966, 1.5707963267948966};  // free yaw about the hinge
  Transform door_frame = Transform::identity();
  door_frame.at(0, 3) = 1.0;
  door_frame.at(2, 3) = 0.8;
  Transform along_door = Transform::identity();
  along_door.at(0, 3) = 0.6;

  Bounds6 handle{};
  handle.rows[2] = {-0.05, 0.05};
  const TSRChain chain({TSR(door_frame, along_door, hinge),
                        TSR(Transform::identity(), Transform::identity(), handle)});

  Rng chain_rng(11);
  const ChainSample s = chain.sample_with_witness(chain_rng);
  if (!chain.validate_witness(s.pose, s.coordinates)) {
    std::fprintf(stderr, "a sampled chain pose was not certified by its own witness\n");
    return 1;
  }
  const ChainSolveResult warm = chain.solve(s.pose, {.initial_guess = s.coordinates});
  if (warm.status != ChainStatus::kSatisfied || warm.nfev != 0 || warm.starts != 0) {
    std::fprintf(stderr, "a warm chain solve did not take the exact path (%s, nfev %d, starts %d)\n",
                 to_string(warm.status), warm.nfev, warm.starts);
    return 1;
  }

  // And the cold inverse: no guess at all, so the bounded search runs. This is what lets a
  // planner use a chain as a path constraint without a witness to hand.
  const ChainSolveResult cold = chain.solve(s.pose);
  if (cold.status != ChainStatus::kSatisfied) {
    std::fprintf(stderr, "a cold chain solve did not recover a pose sampled from the chain (%s, residual %.3e)\n",
                 to_string(cold.status), cold.residual);
    return 1;
  }
  if (cold.nfev > kDefaultMaxNfev || cold.starts > kDefaultMaxStarts) {
    std::fprintf(stderr, "a cold chain solve exceeded its budget (nfev %d, starts %d)\n", cold.nfev, cold.starts);
    return 1;
  }

  // The composition rule is part of the package's behaviour too: a later link cannot carry a
  // frame the chain would ignore.
  bool chain_rejected = false;
  try {
    Transform offset = Transform::identity();
    offset.at(1, 3) = 0.2;
    (void)TSRChain({TSR(door_frame, along_door, hinge), TSR(offset, Transform::identity(), handle)});
  } catch (const std::invalid_argument&) {
    chain_rejected = true;
  }
  if (!chain_rejected) {
    std::fprintf(stderr, "a later chain link carrying a non-identity T0_w was accepted\n");
    return 1;
  }

  std::printf("sstsr_cpp consumer ok: volume %.6f, distance to a far pose %.6f, warm residual %.3e, "
              "cold residual %.3e in %d evaluations\n",
              region.volume(), region.distance(far), warm.residual, cold.residual, cold.nfev);
  return 0;
}
