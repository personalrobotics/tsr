// SPDX-License-Identifier: BSD-2-Clause
// Authors: Siddhartha Srinivasa and contributors to TSR
//
// Hand-written cases for the region. The systematic agreement with the Python is the
// conformance corpus (test_conformance.cpp); these cover the construction rejections and
// the branches a corpus of sampled probes reaches only by luck.
#include <cmath>
#include <numbers>
#include <stdexcept>

#include "harness.hpp"
#include "sstsr/transform.hpp"
#include "sstsr/tsr.hpp"

using namespace sstsr;
constexpr double kPi = std::numbers::pi;

namespace {
Transform frame(double x, double y, double z, double roll = 0, double pitch = 0, double yaw = 0) {
  return xyzrpy_to_trans({x, y, z, roll, pitch, yaw});
}
Bounds6 bounds(std::array<std::array<double, 2>, 6> rows) { return Bounds6{rows}; }
const Bounds6 kBox = bounds({{{-0.05, 0.05}, {-0.05, 0.05}, {0, 0}, {0, 0}, {0, 0}, {-kPi, kPi}}});
}  // namespace

TEST(rpy_roundtrip_and_gimbal_branches) {
  for (const Rpy rpy : {Rpy{0.3, -0.4, 1.2}, Rpy{-2.9, 1.0, -3.0}, Rpy{0.1, 0.2, 0.3}}) {
    const Rpy back = rot_to_rpy(xyzrpy_to_trans({0, 0, 0, rpy[0], rpy[1], rpy[2]}));
    for (int i = 0; i < 3; ++i) CHECK_NEAR(back[static_cast<std::size_t>(i)], rpy[static_cast<std::size_t>(i)], 1e-12);
  }
  const Rpy up = rot_to_rpy(xyzrpy_to_trans({0, 0, 0, 0.4, kPi / 2, 0.0}));
  CHECK_NEAR(up[1], kPi / 2, 1e-12);
  CHECK_NEAR(up[2], 0.0, 1e-12);  // yaw fixed to 0 at the lock
  const Rpy down = rot_to_rpy(xyzrpy_to_trans({0, 0, 0, 0.4, -kPi / 2, 0.0}));
  CHECK_NEAR(down[1], -kPi / 2, 1e-12);
}

TEST(wrap_to_interval_matches_numpy_floored_modulo) {
  CHECK_NEAR(wrap_to_interval(-3.5, -kPi), -3.5 + 2 * kPi, 1e-12);
  CHECK_NEAR(wrap_to_interval(4.0, -kPi), 4.0 - 2 * kPi, 1e-12);
  CHECK_NEAR(wrap_to_interval(0.5, 0.0), 0.5, 1e-12);
  CHECK(wrap_to_interval(-1e-18, 0.0) >= 0.0 && wrap_to_interval(-1e-18, 0.0) < 2 * kPi);
}

TEST(construction_rejects_what_sstsr_rejects) {
  Transform bad = Transform::identity();
  bad.at(3, 3) = 2.0;
  CHECK_THROWS(TSR(bad, Transform::identity(), kBox), std::invalid_argument);
  Transform scaled = Transform::identity();
  scaled.at(0, 0) = 1.5;
  CHECK_THROWS(TSR(scaled, Transform::identity(), kBox), std::invalid_argument);
  Transform reflected = Transform::identity();
  reflected.at(2, 2) = -1.0;
  CHECK_THROWS(TSR(Transform::identity(), reflected, kBox), std::invalid_argument);
  Transform nan = Transform::identity();
  nan.at(0, 3) = std::nan("");
  CHECK_THROWS(TSR(nan, Transform::identity(), kBox), std::invalid_argument);
  CHECK_THROWS(TSR(Transform::identity(), Transform::identity(), bounds({{{1, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}}})), std::invalid_argument);
  CHECK_THROWS(TSR(Transform::identity(), Transform::identity(), bounds({{{0, 0}, {0, std::nan("")}, {0, 0}, {0, 0}, {0, 0}, {0, 0}}})), std::invalid_argument);
  // An outer rotational interval is valid: it wraps through +-pi.
  TSR outer(Transform::identity(), Transform::identity(), bounds({{{0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {3 * kPi / 4, -3 * kPi / 4}}}));
  CHECK_NEAR(outer.continuous_bounds().hi(5) - outer.continuous_bounds().lo(5), kPi / 2, 1e-12);
  CHECK(outer.contains(frame(0, 0, 0, 0, 0, kPi)));
  CHECK(!outer.contains(frame(0, 0, 0, 0, 0, 0.0)));
}

TEST(containment_distance_and_closest_transform) {
  TSR t(frame(1.0, 2.0, 0.0, 0, 0, 0.5), frame(0.1, 0, 0), kBox);
  Rng rng(3);
  const Transform inside = t.sample(rng);  // a sample is contained
  CHECK(t.contains(inside));
  CHECK_NEAR(t.distance(inside), 0.0, 1e-12);
  // Move 0.3 m out along the region's x: distance is the translation excess beyond the box
  const Transform out = frame(0.3, 0, 0) * inside;
  CHECK(!t.contains(out));
  CHECK(t.distance(out) > 0.2 && t.distance(out) < 0.35);
  const auto [d, closest] = t.closest_transform(out);
  CHECK_NEAR(d, t.distance(out), 1e-12);
  CHECK(t.contains(closest));
  CHECK_NEAR(t.distance(closest), 0.0, 1e-9);
}

TEST(sampling_uses_six_draws_in_order_and_stays_in_bounds) {
  TSR t(frame(0.5, 0, 0.2), Transform::identity(), bounds({{{-0.1, 0.1}, {-0.2, 0.2}, {0, 0.3}, {-0.4, 0.4}, {0, 0}, {2.5, -2.5}}}));
  Rng a(11), b(11);
  const XyzRpy s = t.sample_xyzrpy(a);
  std::array<double, 6> draws{};
  for (double& d : draws) d = unit(b);
  const Bounds6& c = t.continuous_bounds();
  for (int i = 0; i < 3; ++i) CHECK_NEAR(s[static_cast<std::size_t>(i)], c.lo(i) + (c.hi(i) - c.lo(i)) * draws[static_cast<std::size_t>(i)], 1e-15);
  CHECK(s[5] >= -kPi && s[5] < kPi);
  for (int k = 0; k < 200; ++k) CHECK(t.contains(t.sample(a)));
  CHECK_NEAR(t.volume(), 0.2 + 0.4 + 0.3 + 0.8 + 0.0 + 0.0, 1e-12);  // the outer yaw row counts as zero, as sstsr's _interval_sum does
}

HARNESS_MAIN()
