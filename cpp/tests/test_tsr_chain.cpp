// SPDX-License-Identifier: BSD-2-Clause
// Authors: Siddhartha Srinivasa and contributors to TSR
//
// Hand-written cases for the chain. The systematic agreement with the Python is the
// conformance corpus (test_conformance.cpp); these cover the rejections, the asymmetries, and
// the properties a corpus of recorded answers cannot express -- sampling (which draws from a
// different engine by design, see sstsr/rng.hpp) and the construction rule.
//
// Stage 2a: the cold inverse is not implemented, so the cases here that would need it assert
// the throw instead. Stage 2b replaces those with the real properties.
#include <cmath>
#include <numbers>
#include <stdexcept>
#include <vector>

#include "harness.hpp"
#include "sstsr/transform.hpp"
#include "sstsr/tsr.hpp"
#include "sstsr/tsr_chain.hpp"

using namespace sstsr;
constexpr double kPi = std::numbers::pi;

namespace {

Transform frame(double x, double y, double z, double roll = 0, double pitch = 0, double yaw = 0) {
  return xyzrpy_to_trans({x, y, z, roll, pitch, yaw});
}
Bounds6 bounds(std::array<std::array<double, 2>, 6> rows) { return Bounds6{rows}; }

// The motivating chain from issue #165: a hinge with free yaw, then a handle a fixed distance
// along the door.
TSRChain door() {
  return TSRChain({
      TSR(frame(1.0, 0.0, 0.8), frame(0.6, 0.0, 0.0),
          bounds({{{0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {-kPi / 2, kPi / 2}}})),
      TSR(Transform::identity(), Transform::identity(),
          bounds({{{0, 0}, {0, 0}, {-0.05, 0.05}, {0, 0}, {-0.2, 0.2}, {0, 0}}})),
  });
}

// An outer rotational interval on the first link: the chart is [3pi/4, 3pi/4 + pi/2].
TSRChain wrapping() {
  return TSRChain({
      TSR(Transform::identity(), frame(0.3, 0, 0),
          bounds({{{0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {3 * kPi / 4, -3 * kPi / 4}}})),
      TSR(Transform::identity(), Transform::identity(),
          bounds({{{0, 0}, {0, 0}, {-0.1, 0.1}, {2.5, -2.5}, {0, 0}, {0, 0}}})),
  });
}

// An independent forward composition, written out rather than reusing to_transform, so the
// chart test has an oracle that is not the thing under test. Mirrors the Python's own
// independent-composition check in tests/tsr/test_tsr_chain.py.
Transform compose_independently(const TSRChain& chain, const ChainCoords& canonical) {
  Transform acc = chain.tsrs()[0].T0_w();
  for (std::size_t i = 0; i < chain.size(); ++i) {
    const Transform Tw = xyzrpy_to_trans(canonical[i]);
    Transform step = Tw * chain.tsrs()[i].Tw_e();
    acc = acc * step;
  }
  return acc;
}

double max_abs_diff(const Transform& a, const Transform& b) {
  double worst = 0.0;
  for (std::size_t k = 0; k < 16; ++k) worst = std::max(worst, std::fabs(a.m[k] - b.m[k]));
  return worst;
}

}  // namespace

TEST(a_later_link_carrying_a_frame_the_chain_would_ignore_is_refused) {
  // Issue #166: the chain positions every link after the first by the previous link's end
  // frame, so a later T0_w is never read. Dropping it silently would make the pose wrong by
  // exactly that offset, which is worse to debug than a rejected argument.
  const Bounds6 b = bounds({{{0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {-0.5, 0.5}}});
  const TSR first(frame(1.0, 2.0, 0.5), Transform::identity(), b);
  const TSR offset(frame(0.1, 0, 0), Transform::identity(), b);
  const TSR identity_framed(Transform::identity(), Transform::identity(), b);

  TSRChain chain;
  chain.append(first);  // the FIRST link may sit anywhere
  CHECK_THROWS(chain.append(offset), std::invalid_argument);
  chain.append(identity_framed);  // and an identity-framed one is fine
  CHECK(chain.size() == 2);

  // The vector constructor goes through append, so it refuses the same thing.
  CHECK_THROWS(TSRChain(std::vector<TSR>{first, offset}), std::invalid_argument);

  // A rotation-only later frame is refused too, not just a translation.
  const TSR rotated(frame(0, 0, 0, 0, 0, 0.3), Transform::identity(), b);
  TSRChain other;
  other.append(first);
  CHECK_THROWS(other.append(rotated), std::invalid_argument);
}

TEST(the_remedy_the_rejection_prescribes_actually_works) {
  // The message tells the caller to put the offset in the PREVIOUS link's Tw_e. That has to be
  // true, or the error is a dead end.
  const Bounds6 hinge = bounds({{{0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {-0.4, 0.4}}});
  const Bounds6 fixed = bounds({{{0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}}});
  const TSRChain chain({
      TSR(frame(1.0, 0, 0), frame(0.25, 0, 0), hinge),  // the offset lives here
      TSR(Transform::identity(), Transform::identity(), fixed),
  });
  const ChainCoords at_zero{XyzRpy{0, 0, 0, 0, 0, 0}, XyzRpy{0, 0, 0, 0, 0, 0}};
  const Transform pose = chain.to_transform(at_zero);
  CHECK_NEAR(pose.at(0, 3), 1.25, 1e-12);  // 1.0 from T0_w, 0.25 from the previous Tw_e
}

TEST(an_empty_chain_is_asymmetric_in_exactly_the_pythons_way) {
  const TSRChain empty;
  CHECK(empty.size() == 0);
  CHECK(!empty.contains(Transform::identity()));
  CHECK(!empty.validate_witness(Transform::identity(), {}));
  CHECK(empty.to_xyzrpy(Transform::identity()).empty());
  CHECK(std::isinf(empty.distance(Transform::identity()).first));
  // These throw: there is nothing to compose. Returning identity would be a confident wrong
  // answer, which is the whole objection of #166.
  CHECK_THROWS(empty.to_transform({}), std::invalid_argument);
  CHECK_THROWS(empty.is_valid({}), std::invalid_argument);
  CHECK_THROWS(empty.continuous_coordinates({}), std::invalid_argument);
  CHECK_THROWS(empty.closest_transform(Transform::identity()), std::invalid_argument);
}

TEST(a_wrong_coordinate_count_throws_from_the_chart_but_is_merely_not_a_witness) {
  const TSRChain chain = door();
  const ChainCoords too_few{XyzRpy{0, 0, 0, 0, 0, 0}};
  CHECK_THROWS(chain.to_transform(too_few), std::invalid_argument);
  CHECK_THROWS(chain.is_valid(too_few), std::invalid_argument);
  CHECK_THROWS(chain.continuous_coordinates(too_few), std::invalid_argument);
  // validate_witness answers the question "are these a witness?", and the answer for the wrong
  // number of coordinates is no, not an exception.
  CHECK(!chain.validate_witness(Transform::identity(), too_few));
  // And solve ignores an ineligible guess rather than failing on it.
  Rng rng(1);
  const ChainSample s = chain.sample_with_witness(rng);
  SolveOptions opt{};
  opt.initial_guess = too_few;
  opt.max_starts = 0;  // stay on an exact path; the cold inverse is stage 2b
  const ChainSolveResult result = chain.solve(s.pose, opt);
  CHECK(result.starts == 0 && result.nfev == 0);
}

TEST(the_chart_wraps_rotations_before_it_clips_them) {
  // Issue #87/#90. A yaw of -3.0 belongs to the interval [3pi/4, -3pi/4] -- it is 2pi away from
  // the in-chart equivalent. Clipping without wrapping first would send it to a boundary of an
  // unrelated interval, and the composed pose would be wrong rather than merely rounded.
  const TSRChain chain = wrapping();
  const Bounds6& chart = chain.tsrs()[0].continuous_bounds();
  CHECK_NEAR(chart.lo(5), 3 * kPi / 4, 1e-12);
  CHECK_NEAR(chart.hi(5), 3 * kPi / 4 + kPi / 2, 1e-12);

  ChainCoords c{XyzRpy{0, 0, 0, 0, 0, -3.0}, XyzRpy{0, 0, 0, 0, 0, 0}};
  const ChainCoords canonical = chain.continuous_coordinates(c);
  CHECK_NEAR(canonical[0][5], -3.0 + 2 * kPi, 1e-12);  // wrapped into the chart, not clipped
  CHECK(canonical[0][5] >= chart.lo(5) && canonical[0][5] <= chart.hi(5));

  // A full turn must not move the composed pose.
  ChainCoords turned = c;
  turned[0][5] += 2 * kPi;
  CHECK(max_abs_diff(chain.to_transform(c), chain.to_transform(turned)) < 1e-12);

  // And the composition agrees with an independent product of the canonical coordinates.
  CHECK(max_abs_diff(chain.to_transform(c), compose_independently(chain, canonical)) < 1e-12);
}

TEST(the_midpoint_of_a_wrapping_chart_row_is_left_outside_minus_pi_to_pi) {
  // The chart for [3pi/4, -3pi/4] is [3pi/4, 3pi/4 + pi/2], whose midpoint is exactly pi. A
  // port that "normalises" it to -pi would push it out of the chart, where the forward map
  // clips it to a boundary instead of the middle -- changing what max_starts == 0 reports.
  const TSRChain chain = wrapping();
  const Bounds6& chart = chain.tsrs()[0].continuous_bounds();
  const double midpoint = (chart.lo(5) + chart.hi(5)) / 2.0;
  CHECK_NEAR(midpoint, kPi, 1e-12);
  CHECK(midpoint >= kPi);  // i.e. NOT inside [-pi, pi)

  SolveOptions opt{};
  opt.max_starts = 0;
  const ChainSolveResult result = chain.solve(Transform::identity(), opt);
  CHECK(result.starts == 0 && result.nfev == 0);
  CHECK_NEAR(result.coordinates[0][5], kPi, 1e-12);
}

TEST(sampling_stays_in_bounds_and_every_sample_carries_a_valid_witness) {
  // Sampling draws from a different engine than numpy by design (sstsr/rng.hpp), so it is
  // checked as a property rather than against recorded coordinates. The property that matters
  // is the one a planner relies on: a sampled pose is a member, provably, via its witness.
  Rng rng(20261001);
  for (const TSRChain& chain : {door(), wrapping()}) {
    for (int k = 0; k < 200; ++k) {
      const ChainSample s = chain.sample_with_witness(rng);
      CHECK(s.coordinates.size() == chain.size());
      // Checked through is_valid rather than against the raw chart interval: a sampled rotation
      // comes back wrapped into [-pi, pi), which for a wrapping interval sits outside the
      // chart's own [lo, hi] while still being a member of it.
      for (std::size_t i = 0; i < chain.size(); ++i) {
        for (bool ok : chain.tsrs()[i].is_valid(s.coordinates[i])) CHECK(ok);
      }
      // The witness is a constructive certificate: no optimiser needed.
      CHECK(chain.validate_witness(s.pose, s.coordinates));
    }
  }
}

TEST(a_validating_witness_takes_the_exact_warm_path_and_comes_back_verbatim) {
  Rng rng(7);
  const TSRChain chain = door();
  const ChainSample s = chain.sample_with_witness(rng);

  SolveOptions opt{};
  opt.initial_guess = s.coordinates;
  const ChainSolveResult result = chain.solve(s.pose, opt);

  CHECK(result.status == ChainStatus::kSatisfied);
  CHECK(result.nfev == 0 && result.starts == 0);  // no optimiser ran
  // Verbatim, not canonicalised: the Python returns the caller's coordinates unchanged.
  CHECK(result.coordinates.size() == s.coordinates.size());
  for (std::size_t i = 0; i < s.coordinates.size(); ++i) {
    CHECK(result.coordinates[i] == s.coordinates[i]);
  }
  // contains takes the same path when handed the witness.
  CHECK(chain.contains(s.pose, s.coordinates));
}

TEST(an_out_of_chart_guess_within_the_containment_slack_is_still_accepted_verbatim) {
  // is_valid grants kEpsilon of slack on each side of every bound, while the chart clips. So a
  // guess just past a zero-width bound validates and is returned as given -- outside the
  // chart. The property "coordinates lie in the chart" therefore cannot apply to this path,
  // and this case is why.
  const TSRChain chain({
      TSR(frame(1.0, 0, 0.8), frame(0.6, 0, 0), bounds({{{0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}}})),
      TSR(Transform::identity(), Transform::identity(),
          bounds({{{0, 0}, {0, 0}, {-0.05, 0.05}, {0, 0}, {0, 0}, {0, 0}}})),
  });
  Rng rng(3);
  const ChainSample s = chain.sample_with_witness(rng);
  ChainCoords guess = s.coordinates;
  guess[0][5] += 9e-4;  // inside kEpsilon, outside the [0, 0] chart row

  SolveOptions opt{};
  opt.initial_guess = guess;
  const ChainSolveResult result = chain.solve(s.pose, opt);
  CHECK(result.status == ChainStatus::kSatisfied);
  CHECK(result.nfev == 0 && result.starts == 0);
  CHECK(result.coordinates[0][5] == guess[0][5]);
  CHECK(result.coordinates[0][5] > chain.tsrs()[0].continuous_bounds().hi(5));
}

TEST(an_all_fixed_chain_answers_from_one_pose_with_no_optimiser) {
  const Bounds6 fixed = bounds({{{0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}}});
  const TSRChain chain({
      TSR(frame(0.2, 0.2, 0.2, 0.1, 0.2, 0.3), frame(0.1, 0, 0), fixed),
      TSR(Transform::identity(), frame(0, 0.05, 0), fixed),
  });
  const ChainCoords zeros{XyzRpy{0, 0, 0, 0, 0, 0}, XyzRpy{0, 0, 0, 0, 0, 0}};
  const Transform only = chain.to_transform(zeros);

  const ChainSolveResult hit = chain.solve(only);
  CHECK(hit.status == ChainStatus::kSatisfied);
  CHECK(hit.nfev == 0 && hit.starts == 0);

  const ChainSolveResult miss = chain.solve(frame(4, 4, 4));
  CHECK(miss.status == ChainStatus::kNotFound);
  CHECK(miss.nfev == 0 && miss.starts == 0);
  CHECK(miss.residual > 1.0);
}

TEST(a_single_link_chain_uses_the_closed_form_and_never_the_optimiser) {
  const TSRChain chain({TSR(frame(0.4, -0.2, 0.3, 0.1, 0.2, 0.3), frame(0.05, 0, 0),
                            bounds({{{-0.05, 0.05}, {-0.05, 0.05}, {0, 0}, {0, 0}, {0, 0}, {-kPi, kPi}}}))});
  Rng rng(5);
  const Transform inside = chain.sample(rng);
  const ChainSolveResult hit = chain.solve(inside);
  CHECK(hit.status == ChainStatus::kSatisfied);
  CHECK(hit.nfev == 0 && hit.starts == 0);
  CHECK(chain.contains(inside));

  const ChainSolveResult miss = chain.solve(frame(3.0, -2.0, 1.0, 0.4, 0.2, -0.8));
  CHECK(miss.status == ChainStatus::kNotFound);
  CHECK(miss.nfev == 0 && miss.starts == 0);
  CHECK(!chain.contains(frame(3.0, -2.0, 1.0, 0.4, 0.2, -0.8)));
}

TEST(a_single_link_may_report_satisfied_with_a_residual_at_or_above_the_tolerance) {
  // Not a defect to fix in the port: contains grants kEpsilon of slack per coordinate while the
  // forward map clips to the chart, so the closed-form path can report kSatisfied with a
  // residual that is not below tolerance. The Python does exactly this (measured up to 2.4e-3
  // against a 1e-3 tolerance), the corpus pins it, and "correcting" it here would make the two
  // implementations disagree. It is why the "satisfied implies residual < tolerance" property
  // is scoped to chains of two or more links.
  const TSRChain chain({TSR(frame(0, 0, 0), Transform::identity(), bounds({{{0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}}}))});
  const Transform just_outside = frame(1e-3, 1e-3, 1e-3);
  const ChainSolveResult result = chain.solve(just_outside);
  CHECK(result.status == ChainStatus::kSatisfied);
  CHECK(result.residual >= kEpsilon);
}

TEST(a_chain_whose_only_freedom_is_an_outer_interval_is_not_mistaken_for_all_fixed) {
  // `free` must be read off the CONTINUOUS bounds, never off Bw. An outer rotational interval
  // has Bw.hi < Bw.lo, so a port that asks "Bw.hi > Bw.lo" concludes the row is fixed -- and a
  // chain whose only freedom is that row would then be answered from a single pose instead of
  // being searched. The two disagree nowhere else, which is why this needs its own chain.
  const Bounds6 outer_yaw_only =
      bounds({{{0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {3 * kPi / 4, -3 * kPi / 4}}});
  const Bounds6 all_fixed = bounds({{{0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}}});
  const TSRChain chain({
      TSR(Transform::identity(), frame(0.3, 0, 0), outer_yaw_only),
      TSR(Transform::identity(), Transform::identity(), all_fixed),
  });

  // The chart says the yaw row spans a quarter turn, so it is free.
  const Bounds6& chart = chain.tsrs()[0].continuous_bounds();
  CHECK(chart.hi(5) > chart.lo(5));
  CHECK_NEAR(chart.hi(5) - chart.lo(5), kPi / 2, 1e-12);

  // Therefore solve must reach the bounded search rather than the all-fixed shortcut. In stage
  // 2a that is the throw; in 2b it becomes a real solve, and either way it is NOT a one-pose
  // answer. A chain misread as all-fixed would return a result here instead.
  CHECK_THROWS(chain.solve(Transform::identity()), std::logic_error);
}

TEST(a_residual_exactly_at_the_tolerance_is_not_satisfied) {
  // The comparison is strictly `residual < tolerance`, matching the Python. Sampling will not
  // land on the boundary, so construct it: take the residual an all-fixed chain reports, then
  // ask again with the tolerance set to exactly that number.
  const Bounds6 fixed = bounds({{{0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}}});
  const TSRChain chain({
      TSR(frame(0.2, 0.2, 0.2, 0.1, 0.2, 0.3), frame(0.1, 0, 0), fixed),
      TSR(Transform::identity(), frame(0, 0.05, 0), fixed),
  });
  const Transform target = frame(1.0, 0.5, -0.3, 0.2, 0.1, 0.4);

  const double residual = chain.solve(target).residual;
  CHECK(residual > 0.0);

  SolveOptions at_boundary{};
  at_boundary.tolerance = residual;  // exactly equal, so `<` fails and `<=` would pass
  CHECK(chain.solve(target, at_boundary).status == ChainStatus::kNotFound);

  SolveOptions just_above{};
  just_above.tolerance = std::nextafter(residual, 1.0e9);
  CHECK(chain.solve(target, just_above).status == ChainStatus::kSatisfied);

  // validate_witness compares the same way.
  const ChainCoords zeros{XyzRpy{0, 0, 0, 0, 0, 0}, XyzRpy{0, 0, 0, 0, 0, 0}};
  const double witness = geodesic_distance(chain.to_transform(zeros), target);
  CHECK(!chain.validate_witness(target, zeros, witness));
  CHECK(chain.validate_witness(target, zeros, std::nextafter(witness, 1.0e9)));
}

TEST(solve_validates_its_budget_before_it_looks_at_the_chain) {
  const TSRChain chain = door();
  const TSRChain empty;
  for (const TSRChain* c : {&chain, &empty}) {
    SolveOptions bad_starts{};
    bad_starts.max_starts = -1;
    CHECK_THROWS(c->solve(Transform::identity(), bad_starts), std::invalid_argument);

    SolveOptions bad_nfev{};
    bad_nfev.max_nfev = 0;
    CHECK_THROWS(c->solve(Transform::identity(), bad_nfev), std::invalid_argument);

    SolveOptions bad_tol{};
    bad_tol.tolerance = 0.0;
    CHECK_THROWS(c->solve(Transform::identity(), bad_tol), std::invalid_argument);

    SolveOptions negative_tol{};
    negative_tol.tolerance = -1.0;
    CHECK_THROWS(c->solve(Transform::identity(), negative_tol), std::invalid_argument);

    SolveOptions nan_tol{};
    nan_tol.tolerance = std::nan("");
    CHECK_THROWS(c->solve(Transform::identity(), nan_tol), std::invalid_argument);
  }
}

TEST(a_non_finite_guess_is_ignored_rather_than_rejected) {
  // A planner passing a neighbouring state's coordinates should not have to know whether that
  // neighbour was itself solved, so an unusable guess is dropped, not an error.
  const TSRChain chain = door();
  Rng rng(11);
  const ChainSample s = chain.sample_with_witness(rng);
  ChainCoords nan_guess = s.coordinates;
  nan_guess[0][5] = std::nan("");

  SolveOptions opt{};
  opt.initial_guess = nan_guess;
  opt.max_starts = 0;  // stay on an exact path, since the cold inverse is stage 2b
  const ChainSolveResult result = chain.solve(s.pose, opt);
  CHECK(result.starts == 0 && result.nfev == 0);
  // The guess was dropped, so the answer is the midpoint rather than the (unusable) guess.
  for (std::size_t j = 0; j < 6; ++j) CHECK(std::isfinite(result.coordinates[0][j]));
}

TEST(the_cold_inverse_says_it_is_not_here_yet_rather_than_guessing) {
  // Stage 2a. A chain of two or more links with a free coordinate and no validating guess needs
  // the bounded numerical search, which lands in stage 2b. Until then it must fail loudly: a
  // plausible-looking not_found would be indistinguishable from a real one.
  const TSRChain chain = door();
  CHECK_THROWS(chain.solve(Transform::identity()), std::logic_error);
  CHECK_THROWS(chain.distance(Transform::identity()), std::logic_error);
  CHECK_THROWS(chain.closest_transform(Transform::identity()), std::logic_error);
  CHECK_THROWS(chain.to_xyzrpy(Transform::identity()), std::logic_error);
  CHECK_THROWS(chain.contains(Transform::identity()), std::logic_error);

  // Everything exact still works on the same chain.
  Rng rng(13);
  const ChainSample s = chain.sample_with_witness(rng);
  CHECK(chain.validate_witness(s.pose, s.coordinates));
  CHECK(chain.contains(s.pose, s.coordinates));
  SolveOptions zero{};
  zero.max_starts = 0;
  CHECK(chain.solve(s.pose, zero).nfev == 0);
}

TEST(geodesic_distance_is_zero_on_equal_poses_and_symmetric) {
  const Transform a = frame(0.3, -0.2, 0.5, 0.4, -0.3, 1.1);
  const Transform b = frame(-0.1, 0.4, 0.2, -0.2, 0.5, -0.7);
  // The acos clamp is what keeps this exactly 0 rather than NaN: the trace of R^T R lands a
  // rounding step above 3 for a pose compared with itself.
  CHECK(geodesic_distance(a, a) == 0.0);
  CHECK_NEAR(geodesic_distance(a, b), geodesic_distance(b, a), 1e-15);
  const Transform shifted = frame(0.3 + 0.25, -0.2, 0.5, 0.4, -0.3, 1.1);
  CHECK_NEAR(geodesic_distance(a, shifted), 0.25, 1e-12);
}

HARNESS_MAIN()
