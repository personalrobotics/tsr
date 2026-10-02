// SPDX-License-Identifier: BSD-2-Clause
// Authors: Siddhartha Srinivasa and contributors to TSR
#include "sstsr/tsr_chain.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace sstsr {

namespace {

// A number formatted as the Python's "%.6g", for the append rejection message. The two
// implementations should complain about the same frame in recognisably the same words.
std::string g6(double v) {
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%.6g", v);
  return std::string(buf);
}

// The flat 6n coordinate vector back to one XyzRpy per link. Only the chart bookkeeping
// needs the flat form; every public entry point is per-link.
ChainCoords unflatten(const std::vector<double>& x) {
  ChainCoords out(x.size() / 6);
  for (std::size_t i = 0; i < out.size(); ++i) {
    for (std::size_t j = 0; j < 6; ++j) out[i][j] = x[i * 6 + j];
  }
  return out;
}

bool all_finite(const ChainCoords& c) {
  for (const XyzRpy& row : c) {
    for (double v : row) {
      if (!std::isfinite(v)) return false;
    }
  }
  return true;
}

}  // namespace

const char* to_string(ChainStatus s) { return s == ChainStatus::kSatisfied ? "satisfied" : "not_found"; }

TSRChain::TSRChain(const std::vector<TSR>& tsrs) {
  for (const TSR& t : tsrs) append(t);
}

void TSRChain::append(const TSR& tsr) {
  if (!tsrs_.empty()) {
    const Transform& T = tsr.T0_w();
    const Transform eye = Transform::identity();
    bool identity = true;
    for (std::size_t k = 0; k < 16; ++k) {
      if (std::fabs(T.m[k] - eye.m[k]) > kFrameAtol) {  // numpy.allclose with rtol=0: <= atol passes
        identity = false;
        break;
      }
    }
    if (!identity) {
      const double trace = T.at(0, 0) + T.at(1, 1) + T.at(2, 2);
      throw std::invalid_argument(
          "a TSR after the first in a chain is positioned by the chain: its frame is the previous link's end "
          "frame, so its T0_w is never read and must be the identity (got a translation of [" +
          g6(T.at(0, 3)) + ", " + g6(T.at(1, 3)) + ", " + g6(T.at(2, 3)) + "] m and rotation trace " + g6(trace) +
          "). Put the offset in the PREVIOUS link's Tw_e, which is what moves the end frame this link hangs off.");
    }
  }
  tsrs_.push_back(tsr);
}

std::vector<std::array<bool, 6>> TSRChain::is_valid(const ChainCoords& c) const {
  if (tsrs_.empty()) throw std::invalid_argument("cannot validate against an empty TSR chain");
  if (c.size() != tsrs_.size()) {
    throw std::invalid_argument("coordinates must be of equal length to the TSR chain (got " +
                                std::to_string(c.size()) + " for " + std::to_string(tsrs_.size()) + " links)");
  }
  std::vector<std::array<bool, 6>> out;
  out.reserve(tsrs_.size());
  for (std::size_t i = 0; i < tsrs_.size(); ++i) out.push_back(tsrs_[i].is_valid(c[i]));
  return out;
}

ChainCoords TSRChain::continuous_coordinates(const ChainCoords& c) const {
  if (tsrs_.empty()) throw std::invalid_argument("an empty TSR chain has no chart to map coordinates into");
  if (c.size() != tsrs_.size()) {
    throw std::invalid_argument("coordinates must be of equal length to the TSR chain (got " +
                                std::to_string(c.size()) + " for " + std::to_string(tsrs_.size()) + " links)");
  }
  ChainCoords out(tsrs_.size());
  for (std::size_t i = 0; i < tsrs_.size(); ++i) {
    const Bounds6& Bw = tsrs_[i].continuous_bounds();
    XyzRpy row = c[i];
    // Wrap the three rotations into their intervals FIRST, then clip all six. Translations
    // are not periodic and are only clipped. Clipping without wrapping would send a valid
    // wrapping coordinate to an unrelated boundary (#87, #90).
    for (int k = 0; k < 3; ++k) {
      row[static_cast<std::size_t>(3 + k)] = wrap_to_interval(row[static_cast<std::size_t>(3 + k)], Bw.lo(3 + k));
    }
    for (int j = 0; j < 6; ++j) {
      row[static_cast<std::size_t>(j)] = std::clamp(row[static_cast<std::size_t>(j)], Bw.lo(j), Bw.hi(j));
    }
    out[i] = row;
  }
  return out;
}

Transform TSRChain::to_transform(const ChainCoords& c) const {
  if (tsrs_.empty()) throw std::invalid_argument("cannot compute a transform for an empty TSR chain");
  const ChainCoords cont = continuous_coordinates(c);  // also checks the length

  // Deliberately NOT TSR::to_transform, which validates and throws on an out-of-bounds
  // coordinate: the chain composes the raw xyzrpy_to_trans with each link's Tw_e, exactly as
  // the Python does, because the solver evaluates points at and just outside bounds.
  //
  // Only the FIRST link's T0_w participates. append forces identity on the rest, and this
  // loop never reads another one.
  Transform current = tsrs_[0].T0_w();
  for (std::size_t i = 0; i < tsrs_.size(); ++i) {
    current = current * (xyzrpy_to_trans(cont[i]) * tsrs_[i].Tw_e());
  }
  return current;
}

ChainCoords TSRChain::sample_xyzrpy(Rng& rng) const {
  ChainCoords out(tsrs_.size());
  for (std::size_t i = 0; i < tsrs_.size(); ++i) out[i] = tsrs_[i].sample_xyzrpy(rng);
  return out;
}

Transform TSRChain::sample(Rng& rng) const { return to_transform(sample_xyzrpy(rng)); }

ChainSample TSRChain::sample_with_witness(Rng& rng) const {
  ChainCoords coords = sample_xyzrpy(rng);
  Transform pose = to_transform(coords);
  return ChainSample{pose, std::move(coords)};
}

std::optional<double> TSRChain::witness_residual(const Transform& T, const ChainCoords& c) const {
  // The size check comes before composing, so a wrong coordinate count is "not a witness"
  // rather than an exception -- to_transform would throw on the same input.
  if (tsrs_.empty() || c.size() != tsrs_.size()) return std::nullopt;
  for (std::size_t i = 0; i < tsrs_.size(); ++i) {
    const std::array<bool, 6> ok = tsrs_[i].is_valid(c[i]);
    for (bool b : ok) {
      if (!b) return std::nullopt;
    }
  }
  return geodesic_distance(to_transform(c), T);
}

bool TSRChain::validate_witness(const Transform& T, const ChainCoords& c, double tolerance) const {
  const std::optional<double> residual = witness_residual(T, c);
  return residual.has_value() && *residual < tolerance;
}

ChainSolveResult TSRChain::solve(const Transform& T, const SolveOptions& opt) const {
  // Validate before anything else, including before the empty-chain return: an out-of-range
  // budget is a caller mistake whatever the chain looks like. This is the Python's order,
  // which validates before it will even import SciPy.
  if (opt.max_starts < 0) {
    throw std::invalid_argument("max_starts must be >= 0, got " + std::to_string(opt.max_starts));
  }
  if (opt.max_nfev < 1) {
    throw std::invalid_argument("max_nfev must be >= 1, got " + std::to_string(opt.max_nfev));
  }
  if (!std::isfinite(opt.tolerance)) {
    throw std::invalid_argument("tolerance must be a finite positive number");
  }
  if (opt.tolerance <= 0.0) {
    throw std::invalid_argument("tolerance must be > 0, got " + std::to_string(opt.tolerance));
  }

  const std::size_t n = tsrs_.size();
  if (n == 0) return ChainSolveResult{};  // kNotFound, no coordinates, +inf, 0, 0

  // A guess is eligible only if it is the right length and finite. Anything else is ignored,
  // not an error -- a planner passing a neighbour's coordinates should not have to know
  // whether that neighbour was itself solved.
  const ChainCoords* guess = nullptr;
  if (opt.initial_guess.has_value() && opt.initial_guess->size() == n && all_finite(*opt.initial_guess)) {
    guess = &*opt.initial_guess;
  }

  // Warm path: a supplied witness that already validates. One forward composition, no
  // optimiser. This precedes the single-link branch, so for n == 1 a valid guess wins and the
  // closed form is never reached. The coordinates come back VERBATIM, un-canonicalised.
  if (guess != nullptr) {
    const std::optional<double> residual = witness_residual(T, *guess);
    if (residual.has_value() && *residual < opt.tolerance) {
      return ChainSolveResult{ChainStatus::kSatisfied, *guess, *residual, 0, 0};
    }
  }

  // Exact path: a single link reduces to the closed-form region check.
  if (n == 1) {
    if (tsrs_[0].contains(T)) {
      const ChainCoords coords{tsrs_[0].to_xyzrpy(T)};
      // No tolerance re-check, matching the Python: contains grants kEpsilon of slack per
      // coordinate while to_transform clips to the chart, so this can report kSatisfied with a
      // residual at or above tolerance. Copied rather than corrected -- the corpus pins it.
      return ChainSolveResult{ChainStatus::kSatisfied, coords, geodesic_distance(to_transform(coords), T), 0, 0};
    }
    const auto [dist, bw] = tsrs_[0].distance_bwopt(T);
    // `dist` is the region's 6-vector displacement norm, NOT a geodesic. The two coincide for
    // a pure translation or a single-axis rotation and diverge otherwise.
    return ChainSolveResult{ChainStatus::kNotFound, ChainCoords{bw}, dist, 0, 0};
  }

  // The chart over all 6n coordinates. continuous_bounds guarantees lower <= upper even for a
  // wrapping rotational interval. `free` is computed on the CONTINUOUS bounds, never on Bw: a
  // Bw row of [pi, pi] is fixed once mapped, [3pi/4, -3pi/4] is free with width pi/2.
  std::vector<double> lower, upper, x_full;
  lower.reserve(n * 6);
  upper.reserve(n * 6);
  x_full.reserve(n * 6);
  std::size_t free_count = 0;
  for (std::size_t i = 0; i < n; ++i) {
    const Bounds6& b = tsrs_[i].continuous_bounds();
    for (int j = 0; j < 6; ++j) {
      lower.push_back(b.lo(j));
      upper.push_back(b.hi(j));
      // The midpoint is NOT re-wrapped. For a chart row like [3pi/4, 3pi/4 + pi/2] it is
      // exactly pi, which is outside [-pi, pi); normalising it would push it out of the chart,
      // where to_transform clips it to a boundary instead of the middle.
      x_full.push_back((b.lo(j) + b.hi(j)) / 2.0);
      if (b.hi(j) > b.lo(j)) ++free_count;
    }
  }
  const ChainCoords midpoint = unflatten(x_full);

  // All-fixed chain: one candidate pose, no optimiser.
  if (free_count == 0) {
    const double res = geodesic_distance(to_transform(midpoint), T);
    const ChainStatus status = res < opt.tolerance ? ChainStatus::kSatisfied : ChainStatus::kNotFound;
    return ChainSolveResult{status, midpoint, res, 0, 0};
  }

  // max_starts == 0 runs no optimiser and reports the un-optimised midpoint, which is what the
  // Python's empty start schedule leaves in its best-point slot.
  if (opt.max_starts == 0) {
    const double res = geodesic_distance(to_transform(midpoint), T);
    const ChainStatus status = res < opt.tolerance ? ChainStatus::kSatisfied : ChainStatus::kNotFound;
    return ChainSolveResult{status, midpoint, res, 0, 0};
  }

  throw std::logic_error(
      "the cold chain inverse is stage 2b of issue #165 and is not implemented yet. Every exact path works: an "
      "empty chain, a single link, an all-fixed chain, max_starts == 0, and an initial_guess that already "
      "validates. Pass a retained witness as SolveOptions::initial_guess -- along a planner edge a neighbouring "
      "state's coordinates are exactly that -- or call the Python TSRChain.solve. See docs/CPP.md.");
}

std::pair<double, ChainCoords> TSRChain::distance(const Transform& T) const {
  const ChainSolveResult result = solve(T);
  return {result.residual, result.coordinates};
}

std::pair<double, Transform> TSRChain::closest_transform(const Transform& T) const {
  const auto [dist, coords] = distance(T);
  // On an empty chain this throws out of to_transform, as the Python does: there is no pose to
  // report, and an identity transform would be a confident wrong answer.
  return {dist, to_transform(coords)};
}

bool TSRChain::contains(const Transform& T, const std::optional<ChainCoords>& guess) const {
  if (tsrs_.empty()) return false;
  if (tsrs_.size() == 1) return tsrs_[0].contains(T);
  SolveOptions opt{};
  opt.initial_guess = guess;
  return solve(T, opt).status == ChainStatus::kSatisfied;
}

ChainCoords TSRChain::to_xyzrpy(const Transform& T) const { return distance(T).second; }

}  // namespace sstsr
