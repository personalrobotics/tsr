// SPDX-License-Identifier: BSD-2-Clause
// Authors: Siddhartha Srinivasa and contributors to TSR
//
// The C++ core answers what the Python answers.
//
// The Python in src/tsr/core/ is the source of truth for the rules; this holds the C++ to
// them on a checked-in corpus of the Python's recorded answers
// (tools/tsr_conformance.py -> tests/reference/tsr_conformance.json). Neither
// implementation reads the other, and a rule change shows up as a corpus diff.
//
// Tolerances, and why they differ: the continuous-bounds chart and the volume are
// arithmetic on the inputs, so they must agree to 1e-12; contains is a verdict, so it must
// agree exactly; distance and the closest transform run through asin/atan2 and a
// nine-candidate minimum, so 1e-9 is the honest floor across standard libraries. These are
// the same tolerances the corpus's own --check mode uses.
//
// What this does NOT check, deliberately: the sampled coordinates the corpus records under
// `sample_xyzrpy_seed7_python`. The two implementations draw from different engines by
// design (see sstsr/rng.hpp), so sampling is checked as a property instead -- in bounds and
// contained -- in test_tsr.cpp. See docs/CPP.md.
#include <cmath>
#include <cstdio>
#include <string>

#include "harness.hpp"
#include "json.hpp"
#include "sstsr/transform.hpp"
#include "sstsr/tsr.hpp"

using namespace sstsr;

namespace {

constexpr double kChartTol = 1e-12;  // arithmetic on the inputs
constexpr double kPoseTol = 1e-9;    // through asin/atan2 and a candidate minimum

Transform transform_from(const testjson::Value& rows) {
  std::array<std::array<double, 4>, 4> m{};
  for (std::size_t r = 0; r < 4; ++r) {
    for (std::size_t c = 0; c < 4; ++c) m[r][c] = rows[r][c].number();
  }
  return Transform::from_rows(m);
}

Bounds6 bounds_from(const testjson::Value& rows) {
  Bounds6 b{};
  for (std::size_t i = 0; i < 6; ++i) b.rows[i] = {rows[i][0].number(), rows[i][1].number()};
  return b;
}

const testjson::Value& corpus() {
  static const testjson::Value v = testjson::parse_file(SSTSR_CONFORMANCE_CORPUS);
  return v;
}

std::string where(const std::string& region, std::size_t probe) {
  return region + " probe " + std::to_string(probe);
}

}  // namespace

TEST(corpus_is_the_one_this_build_expects) {
  // A corpus that cannot be read, or that is empty, must fail rather than vacuously pass:
  // this test is the only thing holding the C++ to the Python.
  const testjson::Value& c = corpus();
  const auto& regions = c["regions"].array();
  CHECK(regions.size() == 9);
  std::size_t probes = 0;
  for (const auto& r : regions) probes += r["probes"].array().size();
  CHECK(probes == 162);
  std::printf("        corpus: sstsr %s, numpy %s, %zu regions, %zu probes\n",
              c["versions"]["sstsr"].string().c_str(), c["versions"]["numpy"].string().c_str(), regions.size(),
              probes);
}

TEST(construction_and_the_continuous_chart_agree) {
  for (const auto& region : corpus()["regions"].array()) {
    const std::string name = region["name"].string();
    const TSR t(transform_from(region["T0_w"]), transform_from(region["Tw_e"]), bounds_from(region["Bw"]));

    const Bounds6 expected = bounds_from(region["continuous_bounds"]);
    for (int i = 0; i < 6; ++i) {
      if (std::fabs(t.continuous_bounds().lo(i) - expected.lo(i)) > kChartTol ||
          std::fabs(t.continuous_bounds().hi(i) - expected.hi(i)) > kChartTol) {
        harness::fail(("continuous bounds row " + std::to_string(i) + " of " + name).c_str(), __FILE__, __LINE__);
      }
    }
    CHECK_NEAR(t.volume(), region["volume"].number(), kChartTol);
  }
}

TEST(every_probe_agrees) {
  for (const auto& region : corpus()["regions"].array()) {
    const std::string name = region["name"].string();
    const TSR t(transform_from(region["T0_w"]), transform_from(region["Tw_e"]), bounds_from(region["Bw"]));
    const auto& probes = region["probes"].array();

    for (std::size_t p = 0; p < probes.size(); ++p) {
      const auto& probe = probes[p];
      const Transform T = transform_from(probe["T"]);

      if (t.contains(T) != probe["contains"].boolean()) {
        harness::fail(("contains disagrees at " + where(name, p)).c_str(), __FILE__, __LINE__);
      }
      if (std::fabs(t.distance(T) - probe["distance"].number()) > kPoseTol) {
        harness::fail(("distance disagrees at " + where(name, p)).c_str(), __FILE__, __LINE__);
      }

      const auto [dist, closest] = t.closest_transform(T);
      if (std::fabs(dist - probe["distance"].number()) > kPoseTol) {
        harness::fail(("closest_transform's distance disagrees at " + where(name, p)).c_str(), __FILE__, __LINE__);
      }
      const Transform expected_closest = transform_from(probe["closest"]);
      for (std::size_t k = 0; k < 16; ++k) {
        if (std::fabs(closest.m[k] - expected_closest.m[k]) > kPoseTol) {
          harness::fail(("closest transform entry " + std::to_string(k) + " disagrees at " + where(name, p)).c_str(),
                        __FILE__, __LINE__);
        }
      }
      // The Python records whether its own closest pose is itself contained. A closest
      // pose that falls outside the region it was projected into is the failure this
      // catches, and it is a verdict, so it must agree exactly.
      if (t.contains(closest) != probe["closest_contained"].boolean()) {
        harness::fail(("closest_contained disagrees at " + where(name, p)).c_str(), __FILE__, __LINE__);
      }
    }
  }
}

HARNESS_MAIN()
