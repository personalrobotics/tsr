// SPDX-License-Identifier: BSD-2-Clause
// Authors: Siddhartha Srinivasa and contributors to TSR
//
// Minimal test harness: the C++ core has no dependencies and neither do its tests.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <exception>
#include <limits>
#include <stdexcept>
#include <functional>
#include <string>
#include <vector>

namespace harness {

struct Case {
  const char* name;
  std::function<void()> body;
};

inline std::vector<Case>& cases() {
  static std::vector<Case> c;
  return c;
}

struct Register {
  Register(const char* name, std::function<void()> body) { cases().push_back({name, std::move(body)}); }
};

struct Failure : std::exception {
  std::string what_;
  explicit Failure(std::string w) : what_(std::move(w)) {}
  const char* what() const noexcept override { return what_.c_str(); }
};

inline void fail(const char* expr, const char* file, int line) {
  throw Failure(std::string(file) + ":" + std::to_string(line) + ": CHECK(" + expr + ") failed");
}

inline int run_all() {
  int failed = 0;
  for (const auto& c : cases()) {
    try {
      c.body();
      std::printf("  ok    %s\n", c.name);
    } catch (const std::exception& e) {
      ++failed;
      std::printf("  FAIL  %s\n        %s\n", c.name, e.what());
    }
  }
  std::printf("%zu tests, %d failed\n", cases().size(), failed);
  return failed == 0 ? 0 : 1;
}

}  // namespace harness

#define TEST(name)                                                   \
  static void name();                                                \
  static harness::Register name##_registration(#name, name);         \
  static void name()

#define CHECK(expr) \
  do {              \
    if (!(expr)) harness::fail(#expr, __FILE__, __LINE__); \
  } while (0)

#define CHECK_NEAR(a, b, tol) CHECK(std::fabs((a) - (b)) <= (tol))

#define CHECK_THROWS(expr, Exception)                                                  \
  do {                                                                                 \
    bool caught_ = false;                                                              \
    try {                                                                              \
      (void)(expr);                                                                    \
    } catch (const Exception&) {                                                       \
      caught_ = true;                                                                  \
    }                                                                                  \
    if (!caught_) harness::fail("throws " #Exception ": " #expr, __FILE__, __LINE__); \
  } while (0)

#define HARNESS_MAIN() \
  int main() { return harness::run_all(); }
