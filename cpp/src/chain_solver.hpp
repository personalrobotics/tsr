// SPDX-License-Identifier: BSD-2-Clause
// Authors: Siddhartha Srinivasa and contributors to TSR
//
// The chain's cold inverse: projected Levenberg-Marquardt with an analytic Jacobian.
//
// INTERNAL. This lives under src/ rather than include/ because it is not part of the package's
// public surface -- a consumer calls TSRChain::solve. It is a header only so the test can
// include it and check the Jacobian against central differences, which is the one thing a
// behavioural test cannot localise (see test_tsr_chain.cpp).
//
// Why least squares. The Python minimises the scalar
//
//     dt . dt + (3 - tr(R_target^T R))
//
// and that is *exactly* ||r||^2 for the 12-vector r = [dt, (R - R_target)/sqrt(2)], because
// 3 - tr(A^T B) = ||A - B||_F^2 / 2 for A, B in SO(3). So the objective is a least-squares
// problem in disguise and Levenberg-Marquardt is the natural method, not an approximation of
// the Python's.
//
// Why analytic. rpy_to_rot is Rz(y) Ry(p) Rx(r), so each rotation derivative is a product of
// matrices already on hand, and one prefix/suffix sweep builds all 6n columns in O(n). A
// finite-difference Jacobian was measured at 31/40 and 18/40 recall on two fixtures where the
// analytic one scores 40/40, and it costs 6n extra compositions per iteration.
//
// Why projected. The bounds are a box, so a step is clamped back into it; and a coordinate
// pinned at a bound whose gradient pushes further out is held fixed for that step (the
// active-set half). Without that second part, a chain carrying a full SO(3) ball -- where RPY
// runs through gimbal lock and the Jacobian loses rank -- scores 18/40 instead of 38/40.
//
// This is NOT held to the Python's trajectory, and cannot be: see "What they do not agree on"
// in docs/CPP.md. It is held to properties, and to a recall floor.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include "sstsr/transform.hpp"
#include "sstsr/tsr.hpp"
#include "sstsr/tsr_chain.hpp"

namespace sstsr::detail {

// r is 12 long; J is 12 x 6n, row-major, left empty when jacobian is false.
struct ChainResidual {
  std::array<double, 12> r{};
  std::vector<double> J{};  // row-major 12 x 6n
};

//: 3 - tr(A^T B) = ||A - B||_F^2 / 2, so the rotation rows carry this factor.
inline const double kRootTwo = std::sqrt(2.0);

// The skew generators of the three axes: d/dtheta Rx(theta) = Ex Rx(theta), and likewise for
// y and z. Written as full 4x4s with a zero last row and column so they compose with
// Transform::operator*, which is a general 4x4 product.
inline Transform generator(int axis) {
  Transform g{};  // all zero, including (3,3): deliberately not homogeneous
  if (axis == 0) {
    g.at(1, 2) = -1.0;
    g.at(2, 1) = 1.0;
  } else if (axis == 1) {
    g.at(0, 2) = 1.0;
    g.at(2, 0) = -1.0;
  } else {
    g.at(0, 1) = -1.0;
    g.at(1, 0) = 1.0;
  }
  return g;
}

inline Transform rot_x(double a) {
  Transform t = Transform::identity();
  t.at(1, 1) = std::cos(a);
  t.at(1, 2) = -std::sin(a);
  t.at(2, 1) = std::sin(a);
  t.at(2, 2) = std::cos(a);
  return t;
}

inline Transform rot_y(double a) {
  Transform t = Transform::identity();
  t.at(0, 0) = std::cos(a);
  t.at(0, 2) = std::sin(a);
  t.at(2, 0) = -std::sin(a);
  t.at(2, 2) = std::cos(a);
  return t;
}

inline Transform rot_z(double a) {
  Transform t = Transform::identity();
  t.at(0, 0) = std::cos(a);
  t.at(0, 1) = -std::sin(a);
  t.at(1, 0) = std::sin(a);
  t.at(1, 1) = std::cos(a);
  return t;
}

// The six derivatives of Tw(c) with respect to its coordinates. The three translations are
// constant matrices; the three rotations use rpy_to_rot's Z-Y-X order, which is the fact the
// finite-difference test in test_tsr_chain.cpp pins.
inline std::array<Transform, 6> d_local_transform(const XyzRpy& c) {
  const Transform Rx = rot_x(c[3]), Ry = rot_y(c[4]), Rz = rot_z(c[5]);
  std::array<Transform, 6> d{};
  for (int k = 0; k < 3; ++k) {
    d[static_cast<std::size_t>(k)] = Transform{};
    d[static_cast<std::size_t>(k)].at(k, 3) = 1.0;
  }
  d[3] = Rz * (Ry * (generator(0) * Rx));
  d[4] = Rz * (generator(1) * (Ry * Rx));
  d[5] = generator(2) * (Rz * (Ry * Rx));
  // Each d[3..5] must carry only a rotation block; the products above already have a zero
  // last row and column because the generator does.
  return d;
}

// The residual, and optionally the Jacobian, at `coords` -- which must already be in the
// chart (the caller canonicalises, exactly as TSRChain::to_transform does).
inline ChainResidual chain_residual(const std::vector<TSR>& links, const ChainCoords& coords,
                                    const Transform& target, bool jacobian) {
  const std::size_t n = links.size();
  ChainResidual out;

  std::vector<Transform> Tw(n), step(n);
  for (std::size_t i = 0; i < n; ++i) {
    Tw[i] = xyzrpy_to_trans(coords[i]);
    step[i] = Tw[i] * links[i].Tw_e();
  }

  // Prefixes: A[i] = T0_w of the FIRST link, times every step before i. Only the first link's
  // T0_w participates, which is the chain's composition rule.
  std::vector<Transform> A(n);
  Transform acc = links[0].T0_w();
  for (std::size_t i = 0; i < n; ++i) {
    A[i] = acc;
    acc = acc * step[i];
  }
  const Transform T = acc;

  for (int k = 0; k < 3; ++k) out.r[static_cast<std::size_t>(k)] = T.at(k, 3) - target.at(k, 3);
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      out.r[static_cast<std::size_t>(3 + row * 3 + col)] = (T.at(row, col) - target.at(row, col)) / kRootTwo;
    }
  }
  if (!jacobian) return out;

  // Suffixes: B[i] = this link's Tw_e, times every step after i.
  std::vector<Transform> B(n);
  Transform suffix = Transform::identity();
  for (std::size_t i = n; i-- > 0;) {
    B[i] = links[i].Tw_e() * suffix;
    suffix = step[i] * suffix;
  }

  out.J.assign(12 * 6 * n, 0.0);
  const std::size_t cols = 6 * n;
  for (std::size_t i = 0; i < n; ++i) {
    const std::array<Transform, 6> d = d_local_transform(coords[i]);
    for (std::size_t j = 0; j < 6; ++j) {
      const Transform dT = A[i] * (d[j] * B[i]);
      const std::size_t c = i * 6 + j;
      for (int k = 0; k < 3; ++k) out.J[static_cast<std::size_t>(k) * cols + c] = dT.at(k, 3);
      for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
          out.J[static_cast<std::size_t>(3 + row * 3 + col) * cols + c] = dT.at(row, col) / kRootTwo;
        }
      }
    }
  }
  return out;
}

// Cholesky solve of a symmetric positive-definite system, in place. Returns false when the
// matrix is not positive definite, which the caller answers by raising lambda rather than by
// giving up -- a larger lambda makes the system diagonally dominant.
inline bool cholesky_solve(std::vector<double>& M, std::vector<double>& b, std::size_t m) {
  for (std::size_t i = 0; i < m; ++i) {
    for (std::size_t j = 0; j <= i; ++j) {
      double s = M[i * m + j];
      for (std::size_t k = 0; k < j; ++k) s -= M[i * m + k] * M[j * m + k];
      if (i == j) {
        if (!(s > 0.0)) return false;
        M[i * m + j] = std::sqrt(s);
      } else {
        M[i * m + j] = s / M[j * m + j];
      }
    }
  }
  for (std::size_t i = 0; i < m; ++i) {  // forward substitution
    double s = b[i];
    for (std::size_t k = 0; k < i; ++k) s -= M[i * m + k] * b[k];
    b[i] = s / M[i * m + i];
  }
  for (std::size_t i = m; i-- > 0;) {  // back substitution
    double s = b[i];
    for (std::size_t k = i + 1; k < m; ++k) s -= M[k * m + i] * b[k];
    b[i] = s / M[i * m + i];
  }
  return true;
}

}  // namespace sstsr::detail
