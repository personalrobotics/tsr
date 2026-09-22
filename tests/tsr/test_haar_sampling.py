#!/usr/bin/env python
# SPDX-License-Identifier: BSD-2-Clause
# Authors: Siddhartha Srinivasa and contributors to TSR

"""Haar-uniform rotation sampling over a TSR's Bw box (issue #72).

Distributional tests use fixed seeds, so they are deterministic; the KS p-value
thresholds are conservative (1e-3) and the contrast test shows that coordinate-
uniform ``TSR.sample`` fails the same test by many orders of magnitude.
"""

import unittest

import numpy as np
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from numpy import pi
from scipy import stats

from tsr import TSR, sample_haar, sample_haar_xyzrpy

N = 20000
KS_P = 1e-3


def _box(rpy_lo, rpy_hi, xyz=((0.0, 0.0),) * 3):
    Bw = np.zeros((6, 2))
    Bw[:3] = xyz
    Bw[3:, 0], Bw[3:, 1] = rpy_lo, rpy_hi
    return Bw


FULL = _box((0.0, -pi / 2, 0.0), (2 * pi, pi / 2, 2 * pi))


def _rotations(tsr, seed, sampler=sample_haar):
    rng = np.random.default_rng(seed)
    if sampler is sample_haar:
        return np.array([sample_haar(tsr, rng)[:3, :3] for _ in range(N)])
    return np.array([tsr.sample(rng=rng)[:3, :3] for _ in range(N)])


def _random_rotation(seed):
    q = np.random.default_rng(seed).normal(size=4)
    w, x, y, z = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


class TestHaarSamplerContract(unittest.TestCase):
    def test_reproducible_under_explicit_rng(self):
        tsr = TSR(Bw=FULL)
        a = [sample_haar_xyzrpy(tsr, np.random.default_rng(7)) for _ in range(3)]
        b = [sample_haar_xyzrpy(tsr, np.random.default_rng(7)) for _ in range(3)]
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(
            sample_haar(tsr, np.random.default_rng(3)), sample_haar(tsr, np.random.default_rng(3))
        )

    def test_pitch_bounds_outside_half_pi_raise(self):
        above = float(np.nextafter(pi / 2, np.inf))
        below = float(np.nextafter(-pi / 2, -np.inf))
        for lo, hi in ((below, 0.0), (0.0, above), (-2.0, 2.0), (3 * pi / 4, -3 * pi / 4)):
            with self.assertRaises(ValueError):
                sample_haar_xyzrpy(TSR(Bw=_box((0.0, lo, 0.0), (0.0, hi, 0.0))))

    def test_point_pitch_at_gimbal_lock_raises(self):
        # At pitch = ±pi/2 roll and yaw are coupled: independent uniform roll/yaw in
        # [0, 1] would give a triangular (not uniform) effective angle (#116).
        for p in (pi / 2, -pi / 2):
            with self.assertRaises(ValueError):
                sample_haar_xyzrpy(TSR(Bw=_box((0.0, p, 0.0), (1.0, p, 1.0))))

    def test_point_pitch_just_inside_gimbal_lock_is_valid(self):
        for p in (float(np.nextafter(pi / 2, 0.0)), float(np.nextafter(-pi / 2, 0.0))):
            tsr = TSR(Bw=_box((0.0, p, 0.0), (1.0, p, 1.0)))
            xyzrpy = sample_haar_xyzrpy(tsr, np.random.default_rng(0))
            self.assertEqual(xyzrpy[4], p)
            self.assertTrue(all(tsr.is_valid(xyzrpy)))

    def test_nonzero_pitch_interval_touching_gimbal_lock_is_valid(self):
        for lo, hi in ((0.5, pi / 2), (-pi / 2, -0.5)):
            tsr = TSR(Bw=_box((0.0, lo, 0.0), (1.0, hi, 1.0)))
            self.assertTrue(all(tsr.is_valid(sample_haar_xyzrpy(tsr, np.random.default_rng(0)))))

    def test_exact_half_pi_pitch_bounds_are_valid(self):
        tsr = TSR(Bw=FULL)
        self.assertTrue(all(tsr.is_valid(sample_haar_xyzrpy(tsr, np.random.default_rng(0)))))

    @settings(max_examples=200, deadline=None)
    @given(
        xyz_lo=st.tuples(*[st.floats(-1, 1)] * 3),
        xyz_w=st.tuples(*[st.floats(0, 1)] * 3),
        roll_lo=st.floats(-2 * pi, 2 * pi),
        roll_w=st.floats(0, 2 * pi + 1),
        p=st.tuples(st.floats(-pi / 2, pi / 2), st.floats(-pi / 2, pi / 2)),
        yaw_lo=st.floats(-2 * pi, 2 * pi),
        yaw_w=st.floats(0, 2 * pi + 1),
        seed=st.integers(0, 2**32 - 1),
    )
    def test_samples_lie_in_the_represented_set(self, xyz_lo, xyz_w, roll_lo, roll_w, p, yaw_lo, yaw_w, seed):
        assume(not (p[0] == p[1] and abs(p[0]) == pi / 2))  # gimbal-lock point, rejected (#116)
        xyz = tuple((lo, lo + w) for lo, w in zip(xyz_lo, xyz_w))
        Bw = _box((roll_lo, min(p), yaw_lo), (roll_lo + roll_w, max(p), yaw_lo + yaw_w), xyz)
        tsr = TSR(Bw=Bw)
        rng = np.random.default_rng(seed)
        for _ in range(5):
            xyzrpy = sample_haar_xyzrpy(tsr, rng)
            self.assertTrue(all(tsr.is_valid(xyzrpy)), (Bw, xyzrpy))
            self.assertTrue(min(p) <= xyzrpy[4] <= max(p))
            tsr.to_transform(xyzrpy)  # raises if out of bounds


class TestHaarDistribution(unittest.TestCase):
    def test_full_box_rotation_angle_matches_haar(self):
        # Haar on SO(3): rotation angle θ has CDF (θ - sin θ) / π on [0, π].
        R = _rotations(TSR(Bw=FULL), seed=1)
        theta = np.arccos(np.clip((np.trace(R, axis1=1, axis2=2) - 1) / 2, -1, 1))
        self.assertGreater(stats.kstest(theta, lambda t: (t - np.sin(t)) / pi).pvalue, KS_P)
        # E[R] = 0 under Haar; each entry has std 1/sqrt(3N) ≈ 0.004.
        self.assertLess(np.abs(R.mean(axis=0)).max(), 0.02)

    def test_full_box_axis_image_is_uniform_on_sphere(self):
        # Archimedes: z of a uniform unit vector is uniform on [-1, 1].
        R = _rotations(TSR(Bw=FULL), seed=2)
        for col in range(3):
            self.assertGreater(stats.kstest(R[:, 2, col], stats.uniform(-1, 2).cdf).pvalue, KS_P, col)

    def test_coordinate_uniform_sampler_is_not_haar(self):
        # Contrast (#72): TSR.sample's uniform pitch piles directions at the poles.
        R = _rotations(TSR(Bw=FULL), seed=2, sampler=None)
        self.assertLess(stats.kstest(R[:, 2, 0], stats.uniform(-1, 2).cdf).pvalue, 1e-20)

    def test_restricted_pitch_is_sin_uniform(self):
        lo, hi = 0.2, 0.9
        tsr = TSR(Bw=_box((0.0, lo, 0.0), (2 * pi, hi, 2 * pi)))
        rng = np.random.default_rng(4)
        pitch = np.array([sample_haar_xyzrpy(tsr, rng)[4] for _ in range(N)])
        s_lo, s_hi = np.sin(lo), np.sin(hi)
        self.assertGreater(stats.kstest(np.sin(pitch), stats.uniform(s_lo, s_hi - s_lo).cdf).pvalue, KS_P)

    def test_translation_is_uniform_over_xyz_bounds(self):
        tsr = TSR(Bw=_box((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), xyz=((0.0, 1.0), (-2.0, 2.0), (0.5, 0.5))))
        rng = np.random.default_rng(5)
        xyz = np.array([sample_haar_xyzrpy(tsr, rng)[:3] for _ in range(N)])
        self.assertGreater(stats.kstest(xyz[:, 0], stats.uniform(0.0, 1.0).cdf).pvalue, KS_P)
        self.assertGreater(stats.kstest(xyz[:, 1], stats.uniform(-2.0, 4.0).cdf).pvalue, KS_P)
        np.testing.assert_array_equal(xyz[:, 2], 0.5)


class TestHaarEquivariance(unittest.TestCase):
    def test_reference_rotation_maps_samples_exactly(self):
        for seed in range(5):
            Q = np.eye(4)
            Q[:3, :3] = _random_rotation(seed)
            Q[:3, 3] = [0.3, -0.1, 0.7]
            base = sample_haar(TSR(Bw=FULL), np.random.default_rng(seed))
            moved = sample_haar(TSR(T0_w=Q, Bw=FULL), np.random.default_rng(seed))
            np.testing.assert_allclose(moved, Q @ base, atol=1e-12)

    def test_reference_rotation_preserves_uniformity(self):
        # Left-invariance of Haar: a rotated reference still gives uniform axes.
        Q = np.eye(4)
        Q[:3, :3] = _random_rotation(11)
        R = _rotations(TSR(T0_w=Q, Bw=FULL), seed=6)
        self.assertGreater(stats.kstest(R[:, 2, 0], stats.uniform(-1, 2).cdf).pvalue, KS_P)


if __name__ == "__main__":
    unittest.main()
