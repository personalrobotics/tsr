#!/usr/bin/env python
# SPDX-License-Identifier: BSD-2-Clause
# Authors: Siddhartha Srinivasa and contributors to TSR

"""Sphere-grasp coverage vs. sampling distribution (issue #72).

Haar samples of sphere-grasp templates stay in the represented set and keep an
oracle witness; their approach directions are uniform by area over the sphere
(default ``angle_range``) or over the lune ``azimuth ∈ angle_range``. Fixed
seeds keep the KS tests deterministic.
"""

import unittest

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import stats

from tsr import sample_haar
from tsr.hands import ParallelJawGripper

from ._grasp_oracle import Sphere, certify

N = 20000
KS_P = 1e-3


def _directions(tsr, seed, n=N):
    """Unit vectors from the sphere centre to the sampled hand position."""
    rng = np.random.default_rng(seed)
    p = np.array([sample_haar(tsr, rng)[:3, 3] for _ in range(n)])
    return p / np.linalg.norm(p, axis=1, keepdims=True)


class TestSphereHaarSoundness(unittest.TestCase):
    @settings(max_examples=60, deadline=None)
    @given(
        # Ranges chosen so the band [r, min(fl, 2r) - c] is never empty.
        fl=st.floats(0.05, 0.2),
        radius=st.floats(0.01, 0.04),
        c=st.floats(0.001, 0.01),
        yaw=st.tuples(st.floats(-np.pi, np.pi), st.floats(0.0, 2 * np.pi)),
        seed=st.integers(0, 2**32 - 1),
    )
    def test_haar_samples_have_oracle_witness(self, fl, radius, c, yaw, seed):
        g = ParallelJawGripper(finger_length=fl, max_aperture=0.3)
        angle_range = (yaw[0], yaw[0] + yaw[1])
        rng = np.random.default_rng(seed)
        templates = g.grasp_sphere(radius, clearance=c, angle_range=angle_range)
        self.assertTrue(templates)
        for t in templates:
            tsr = t.instantiate(np.eye(4))
            for _ in range(5):
                w = certify(
                    Sphere(radius),
                    sample_haar(tsr, rng),
                    finger_length=fl,
                    max_aperture=0.3,
                    preshape=float(t.preshape[0]),
                    clearance=c,
                    mode="surface",
                    approach="radial",
                    finger_orientation="diameter",
                )
                self.assertTrue(w.ok, w.failed)


class TestSphereApproachDistribution(unittest.TestCase):
    G = ParallelJawGripper(finger_length=0.08, max_aperture=0.3)

    def test_default_range_directions_are_uniform_on_sphere(self):
        tsr = self.G.grasp_sphere(0.04, k=1)[0].instantiate(np.eye(4))
        d = _directions(tsr, seed=1)
        self.assertGreater(stats.kstest(d[:, 2], stats.uniform(-1, 2).cdf).pvalue, KS_P)
        az = np.arctan2(d[:, 1], d[:, 0])
        self.assertGreater(stats.kstest(az, stats.uniform(-np.pi, 2 * np.pi).cdf).pvalue, KS_P)

    def test_default_sampler_concentrates_at_poles(self):
        # Contrast: coordinate-uniform TSR.sample is not area-uniform (#72).
        tsr = self.G.grasp_sphere(0.04, k=1)[0].instantiate(np.eye(4))
        rng = np.random.default_rng(1)
        p = np.array([tsr.sample(rng=rng)[:3, 3] for _ in range(N)])
        z = p[:, 2] / np.linalg.norm(p, axis=1)
        self.assertLess(stats.kstest(z, stats.uniform(-1, 2).cdf).pvalue, 1e-20)

    def test_restricted_range_is_an_area_uniform_lune(self):
        lo, hi = -1.0, 2.0
        tsr = self.G.grasp_sphere(0.04, k=1, angle_range=(lo, hi))[0].instantiate(np.eye(4))
        d = _directions(tsr, seed=2)
        az = np.arctan2(d[:, 1], d[:, 0])
        self.assertTrue(np.all((az >= lo - 1e-9) & (az <= hi + 1e-9)))
        # A lune, not a cap: every elevation is present, and z is uniform on [-1, 1].
        self.assertGreater(stats.kstest(d[:, 2], stats.uniform(-1, 2).cdf).pvalue, KS_P)
        self.assertGreater(stats.kstest(az, stats.uniform(lo, hi - lo).cdf).pvalue, KS_P)


class TestSphereCoverageDescription(unittest.TestCase):
    G = ParallelJawGripper(finger_length=0.08, max_aperture=0.3)

    def test_default_range_is_described_as_full_so3(self):
        self.assertTrue(all("full SO(3)" in t.description for t in self.G.grasp_sphere(0.04)))

    def test_restricted_range_is_described_as_a_lune(self):
        for t in self.G.grasp_sphere(0.04, angle_range=(0.0, np.pi / 2)):
            self.assertNotIn("SO(3)", t.description)
            self.assertIn("azimuth lune 0°–90°", t.description)


if __name__ == "__main__":
    unittest.main()
