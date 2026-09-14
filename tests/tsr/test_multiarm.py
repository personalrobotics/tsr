#!/usr/bin/env python
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Tests for the three- and four-arm similarity-transform TSRs."""

import unittest

import numpy as np
from gafro import SimilarityTransformation

from tsr.multiarm import (
    CircleTSR,
    SphereTSR,
    circle_from_center_radius_normal,
    decompose_similarity,
    dilation_log,
    dilator_from_log,
    sphere_from_center_radius,
)


class TestDilationLog(unittest.TestCase):
    """dilation is the log of the scale factor."""

    def test_identity_is_zero(self):
        self.assertAlmostEqual(dilation_log(dilator_from_log(0.0)), 0.0, places=12)

    def test_roundtrips(self):
        for value in (-2.0, -0.5, 0.0, 0.25, 1.5):
            self.assertAlmostEqual(dilation_log(dilator_from_log(value)), value, places=10)

    def test_matches_scale_factor(self):
        """exp(dilation) is the raw scale, so doubling gives log(2)."""
        for scale in (0.25, 0.5, 1.0, 2.0, 4.0):
            self.assertAlmostEqual(np.exp(dilation_log(dilator_from_log(np.log(scale)))),
                                   scale, places=10)

    def test_is_signed_and_symmetric(self):
        """Halving and doubling are equally far from unity, with opposite sign."""
        self.assertAlmostEqual(dilation_log(dilator_from_log(np.log(2.0))),
                               -dilation_log(dilator_from_log(np.log(0.5))), places=10)


class TestDegenerateDilation(unittest.TestCase):
    """A degenerate primitive must raise, not silently yield inf/NaN."""

    def test_raises_at_the_atanh_pole(self):
        """Four coplanar end-effectors have no circumsphere; the scale diverges."""
        class _PoleDilator:
            @staticmethod
            def to_array():
                return np.array([1.0, 1.0])

        with self.assertRaises(ValueError) as caught:
            dilation_log(_PoleDilator())
        self.assertIn("degenerate dilation", str(caught.exception))

    def test_raises_on_zero_leading_coefficient(self):
        class _ZeroDilator:
            @staticmethod
            def to_array():
                return np.array([0.0, 1.0])

        with self.assertRaises(ValueError):
            dilation_log(_ZeroDilator())

    def test_detects_a_wholly_non_finite_decomposition(self):
        """A perfectly level four-point grasp makes the *whole* decomposition NaN.

        Checking only the dilator ratio misses it: the translator and rotor come
        back NaN too, and NaN comparisons are False, so the ratio test passes.
        """
        class _NanPart:
            @staticmethod
            def x():
                return float("nan")

            y = z = x

            @staticmethod
            def to_array():
                return np.array([float("nan"), float("nan")])

            @staticmethod
            def log():
                return _NanPart()

        class _NanDecomposition:
            get_translator = get_rotor = get_dilator = staticmethod(lambda: _NanPart())

        class _NanSimilarity:
            get_canonical_decomposition = staticmethod(lambda: _NanDecomposition())

        from tsr.multiarm import is_degenerate

        assert is_degenerate(_NanSimilarity())

    def test_ordinary_scales_are_unaffected(self):
        for scale in (0.1, 1.0, 9.0):
            self.assertTrue(np.isfinite(dilation_log(dilator_from_log(np.log(scale)))))


class TestPrimitives(unittest.TestCase):
    """The canonical primitives recover the centre / radius / normal asked for."""

    def test_sphere_between_recovers_translation_and_scale(self):
        unit = sphere_from_center_radius([0, 0, 0], 1.0)
        target = sphere_from_center_radius([1, 2, 3], 2.0)
        translation, _rotor, dilation = decompose_similarity(
            SimilarityTransformation.between(unit, target))
        np.testing.assert_allclose(translation, [1, 2, 3], atol=1e-9)
        self.assertAlmostEqual(float(np.exp(dilation)), 2.0, places=9)

    def test_circle_between_recovers_translation_and_scale(self):
        unit = circle_from_center_radius_normal([0, 0, 0], 1.0, [0, 0, 1])
        target = circle_from_center_radius_normal([1, 2, 3], 2.0, [0, 0, 1])
        translation, _rotor, dilation = decompose_similarity(
            SimilarityTransformation.between(unit, target))
        np.testing.assert_allclose(translation, [1, 2, 3], atol=1e-9)
        self.assertAlmostEqual(float(np.exp(dilation)), 2.0, places=9)

    def test_circle_normal_has_no_spin_component(self):
        """The rotor carrying one circle onto another never spins about the normal.

        This is what makes the circle a 6-DOF region rather than 7: the first
        rotor-bivector component is structurally zero.
        """
        unit = circle_from_center_radius_normal([0, 0, 0], 1.0, [0, 0, 1])
        for normal in ([0, 0, 1], [1, 0, 0], [0, 1, 0], [1, 1, 1], [-1, 2, 0.5]):
            _t, rotor_log, _d = decompose_similarity(SimilarityTransformation.between(
                unit, circle_from_center_radius_normal([0, 0, 0], 1.0, normal)))
            self.assertAlmostEqual(rotor_log[0], 0.0, places=9, msg=f"normal={normal}")


class TestSphereTSR(unittest.TestCase):
    """Four-arm region: [tx, ty, tz, dilation]; no rotation coordinates."""

    def test_dof_and_labels(self):
        tsr = SphereTSR()
        self.assertEqual(tsr.dof, 4)
        self.assertEqual(tsr._LABELS, ("tx", "ty", "tz", "dilation"))

    def test_rejects_wrong_shape(self):
        with self.assertRaises(ValueError):
            SphereTSR(Bw=np.zeros((6, 2)))

    def test_rejects_inverted_bounds(self):
        with self.assertRaises(ValueError):
            SphereTSR(Bw=np.array([[1.0, -1.0], [0, 0], [0, 0], [0, 0]]))

    def test_bw_roundtrip(self):
        tsr = SphereTSR(Bw=np.array([[-1.0, 1.0]] * 4))
        rng = np.random.default_rng(0)
        for _ in range(100):
            bw = rng.uniform(-1, 1, 4)
            np.testing.assert_allclose(tsr.to_bw(tsr.to_transform(bw)), bw, atol=1e-12)

    def test_distance_zero_inside(self):
        tsr = SphereTSR(Bw=np.array([[-0.1, 0.1]] * 3 + [[0.0, 0.0]]))
        pose = tsr.to_transform(np.array([0.05, 0.0, 0.0, 0.0]))
        self.assertAlmostEqual(tsr.distance(pose)[0], 0.0, places=9)
        self.assertTrue(tsr.contains(pose))

    def test_distance_is_overshoot_outside(self):
        tsr = SphereTSR(Bw=np.array([[-0.1, 0.1]] * 3 + [[0.0, 0.0]]))
        pose = tsr.to_transform(np.array([0.5, 0.0, 0.0, 0.0]))
        self.assertAlmostEqual(tsr.distance(pose)[0], 0.4, places=6)
        self.assertFalse(tsr.contains(pose))

    def test_dilation_violation_is_measured(self):
        """A sphere of the wrong size is outside, even at the right centre."""
        tsr = SphereTSR(Bw=np.array([[0.0, 0.0]] * 3 + [[-0.1, 0.1]]))
        pose = tsr.to_transform(np.array([0.0, 0.0, 0.0, 0.6]))
        self.assertAlmostEqual(tsr.distance(pose)[0], 0.5, places=6)

    def test_witness_is_clamped_into_region(self):
        tsr = SphereTSR(Bw=np.array([[-0.1, 0.1]] * 3 + [[0.0, 0.0]]))
        _dist, witness = tsr.distance(tsr.to_transform(np.array([0.5, 0.0, 0.0, 0.0])))
        self.assertAlmostEqual(witness[0], 0.1, places=9)
        self.assertTrue(tsr.contains(tsr.to_transform(witness), tolerance=1e-6))

    def test_samples_lie_inside(self):
        tsr = SphereTSR(Bw=np.array([[-0.2, 0.2], [-0.1, 0.3], [0.0, 0.5], [-0.4, 0.4]]))
        for _ in range(50):
            self.assertTrue(tsr.contains(tsr.sample(), tolerance=1e-6))

    def test_volume_sums_widths(self):
        tsr = SphereTSR(Bw=np.array([[-0.5, 0.5], [0, 0], [0, 0], [-1.0, 1.0]]))
        self.assertAlmostEqual(tsr.volume, 3.0, places=9)


class TestCircleTSR(unittest.TestCase):
    """Three-arm region: [tx, ty, tz, dilation, n1, n2]; spin dropped."""

    def test_dof_and_labels(self):
        tsr = CircleTSR()
        self.assertEqual(tsr.dof, 6)
        self.assertEqual(tsr._LABELS, ("tx", "ty", "tz", "dilation", "n1", "n2"))

    def test_rejects_wrong_shape(self):
        with self.assertRaises(ValueError):
            CircleTSR(Bw=np.zeros((4, 2)))

    def test_bw_roundtrip(self):
        tsr = CircleTSR(Bw=np.array([[-1.0, 1.0]] * 6))
        rng = np.random.default_rng(1)
        for _ in range(100):
            bw = rng.uniform(-1, 1, 6)
            np.testing.assert_allclose(tsr.to_bw(tsr.to_transform(bw)), bw, atol=1e-11)

    def test_plane_rotation_is_constrained(self):
        """The two retained rotation coordinates orient the circle's plane."""
        tsr = CircleTSR(Bw=np.array([[0.0, 0.0]] * 4 + [[-0.05, 0.05]] * 2))
        tilted = tsr.to_transform(np.array([0, 0, 0, 0, 0.5, 0.0]))
        self.assertGreater(tsr.distance(tilted)[0], 0.4)
        self.assertFalse(tsr.contains(tilted))

    def test_angular_rows_clamped_to_pi(self):
        tsr = CircleTSR(Bw=np.array([[0.0, 0.0]] * 4 + [[-10.0, 10.0]] * 2))
        np.testing.assert_allclose(tsr._Bw_cont[4], [-np.pi, np.pi])
        np.testing.assert_allclose(tsr._Bw_cont[5], [-np.pi, np.pi])

    def test_samples_lie_inside(self):
        tsr = CircleTSR(Bw=np.array([[-0.2, 0.2]] * 3 + [[-0.3, 0.3], [-0.4, 0.4], [-0.4, 0.4]]))
        for _ in range(50):
            self.assertTrue(tsr.contains(tsr.sample(), tolerance=1e-6))

    def test_volume_clamps_angular_rows(self):
        tsr = CircleTSR(Bw=np.array([[0.0, 0.0]] * 4 + [[-10.0, 10.0]] * 2))
        self.assertAlmostEqual(tsr.volume, 4.0 * np.pi, places=9)


class TestSampleBw(unittest.TestCase):
    """Partially specified samples keep the finite components."""

    def test_fixed_components_are_kept(self):
        tsr = SphereTSR(Bw=np.array([[-1.0, 1.0]] * 4))
        drawn = tsr.sample_bw(np.array([0.3, np.nan, np.nan, -0.2]))
        self.assertAlmostEqual(drawn[0], 0.3, places=12)
        self.assertAlmostEqual(drawn[3], -0.2, places=12)

    def test_free_components_stay_in_bounds(self):
        tsr = SphereTSR(Bw=np.array([[-0.25, 0.75]] * 4))
        for _ in range(100):
            drawn = tsr.sample_bw()
            self.assertTrue(np.all(drawn >= -0.25 - 1e-12))
            self.assertTrue(np.all(drawn <= 0.75 + 1e-12))


if __name__ == "__main__":
    unittest.main()
