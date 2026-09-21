#!/usr/bin/env python
# SPDX-License-Identifier: BSD-2-Clause
# Authors: Siddhartha Srinivasa and contributors to TSR

"""Torus side/span minor-angle, reach, clearance and coverage soundness (issue #71).

Every emitted torus template is certified against the independent #67 analytic
oracle. Coverage is the externally accessible outer half of the tube cross-section
(``minor_angle_range`` within ``[-pi/2, pi/2]``); inner-hole approaches are out of
scope. ``n_minor == 1`` is the centre of the requested range (the equator by
default), not an endpoint.
"""

import unittest

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from tsr.hands import ParallelJawGripper

from ._grasp_oracle import Torus, certify


def _bw_samples(t):
    lo, hi = t.Bw[:, 0], t.Bw[:, 1]
    mid = (lo + hi) / 2.0
    samples = [mid]
    for i in range(6):
        if hi[i] > lo[i]:
            for v in (lo[i], hi[i]):
                s = mid.copy()
                s[i] = v
                samples.append(s)
    return samples


def _assert_all_sound(testcase, torus, templates, gripper, clearance):
    for t in templates:
        prov = t.provenance
        testcase.assertEqual(prov.primitive, "torus")
        tsr = t.instantiate(np.eye(4))
        for xyz in _bw_samples(t):
            w = certify(
                torus,
                tsr.to_transform(xyz),
                finger_length=gripper.finger_length,
                max_aperture=gripper.max_aperture,
                preshape=float(t.preshape[0]),
                clearance=clearance,
                mode=prov.mode,
                approach=prov.approach,
                finger_orientation=prov.finger_orientation,
            )
            testcase.assertTrue(w.ok, (prov.mode, prov.approach, w.failed))


def _rotz(theta):
    c, s = np.cos(theta), np.sin(theta)
    T = np.eye(4)
    T[:2, :2] = [[c, -s], [s, c]]
    return T


class TestMinorAngleSemantics(unittest.TestCase):
    """n_minor and the minor-angle range are deliberate and documented (#71)."""

    G = ParallelJawGripper(finger_length=0.08, max_aperture=0.30)

    def test_n_minor_one_is_the_range_centre(self):
        angles = {t.provenance.metadata["minor_angle"] for t in self.G.grasp_torus_side(0.06, 0.02, n_minor=1)}
        self.assertEqual(angles, {0.0})  # default range centre = equator

    def test_n_minor_one_of_custom_range_is_its_centre(self):
        angles = {
            t.provenance.metadata["minor_angle"]
            for t in self.G.grasp_torus_side(0.06, 0.02, n_minor=1, minor_angle_range=(0.0, np.pi / 2))
        }
        self.assertEqual(len(angles), 1)
        self.assertAlmostEqual(next(iter(angles)), np.pi / 4, places=12)

    def test_coverage_matches_represented_set(self):
        # The emitted minor angles are exactly linspace over the requested range.
        angles = sorted({t.provenance.metadata["minor_angle"] for t in self.G.grasp_torus_side(0.06, 0.02, n_minor=5)})
        np.testing.assert_allclose(angles, np.linspace(-np.pi / 2, np.pi / 2, 5), atol=1e-12)

    def test_inner_hole_range_is_out_of_scope(self):
        for bad in ((-np.pi, np.pi), (-np.pi, 0.0), (0.0, np.pi)):
            with self.assertRaises(ValueError):
                self.G.grasp_torus_side(0.06, 0.02, minor_angle_range=bad)


class TestTorusSoundnessProperty(unittest.TestCase):
    """Every side and span template has a valid oracle witness (#71)."""

    @settings(max_examples=70, deadline=None)
    @given(
        fl=st.floats(0.02, 0.2),
        ma=st.floats(0.05, 0.5),
        big=st.floats(0.03, 0.15),
        ratio=st.floats(0.05, 0.7),
        c=st.floats(0.0, 0.02),
        k=st.integers(1, 3),
        nm=st.integers(1, 5),
    )
    def test_side_and_span_are_oracle_sound(self, fl, ma, big, ratio, c, k, nm):
        R, r = big, big * ratio  # r < R (valid torus)
        g = ParallelJawGripper(finger_length=fl, max_aperture=ma)
        torus = Torus(R, r)
        _assert_all_sound(self, torus, g.grasp_torus_side(R, r, clearance=c, k=k, n_minor=nm), g, c)
        _assert_all_sound(self, torus, g.grasp_torus_span(R, r, clearance=c, k=k), g, c)

    @settings(max_examples=30, deadline=None)
    @given(R=st.floats(0.04, 0.12), ratio=st.floats(0.1, 0.5), theta=st.floats(-3.0, 3.0))
    def test_rotational_equivariance_about_axis(self, R, ratio, theta):
        # Rotating the whole grasp about the torus z-axis is a symmetry.
        r, c = R * ratio, 0.006
        g = ParallelJawGripper(finger_length=0.08, max_aperture=0.50)
        torus = Torus(R, r)
        rot = _rotz(theta)
        for prov_templates, mode in (
            (g.grasp_torus_side(R, r, clearance=c), "side"),
            (g.grasp_torus_span(R, r, clearance=c), "span"),
        ):
            for t in prov_templates:
                p = t.provenance
                pose = rot @ t.instantiate(np.eye(4)).to_transform(np.zeros(6))
                w = certify(
                    torus,
                    pose,
                    finger_length=0.08,
                    max_aperture=0.50,
                    preshape=float(t.preshape[0]),
                    clearance=c,
                    mode=p.mode,
                    approach=p.approach,
                    finger_orientation=p.finger_orientation,
                )
                self.assertTrue(w.ok, (mode, theta, w.failed))


class TestTorusReach(unittest.TestCase):
    """Reach/clearance failures return [] with a precise diagnostic (#71)."""

    def _counts(self, fl, R, r, c):
        g = ParallelJawGripper(finger_length=fl, max_aperture=0.50)
        return len(g.grasp_torus_side(R, r, clearance=c)), len(g.grasp_torus_span(R, r, clearance=c))

    def test_finger_too_short_returns_empty(self):
        # finger_length < tube_radius + clearance -> [] for both side and span.
        g = ParallelJawGripper(finger_length=0.02, max_aperture=0.50)
        self.assertEqual(g.grasp_torus_side(0.06, 0.03, clearance=0.006), [])
        self.assertEqual(g.grasp_torus_span(0.06, 0.03, clearance=0.006), [])

    def test_finger_too_short_logs_reason(self):
        g = ParallelJawGripper(finger_length=0.02, max_aperture=0.50)
        with self.assertLogs("tsr.hands.base", level="DEBUG") as cm:
            g.grasp_torus_side(0.06, 0.03, clearance=0.006)
        self.assertTrue(any("finger_too_short" in m for m in cm.output))

    def test_reach_boundary_and_neighbours(self):
        # The reach band [tube_radius, min(2r, L) - clearance] is nonempty iff
        # L >= tube_radius + clearance. Below the boundary both families are empty;
        # above, both emit sound templates.
        R, r, c = 0.06, 0.02, 0.006
        boundary = r + c
        self.assertEqual(self._counts(boundary - 1e-4, R, r, c), (0, 0))
        ns, nsp = self._counts(boundary + 1e-4, R, r, c)
        self.assertGreater(ns, 0)
        self.assertGreater(nsp, 0)
        g = ParallelJawGripper(finger_length=boundary + 1e-4, max_aperture=0.50)
        _assert_all_sound(self, Torus(R, r), g.grasp_torus_side(R, r, clearance=c), g, c)
        _assert_all_sound(self, Torus(R, r), g.grasp_torus_span(R, r, clearance=c), g, c)

    def test_boundary_templates_are_sound(self):
        R, r, c = 0.06, 0.02, 0.006
        g = ParallelJawGripper(finger_length=r + c, max_aperture=0.50)
        torus = Torus(R, r)
        _assert_all_sound(self, torus, g.grasp_torus_side(R, r, clearance=c), g, c)
        _assert_all_sound(self, torus, g.grasp_torus_span(R, r, clearance=c), g, c)


class TestTorusSymmetry(unittest.TestCase):
    """Mirror symmetry across the equatorial plane (#71)."""

    G = ParallelJawGripper(finger_length=0.08, max_aperture=0.50)

    def test_span_top_and_bottom_are_balanced(self):
        templates = self.G.grasp_torus_span(0.06, 0.02, clearance=0.006)
        top = [t for t in templates if t.provenance.approach == "+z"]
        bot = [t for t in templates if t.provenance.approach == "-z"]
        self.assertEqual(len(top), len(bot))

    def test_side_minor_angles_are_symmetric_about_equator(self):
        angles = sorted({t.provenance.metadata["minor_angle"] for t in self.G.grasp_torus_side(0.06, 0.02, n_minor=5)})
        np.testing.assert_allclose(angles, [-a for a in reversed(angles)], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
