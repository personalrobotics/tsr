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
from hypothesis import example, given, settings
from hypothesis import strategies as st

from tsr.hands import ParallelJawGripper, default_registry

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


_HALF = np.pi / 2
_HALF_IN = float(np.nextafter(_HALF, 0.0))
_minor_angle = st.floats(-_HALF, _HALF, allow_nan=False)


@st.composite
def _minor_ranges(draw):
    """Ordered valid subintervals of the outer half, including singletons (#113)."""
    a = draw(_minor_angle)
    b = draw(st.one_of(st.just(a), _minor_angle))
    return (min(a, b), max(a, b))


def _side_angles(templates):
    return sorted({t.provenance.metadata["minor_angle"] for t in templates if t.provenance.mode == "side"})


class TestMinorAngleRangeProperty(unittest.TestCase):
    """Custom minor-angle intervals are sampled exactly and oracle-sound (#113)."""

    G = ParallelJawGripper(finger_length=0.08, max_aperture=0.30)
    R, r = 0.06, 0.02

    @settings(max_examples=60, deadline=None)
    @given(rng=_minor_ranges(), nm=st.integers(1, 5), c=st.floats(0.001, 0.01))
    @example(rng=(-_HALF, _HALF), nm=5, c=0.006)
    @example(rng=(-_HALF, -_HALF), nm=1, c=0.006)
    @example(rng=(_HALF, _HALF), nm=3, c=0.006)
    @example(rng=(-_HALF_IN, _HALF_IN), nm=2, c=0.006)
    @example(rng=(0.0, _HALF), nm=1, c=0.006)
    @example(rng=(-_HALF, 0.3), nm=4, c=0.001)
    def test_custom_range_is_sampled_exactly_and_sound(self, rng, nm, c):
        side = self.G.grasp_torus_side(self.R, self.r, clearance=c, n_minor=nm, minor_angle_range=rng)
        self.assertTrue(side)
        expected = [(rng[0] + rng[1]) / 2.0] if nm == 1 else np.linspace(rng[0], rng[1], nm)
        np.testing.assert_allclose(_side_angles(side), sorted(set(expected)), atol=1e-12)
        _assert_all_sound(self, Torus(self.R, self.r), side, self.G, c)

        # The combined API forwards the interval to side grasps only.
        combined = self.G.grasp_torus(self.R, self.r, clearance=c, n_minor=nm, minor_angle_range=rng)
        combined_side = [t for t in combined if t.provenance.mode == "side"]
        self.assertEqual(len(combined_side), len(side))
        for a, b in zip(side, combined_side):
            np.testing.assert_array_equal(a.Tw_e, b.Tw_e)
            np.testing.assert_array_equal(a.Bw, b.Bw)
        _assert_all_sound(self, Torus(self.R, self.r), combined, self.G, c)


class TestMinorAngleBoundaryPolicy(unittest.TestCase):
    """Exact closed-interval validation of the outer half, no tolerance (#113)."""

    G = ParallelJawGripper(finger_length=0.08, max_aperture=0.30)

    def test_exact_endpoints_and_inner_neighbours_are_valid(self):
        for rng in ((-_HALF, _HALF), (-_HALF, -_HALF), (_HALF, _HALF), (-_HALF_IN, _HALF_IN)):
            self.assertTrue(self.G.grasp_torus_side(0.06, 0.02, minor_angle_range=rng), rng)

    def test_outer_neighbours_raise(self):
        below = float(np.nextafter(-_HALF, -np.inf))
        above = float(np.nextafter(_HALF, np.inf))
        for rng in ((below, 0.0), (0.0, above), (below, above)):
            with self.assertRaises(ValueError):
                self.G.grasp_torus_side(0.06, 0.02, minor_angle_range=rng)
            with self.assertRaises(ValueError):
                self.G.grasp_torus(0.06, 0.02, minor_angle_range=rng)


class TestMinorAngleRangeInterface(unittest.TestCase):
    """minor_angle_range is keyword-only and reaches every public entry point (#112)."""

    G = ParallelJawGripper(finger_length=0.08, max_aperture=0.30)

    def test_positional_arguments_through_description_still_bind(self):
        templates = self.G.grasp_torus_side(
            0.06, 0.02, None, 2, 3, 0.006, (0.0, np.pi), "hand", "ring", "My Name", "My Desc"
        )
        self.assertEqual(len(templates), 2 * 2 * 3)
        for t in templates:
            self.assertEqual((t.subject, t.reference, t.description), ("hand", "ring", "My Desc"))
            self.assertTrue(t.name.startswith("My Name"), t.name)
            self.assertEqual((t.Bw[5, 0], t.Bw[5, 1]), (0.0, np.pi))

    def test_combined_positional_arguments_still_bind(self):
        templates = self.G.grasp_torus(0.06, 0.02, None, 2, 3, 0.006, (0.0, np.pi), "hand", "ring")
        self.assertTrue(templates)
        self.assertTrue(all((t.subject, t.reference) == ("hand", "ring") for t in templates))

    def test_minor_angle_range_is_keyword_only(self):
        with self.assertRaises(TypeError):
            self.G.grasp_torus_side(0.06, 0.02, None, 3, 5, None, (0.0, 2 * np.pi), "g", "t", "", "", (0.0, 0.5))
        with self.assertRaises(TypeError):
            self.G.grasp_torus(0.06, 0.02, None, 3, 5, None, (0.0, 2 * np.pi), "g", "t", (0.0, 0.5))

    def test_registry_route_forwards_minor_angle_range(self):
        gen = default_registry.get("parallel_jaw", "torus", "grasp")
        templates = gen(self.G, torus_radius=0.06, tube_radius=0.02, n_minor=1, minor_angle_range=(0.0, _HALF))
        angles = _side_angles(templates)
        self.assertEqual(len(angles), 1)
        self.assertAlmostEqual(angles[0], np.pi / 4, places=12)


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

    def test_excessive_clearance_logs_clearance_band_reason(self):
        # Ample reach (0.10 >= 0.02 + 0.03) but clearance > tube_radius empties the
        # far-surface limit: band [0.02, 0.01] -> [] with insufficient_clearance_band (#114).
        g = ParallelJawGripper(finger_length=0.10, max_aperture=0.50)
        with self.assertLogs("tsr.hands.base", level="DEBUG") as cm:
            self.assertEqual(g.grasp_torus_side(0.06, 0.02, clearance=0.03), [])
        self.assertTrue(any("insufficient_clearance_band" in m for m in cm.output))
        self.assertFalse(any("finger_too_short" in m for m in cm.output))

    def test_coincident_band_endpoints_emit_one_depth(self):
        # clearance == tube_radius with ample reach: band [r, 2r - c] = [r, r] (exact in
        # binary), a single deduplicated depth -> 2 flips x 1 depth x n_minor (#114).
        r = c = 0.03125
        g = ParallelJawGripper(finger_length=0.10, max_aperture=0.50)
        templates = g.grasp_torus_side(0.125, r, clearance=c, k=3, n_minor=1)
        self.assertEqual(len(templates), 2)
        _assert_all_sound(self, Torus(0.125, r), templates, g, c)

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
