#!/usr/bin/env python
# SPDX-License-Identifier: BSD-2-Clause
# Authors: Siddhartha Srinivasa and contributors to TSR

"""``depth_index`` / ``depth_count`` semantics for every grasp factory (issue #81).

``depth_count`` is the number of **distinct** depth levels emitted for a family
(``<= k``), and ``depth_index`` orders them shallow to deep. A family is every
provenance field except the depth fields, so it holds exactly one template per
depth level. A one-point feasible band gives ``depth_count == 1``; an empty band
emits no templates (and so no provenance records).
"""

import unittest
from collections import defaultdict

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from tsr.hands import GripperBase, ParallelJawGripper


def _families(templates):
    fam = defaultdict(list)
    for t in templates:
        p = t.provenance
        key = (p.primitive, p.mode, p.approach, p.finger_orientation, p.symmetry, tuple(sorted(p.metadata.items())))
        fam[key].append(p)
    return fam


def _assert_depth_semantics(testcase, templates, k):
    for key, provs in _families(templates).items():
        n = len(provs)
        testcase.assertLessEqual(n, k, key)
        testcase.assertEqual({p.depth_count for p in provs}, {n}, key)
        testcase.assertEqual(sorted(p.depth_index for p in provs), list(range(n)), key)
        depths = [p.depth for p in sorted(provs, key=lambda p: p.depth_index)]
        testcase.assertTrue(all(a < b for a, b in zip(depths, depths[1:])), (key, depths))


def _all_factories(g, s, k, c):
    """Every grasp factory, including combined entry points, at scale ``s``."""
    return {
        "grasp_cylinder": lambda: g.grasp_cylinder(0.5 * s, 2.0 * s, k=k, clearance=c),
        "grasp_cylinder_side": lambda: g.grasp_cylinder_side(0.5 * s, 2.0 * s, k=k, clearance=c),
        "grasp_cylinder_top": lambda: g.grasp_cylinder_top(0.5 * s, 2.0 * s, k=k, clearance=c),
        "grasp_cylinder_bottom": lambda: g.grasp_cylinder_bottom(0.5 * s, 2.0 * s, k=k, clearance=c),
        "grasp_box": lambda: g.grasp_box(s, 1.3 * s, 0.8 * s, k=k, clearance=c),
        "grasp_box_top": lambda: g.grasp_box_top(s, 1.3 * s, 0.8 * s, k=k, clearance=c),
        "grasp_box_bottom": lambda: g.grasp_box_bottom(s, 1.3 * s, 0.8 * s, k=k, clearance=c),
        "grasp_box_face_x": lambda: g.grasp_box_face_x(s, 1.3 * s, 0.8 * s, k=k, clearance=c),
        "grasp_box_face_y": lambda: g.grasp_box_face_y(s, 1.3 * s, 0.8 * s, k=k, clearance=c),
        "grasp_sphere": lambda: g.grasp_sphere(0.5 * s, k=k, clearance=c),
        "grasp_torus": lambda: g.grasp_torus(s, 0.3 * s, k=k, n_minor=3, clearance=c),
        "grasp_torus_side": lambda: g.grasp_torus_side(s, 0.3 * s, k=k, n_minor=3, clearance=c),
        "grasp_torus_span": lambda: g.grasp_torus_span(s, 0.3 * s, k=k, clearance=c),
    }


class TestFactoryCoverage(unittest.TestCase):
    def test_every_public_grasp_factory_is_exercised(self):
        public = {n for n in dir(GripperBase) if n.startswith("grasp_")}
        self.assertEqual(public, set(_all_factories(None, 1.0, 1, 0.0)))


class TestDepthProvenanceProperty(unittest.TestCase):
    @settings(max_examples=80, deadline=None)
    @given(
        fl=st.floats(0.01, 0.2),
        s=st.floats(0.005, 0.12),
        k=st.integers(1, 6),
        c=st.floats(0.0, 0.03),
    )
    def test_every_factory_reports_distinct_ordered_levels(self, fl, s, k, c):
        g = ParallelJawGripper(finger_length=fl, max_aperture=0.4)
        for call in _all_factories(g, s, k, c).values():
            _assert_depth_semantics(self, call(), k)


class TestOnePointBand(unittest.TestCase):
    """At an exact equality boundary every family has one depth, depth_count == 1."""

    def _assert_one_level(self, templates):
        self.assertTrue(templates)
        for provs in _families(templates).values():
            self.assertEqual([(p.depth_index, p.depth_count) for p in provs], [(0, 1)])
        _assert_depth_semantics(self, templates, 5)

    def test_radial_families(self):
        # Band [r, min(L, 2r) - c] with c == r (exact in binary): cylinder side, sphere, torus side.
        r = c = 0.03125
        g = ParallelJawGripper(finger_length=0.125, max_aperture=0.5)
        self._assert_one_level(g.grasp_cylinder_side(r, 0.25, k=5, clearance=c))
        self._assert_one_level(g.grasp_sphere(r, k=5, clearance=c))
        self._assert_one_level(g.grasp_torus_side(0.125, r, k=5, clearance=c))

    def test_cap_families(self):
        # Band [c, L - c] with L == 2c: cylinder top/bottom.
        g = ParallelJawGripper(finger_length=0.0625, max_aperture=0.5)
        c = 0.03125
        self._assert_one_level(g.grasp_cylinder_top(0.03, 0.12, k=5, clearance=c))
        self._assert_one_level(g.grasp_cylinder_bottom(0.03, 0.12, k=5, clearance=c))

    def test_box_families(self):
        # Band [c, min(L, extent) - c] with extent == 2c on every face.
        e = 0.0625
        c = e / 2
        g = ParallelJawGripper(finger_length=0.125, max_aperture=0.5)
        for f in (g.grasp_box_top, g.grasp_box_bottom, g.grasp_box_face_x, g.grasp_box_face_y):
            with self.subTest(f.__name__):
                self._assert_one_level(f(e, e, e, k=5, clearance=c))

    def test_torus_span(self):
        # Band [r, L - c] with L == r + c.
        r, c = 0.03125, 0.015625
        g = ParallelJawGripper(finger_length=r + c, max_aperture=0.5)
        self._assert_one_level(g.grasp_torus_span(0.0625, r, k=5, clearance=c))


class TestUlpWideBand(unittest.TestCase):
    def test_usable_depths_are_distinct_for_an_ulp_wide_band(self):
        # np.linspace over a 2-ulp band rounds k samples onto the same floats (#81).
        lo = 0.04
        hi = float(np.nextafter(np.nextafter(lo, 1.0), 1.0))
        depths = GripperBase._usable_depths(lo, hi, 5)
        self.assertEqual(depths[0], lo)
        self.assertEqual(depths[-1], hi)
        self.assertTrue(np.all(np.diff(depths) > 0))
        self.assertLessEqual(len(depths), 3)

    def test_factory_reports_distinct_levels_for_an_ulp_wide_band(self):
        # finger_length one ulp above 2c: the cap band [c, L - c] is 2 ulps wide, so
        # k = 10 linspace samples collide; provenance must count distinct levels.
        c = 0.04
        g = ParallelJawGripper(finger_length=float(np.nextafter(2 * c, 1.0)), max_aperture=0.3)
        templates = g.grasp_cylinder_top(0.03, 0.12, k=10, clearance=c)
        self.assertTrue(templates)
        self.assertLess(len(templates), 10)
        _assert_depth_semantics(self, templates, 10)

    def test_usable_depths_boundary_policy_is_unchanged(self):
        self.assertIsNone(GripperBase._usable_depths(0.05, 0.04, 3))
        np.testing.assert_array_equal(GripperBase._usable_depths(0.04, 0.04, 3), [0.04])
        np.testing.assert_array_equal(GripperBase._usable_depths(0.0, 0.2, 3), [0.0, 0.1, 0.2])


class TestEmptyBandEmitsNothing(unittest.TestCase):
    def test_empty_bands_return_no_records(self):
        g = ParallelJawGripper(finger_length=0.02, max_aperture=0.5)
        c = 0.015  # L < 2c and L < r + c: every depth band is empty
        for name, call in _all_factories(g, 0.06, 3, c).items():
            with self.subTest(name):
                self.assertEqual(call(), [])


if __name__ == "__main__":
    unittest.main()
