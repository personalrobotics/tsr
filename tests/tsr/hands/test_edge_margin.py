#!/usr/bin/env python
# SPDX-License-Identifier: BSD-2-Clause
# Authors: Siddhartha Srinivasa and contributors to TSR

"""Edge-facing band limits keep contacts off the edges (issue #121).

``clearance = 0`` is a valid request (#104), but a closed band whose limits are the
edges themselves puts contacts exactly on an edge: zero insertion, or a contact on
the far face, a lateral slide edge, or a cylinder rim, where the surface normal is
ambiguous. Edge-facing limits now use ``max(clearance, 2 * _length_atol(scale))``,
mirroring the scale-aware straddle margin (#107). Positive clearances above the
tolerance are unaffected.
"""

import unittest

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from tsr.hands import ParallelJawGripper

from ._grasp_oracle import Box, Cylinder, certify, length_atol


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


def _assert_all_sound(testcase, prim, templates, gripper, clearance):
    for t in templates:
        p = t.provenance
        tsr = t.instantiate(np.eye(4))
        for xi in _bw_samples(t):
            w = certify(
                prim,
                tsr.to_transform(xi),
                finger_length=gripper.finger_length,
                max_aperture=gripper.max_aperture,
                preshape=float(t.preshape[0]),
                clearance=clearance,
                mode=p.mode,
                approach=p.approach,
                finger_orientation=p.finger_orientation,
            )
            testcase.assertTrue(w.ok, (p.mode, p.approach, p.finger_orientation, p.depth, w.failed))


G = ParallelJawGripper(finger_length=0.08, max_aperture=0.2)


class TestZeroClearanceReproduction(unittest.TestCase):
    """#121: at clearance 0 every box and cylinder family stays oracle-sound."""

    def test_box_top_reproduction(self):
        templates = G.grasp_box_top(0.05, 0.05, 0.05, k=3, clearance=0.0, preshape=0.07)
        self.assertTrue(templates)
        _assert_all_sound(self, Box(0.05, 0.05, 0.05), templates, G, 0.0)

    def test_every_box_family(self):
        prim = Box(0.05, 0.06, 0.07)
        for f in (G.grasp_box_top, G.grasp_box_bottom, G.grasp_box_face_x, G.grasp_box_face_y, G.grasp_box):
            with self.subTest(f.__name__):
                templates = f(0.05, 0.06, 0.07, k=3, clearance=0.0, preshape=0.09)
                self.assertTrue(templates)
                _assert_all_sound(self, prim, templates, G, 0.0)

    def test_every_cylinder_family(self):
        prim = Cylinder(0.02, 0.12)
        for f in (G.grasp_cylinder_side, G.grasp_cylinder_top, G.grasp_cylinder_bottom, G.grasp_cylinder):
            with self.subTest(f.__name__):
                templates = f(0.02, 0.12, k=3, clearance=0.0, preshape=0.06)
                self.assertTrue(templates)
                _assert_all_sound(self, prim, templates, G, 0.0)

    def test_sub_tolerance_clearance(self):
        c = 1e-15  # far below length_atol; indistinguishable from zero
        _assert_all_sound(self, Box(0.05, 0.06, 0.07), G.grasp_box(0.05, 0.06, 0.07, clearance=c, preshape=0.09), G, c)
        _assert_all_sound(self, Cylinder(0.02, 0.12), G.grasp_cylinder(0.02, 0.12, clearance=c, preshape=0.06), G, c)


class TestMarginValue(unittest.TestCase):
    """The floor is twice the contract tolerance; larger clearances win."""

    def test_zero_clearance_uses_twice_the_tolerance(self):
        floor = 2 * length_atol(0.07)  # box scale = max dimension
        depths = sorted(
            {t.provenance.depth for t in G.grasp_box_top(0.05, 0.06, 0.07, k=3, clearance=0.0, preshape=0.09)}
        )
        self.assertAlmostEqual(depths[0], floor, places=15)

    def test_slide_band_uses_the_margin(self):
        floor = 2 * length_atol(0.07)
        t = G.grasp_box_top(0.05, 0.06, 0.07, k=1, clearance=0.0, preshape=0.09)[0]
        half = max(t.Bw[i, 1] for i in range(3))
        self.assertAlmostEqual(half, 0.06 / 2 - floor, places=15)

    def test_cylinder_side_height_band_uses_the_margin(self):
        floor = 2 * length_atol(0.12)
        t = G.grasp_cylinder_side(0.02, 0.12, k=1, clearance=0.0, preshape=0.06)[0]
        self.assertAlmostEqual(t.Bw[2, 1], 0.12 / 2 - floor, places=15)

    def test_ordinary_clearance_is_unchanged(self):
        # A clearance well above the tolerance is used verbatim.
        c = 0.006
        depths = sorted({t.provenance.depth for t in G.grasp_box_top(0.05, 0.06, 0.07, k=3, clearance=c)})
        np.testing.assert_allclose(depths, np.linspace(c, min(0.08, 0.07) - c, 3), atol=1e-15)
        t = G.grasp_cylinder_side(0.02, 0.12, k=1, clearance=c)[0]
        self.assertAlmostEqual(t.Bw[2, 1], 0.12 / 2 - c, places=15)


class TestMarginBoundaries(unittest.TestCase):
    def test_slide_band_collapses_at_twice_the_margin(self):
        # A slide dimension of exactly 2*margin is one centred pose (zero-width band).
        thin = 4 * length_atol(0.05)  # scale is the largest dimension
        templates = G.grasp_box_top(0.05, thin, 0.05, k=1, clearance=0.0, preshape=0.07)
        spans = {t.provenance.finger_orientation for t in templates}
        self.assertIn("x", spans)  # span-x slides in y: zero-width but feasible
        _assert_all_sound(self, Box(0.05, thin, 0.05), templates, G, 0.0)


class TestEdgeMarginProperty(unittest.TestCase):
    @settings(max_examples=100, deadline=None)
    @given(
        bx=st.floats(0.01, 0.15),
        by=st.floats(0.01, 0.15),
        bz=st.floats(0.01, 0.15),
        radius=st.floats(0.005, 0.04),
        height=st.floats(0.01, 0.3),
        c=st.floats(0.0, 1e-6),
        k=st.integers(1, 4),
    )
    def test_near_zero_clearance_is_sound(self, bx, by, bz, radius, height, c, k):
        g = ParallelJawGripper(finger_length=0.08, max_aperture=0.5)
        _assert_all_sound(self, Box(bx, by, bz), g.grasp_box(bx, by, bz, k=k, clearance=c, preshape=0.3), g, c)
        _assert_all_sound(
            self, Cylinder(radius, height), g.grasp_cylinder(radius, height, k=k, clearance=c, preshape=0.3), g, c
        )


if __name__ == "__main__":
    unittest.main()
