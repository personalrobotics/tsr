#!/usr/bin/env python
# SPDX-License-Identifier: BSD-2-Clause
# Authors: Siddhartha Srinivasa and contributors to TSR

"""Cylinder cap insertion depth is bounded by the height (issue #122).

The cap band used to be ``[c, finger_length - c]``, ignoring the cylinder, so on a
short cylinder the deeper templates pushed the fingertips past the opposite cap
(#67 oracle clause 4). The band is now ``[c, min(finger_length, height) - c]``,
matching the box approach bands. Every emitted pose is certified against the
independent analytic oracle.
"""

import unittest

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from tsr.hands import ParallelJawGripper

from ._grasp_oracle import Cylinder, certify


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


def _assert_all_sound(testcase, radius, height, templates, gripper, clearance):
    for t in templates:
        p = t.provenance
        tsr = t.instantiate(np.eye(4))
        for xi in _bw_samples(t):
            w = certify(
                Cylinder(radius, height),
                tsr.to_transform(xi),
                finger_length=gripper.finger_length,
                max_aperture=gripper.max_aperture,
                preshape=float(t.preshape[0]),
                clearance=clearance,
                mode=p.mode,
                approach=p.approach,
                finger_orientation=p.finger_orientation,
            )
            testcase.assertTrue(w.ok, (p.mode, p.depth, w.failed))


class TestShortCylinderReproduction(unittest.TestCase):
    """The #122 reproduction: 80 mm fingers, a 30 mm cylinder, 6 mm clearance."""

    G = ParallelJawGripper(finger_length=0.08, max_aperture=0.2)

    def test_depths_stay_inside_the_height(self):
        for f in (self.G.grasp_cylinder_top, self.G.grasp_cylinder_bottom):
            templates = f(0.02, 0.03, k=3, clearance=0.006)
            self.assertTrue(templates)
            for t in templates:
                self.assertLessEqual(t.provenance.depth, 0.03 - 0.006 + 1e-12, f.__name__)
            _assert_all_sound(self, 0.02, 0.03, templates, self.G, 0.006)

    def test_combined_entry_point_is_sound_too(self):
        # One explicit clearance, since each part resolves its own default.
        templates = self.G.grasp_cylinder(0.02, 0.03, clearance=0.006)
        self.assertTrue(templates)
        self.assertEqual({t.provenance.mode for t in templates}, {"side", "top", "bottom"})
        _assert_all_sound(self, 0.02, 0.03, templates, self.G, 0.006)

    def test_tall_cylinder_band_is_unchanged(self):
        # height > finger_length: the fingers remain the binding limit.
        templates = self.G.grasp_cylinder_top(0.02, 0.5, k=3, clearance=0.006)
        depths = [t.provenance.depth for t in templates]
        np.testing.assert_allclose(depths, np.linspace(0.006, 0.08 - 0.006, 3), atol=1e-12)


class TestCapBandBoundary(unittest.TestCase):
    G = ParallelJawGripper(finger_length=0.08, max_aperture=0.2)

    def _counts(self, height, clearance):
        return (
            len(self.G.grasp_cylinder_top(0.02, height, k=5, clearance=clearance)),
            len(self.G.grasp_cylinder_bottom(0.02, height, k=5, clearance=clearance)),
        )

    def test_height_equal_twice_clearance_emits_one_depth(self):
        c = 0.015625
        self.assertEqual(self._counts(2 * c, c), (1, 1))

    def test_below_boundary_is_empty_with_reason(self):
        c = 0.015625
        h = float(np.nextafter(2 * c, 0.0))
        self.assertEqual(self._counts(h, c), (0, 0))
        with self.assertLogs("tsr.hands.base", level="DEBUG") as cm:
            self.G.grasp_cylinder_top(0.02, h, k=5, clearance=c)
        self.assertTrue(any("insufficient_clearance_band" in m for m in cm.output))

    def test_above_boundary_emits_sound_templates(self):
        c = 0.015625
        h = float(np.nextafter(2 * c, 1.0))
        top, bottom = self._counts(h, c)
        self.assertGreater(top, 0)
        self.assertGreater(bottom, 0)
        _assert_all_sound(self, 0.02, h, self.G.grasp_cylinder_top(0.02, h, k=5, clearance=c), self.G, c)


class TestCapSoundnessProperty(unittest.TestCase):
    """Cap templates are oracle-sound for heights below and above finger reach."""

    @settings(max_examples=100, deadline=None)
    @given(
        fl=st.floats(0.02, 0.2),
        radius=st.floats(0.005, 0.05),
        height=st.floats(0.005, 0.4),
        c=st.floats(0.001, 0.02),
        k=st.integers(1, 5),
    )
    def test_cap_templates_are_oracle_sound(self, fl, radius, height, c, k):
        g = ParallelJawGripper(finger_length=fl, max_aperture=0.3)
        for f in (g.grasp_cylinder_top, g.grasp_cylinder_bottom):
            _assert_all_sound(self, radius, height, f(radius, height, k=k, clearance=c), g, c)


if __name__ == "__main__":
    unittest.main()
