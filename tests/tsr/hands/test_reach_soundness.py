#!/usr/bin/env python
# SPDX-License-Identifier: BSD-2-Clause
# Authors: Siddhartha Srinivasa and contributors to TSR

"""Reach soundness for cylinder-side and sphere grasps (issue #69).

When the fingers are too short to grasp past the object with the requested
clearance (``finger_length < radius + clearance``), the factory must return an
empty feasible set rather than a template with the palm inside the primitive.
These tests certify every emitted pose against the **independent** analytic oracle
(#67), so soundness is checked from the concrete geometry, not the generator's own
formulas.
"""

import unittest

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from tsr.hands import FrankaHand, ParallelJawGripper, Robotiq2F85, Robotiq2F140

from ._grasp_oracle import Cylinder, Sphere, certify

positive = st.floats(min_value=0.01, max_value=0.2, allow_nan=False, width=64)


def _bw_samples(t):
    """Midpoint plus both extrema of every free Bw coordinate (clause 7 sampling)."""
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
        prov = t.provenance
        tsr = t.instantiate(np.eye(4))  # object-frame end-effector poses
        for xyz in _bw_samples(t):
            pose = tsr.to_transform(xyz)
            w = certify(
                prim,
                pose,
                finger_length=gripper.finger_length,
                max_aperture=gripper.max_aperture,
                preshape=float(t.preshape[0]),
                clearance=clearance,
                mode=prov.mode,
                approach=prov.approach,
                finger_orientation=prov.finger_orientation,
            )
            testcase.assertTrue(w.ok, (prov.mode, w.failed, "palm_clearance", w.palm_clearance))


class TestReachReproduction(unittest.TestCase):
    """The #69 reproduction returns [] instead of a palm-inside template."""

    def test_reproduction_returns_empty(self):
        g = ParallelJawGripper(finger_length=0.05, max_aperture=0.30)
        self.assertEqual(g.grasp_sphere(0.10), [])
        self.assertEqual(g.grasp_cylinder_side(0.10, 0.30), [])

    def test_reach_empty_logs_finger_too_short(self):
        import logging

        g = ParallelJawGripper(finger_length=0.05, max_aperture=0.30)
        with self.assertLogs("tsr.hands.base", level=logging.DEBUG) as cm:
            g.grasp_sphere(0.10)
        self.assertTrue(any("finger_too_short" in m for m in cm.output))


class TestReachBoundary(unittest.TestCase):
    """Exact reach boundary: finger_length == radius + clearance (#69)."""

    def _counts(self, fl, r, c, h=0.3):
        g = ParallelJawGripper(finger_length=fl, max_aperture=0.30)
        return len(g.grasp_sphere(r, clearance=c)), len(g.grasp_cylinder_side(r, h, clearance=c))

    def test_boundary_and_neighbours(self):
        r, c = 0.10, 0.02  # boundary at fl = r + c = 0.12 (exactly representable)
        boundary = r + c
        # At the boundary the reach band degenerates to one point -> a single depth.
        ns, nc = self._counts(boundary, r, c)
        self.assertEqual(ns, 1)
        self.assertEqual(nc, 2)  # cylinder side: 1 depth x 2 roll variants
        # Just below the boundary the band is empty -> [].
        below = np.nextafter(boundary, -np.inf)
        self.assertEqual(self._counts(below, r, c), (0, 0))
        # Just above the boundary the band is a positive width -> templates.
        above = np.nextafter(boundary, np.inf)
        ns_a, nc_a = self._counts(above, r, c)
        self.assertGreater(ns_a, 0)
        self.assertGreater(nc_a, 0)

    def test_boundary_template_is_sound(self):
        r, c = 0.10, 0.02
        g = ParallelJawGripper(finger_length=r + c, max_aperture=0.30)
        _assert_all_sound(self, Sphere(r), g.grasp_sphere(r, clearance=c), g, c)
        _assert_all_sound(self, Cylinder(r, 0.3), g.grasp_cylinder_side(r, 0.3, clearance=c), g, c)


class TestReachSoundnessProperty(unittest.TestCase):
    """Every emitted cylinder-side / sphere pose has a valid oracle witness (#69)."""

    @settings(max_examples=80, deadline=None)
    @given(
        fl=positive, ma=positive, r=positive, h=positive, c=st.floats(0.0, 0.05, allow_nan=False), k=st.integers(1, 4)
    )
    def test_emitted_poses_are_oracle_sound(self, fl, ma, r, h, c, k):
        g = ParallelJawGripper(finger_length=fl, max_aperture=ma)
        _assert_all_sound(self, Sphere(r), g.grasp_sphere(r, clearance=c, k=k), g, c)
        _assert_all_sound(self, Cylinder(r, h), g.grasp_cylinder_side(r, h, clearance=c, k=k), g, c)

    @settings(max_examples=40, deadline=None)
    @given(r=st.floats(0.01, 0.06), seed=st.integers(0, 2**32 - 1))
    def test_sphere_grasp_is_rotationally_equivariant(self, r, seed):
        # A sphere is rotationally symmetric, so rotating a sound grasp pose about the
        # centre keeps it sound -- the oracle witness is preserved under SO(3).
        rng = np.random.default_rng(seed)
        g = ParallelJawGripper(finger_length=0.08, max_aperture=0.30)
        templates = g.grasp_sphere(r, clearance=0.006)
        rot = np.eye(4)
        rot[:3, :3] = np.linalg.qr(rng.normal(size=(3, 3)))[0]
        rot[:3, :3] *= np.sign(np.linalg.det(rot[:3, :3]))  # ensure a proper rotation
        for t in templates:
            tsr = t.instantiate(np.eye(4))
            pose = rot @ tsr.to_transform(np.zeros(6))
            w = certify(
                Sphere(r),
                pose,
                finger_length=0.08,
                max_aperture=0.30,
                preshape=float(t.preshape[0]),
                clearance=0.006,
                mode="surface",
            )
            self.assertTrue(w.ok, w.failed)


class TestNamedGripperReach(unittest.TestCase):
    """Named grippers stay sound (or empty) around their aperture/reach boundaries (#69)."""

    def test_named_grippers_near_reach_boundary(self):
        for cls in (Robotiq2F85, Robotiq2F140, FrankaHand):
            g = cls()
            fl = g.finger_length
            for r in (0.4 * fl, fl, 1.5 * fl):  # inside, at, and beyond finger reach
                c = 0.1 * min(fl, r)
                _assert_all_sound(self, Sphere(r), g.grasp_sphere(r, clearance=c), g, c)
                _assert_all_sound(
                    self, Cylinder(r, max(4 * r, 0.1)), g.grasp_cylinder_side(r, max(4 * r, 0.1), clearance=c), g, c
                )
                # A radius well beyond finger reach must yield no palm-inside template.
                if r > fl:
                    self.assertEqual(g.grasp_sphere(r, clearance=c), [])


if __name__ == "__main__":
    unittest.main()
