#!/usr/bin/env python
# SPDX-License-Identifier: BSD-2-Clause
# Authors: Siddhartha Srinivasa and contributors to TSR

"""Straddle feasibility is decided on the margin, with slack (issue #129).

``_infeasibility_reason`` used to compare **sums** -- ``preshape`` against
``object_span + 2*atol`` -- so a margin far smaller than the span was lost to
rounding and a preshape below the floor was accepted. It now compares the margin
``preshape - object_span`` (exact by Sterbenz) and requires each pad to clear the
object by ``2*atol``, i.e. a margin of ``4*atol``: at exactly ``2*atol`` the oracle's
own clause-2 bound lands on the contact coordinate, so whether a pose certifies is
decided by rounding, and most sampled poses failed.
"""

import math
import unittest

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from tsr.hands import ParallelJawGripper, Robotiq2F85

from ._grasp_oracle import Sphere, certify, length_atol

SCALES = (1e-3, 1.0, 1e3)


def _pose_failures(gripper, radius, clearance, template, n=400, seed=0):
    """How many sampled poses of ``template`` the oracle rejects."""
    tsr = template.instantiate(np.eye(4))
    rng = np.random.default_rng(seed)
    lo, hi = template.Bw[:, 0], template.Bw[:, 1]
    bad = 0
    for _ in range(n):
        xi = lo + (hi - lo) * rng.random(6)
        w = certify(
            Sphere(radius),
            tsr.to_transform(xi),
            finger_length=gripper.finger_length,
            max_aperture=gripper.max_aperture,
            preshape=float(template.preshape[0]),
            clearance=clearance,
            mode="surface",
            approach="radial",
            finger_orientation="diameter",
        )
        bad += not w.ok
    return bad


def _floor_preshape(span, scale):
    """Smallest preshape whose realized margin clears the 4*atol floor."""
    floor = 4.0 * length_atol(scale)
    p = span + floor
    while p - span < floor:
        p = math.nextafter(p, math.inf)
    return p


class TestSumComparisonReproduction(unittest.TestCase):
    """The #129 reproduction: a margin below the floor was accepted via the sums."""

    G = Robotiq2F85()
    R = 0.02125

    def test_margin_below_the_floor_is_rejected(self):
        atol = length_atol(self.R)
        span = 2 * self.R
        # The decision is on the REALIZED margin ``preshape - span``, whose resolution
        # is ulp(span) -- far coarser than ulp(margin) -- so the nearest rejected
        # neighbour is one ulp below the boundary PRESHAPE, not below 4*atol.
        preshapes = (span + 2 * atol, span + 3.9 * atol, math.nextafter(_floor_preshape(span, self.R), 0.0))
        for preshape in preshapes:
            with self.subTest(margin=preshape - span):
                self.assertLess(preshape - span, 4 * atol)
                self.assertEqual(self.G.grasp_sphere(self.R, preshape=preshape, clearance=0.002), [])

    def test_rejection_reason_is_cannot_straddle(self):
        atol = length_atol(self.R)
        with self.assertLogs("tsr.hands.base", level="DEBUG") as cm:
            self.G.grasp_sphere(self.R, preshape=2 * self.R + 2 * atol, clearance=0.002)
        self.assertTrue(any("cannot_straddle" in m for m in cm.output))

    def test_accepted_templates_certify_at_rotated_samples(self):
        # The old floor left ~59% of sampled poses uncertifiable; at the new floor the
        # accepted templates certify everywhere in Bw, not only at Bw = 0.
        preshape = _floor_preshape(2 * self.R, self.R)
        templates = self.G.grasp_sphere(self.R, preshape=preshape, clearance=0.002, k=1)
        self.assertTrue(templates)
        self.assertEqual(_pose_failures(self.G, self.R, 0.002, templates[0]), 0)


class TestMarginNotSums(unittest.TestCase):
    """The decision does not depend on the span dwarfing the margin."""

    @given(scale=st.sampled_from(SCALES))
    @settings(max_examples=3, deadline=None)
    def test_floor_is_exact_at_every_scale(self, scale):
        r = 0.05 * scale
        g = ParallelJawGripper(finger_length=0.2 * scale, max_aperture=0.5 * scale)
        atol = length_atol(r)
        at = _floor_preshape(2 * r, r)
        self.assertEqual(g.grasp_sphere(r, preshape=math.nextafter(at, 0.0), clearance=0.1 * r), [])
        self.assertTrue(g.grasp_sphere(r, preshape=at, clearance=0.1 * r))
        self.assertTrue(g.grasp_sphere(r, preshape=math.nextafter(at, math.inf), clearance=0.1 * r))
        self.assertEqual(g.grasp_sphere(r, preshape=2 * r + 3 * atol, clearance=0.1 * r), [])

    def test_a_tiny_margin_on_a_huge_object_is_still_rejected(self):
        # The case the sum comparison could not see: margin ~1e-9 against a 1 m span.
        r = 0.5
        g = ParallelJawGripper(finger_length=0.6, max_aperture=2.0)
        self.assertEqual(g.grasp_sphere(r, preshape=2 * r + 1e-9, clearance=0.05), [])
        self.assertTrue(g.grasp_sphere(r, preshape=_floor_preshape(2 * r, r), clearance=0.05))


class TestOrdinaryRequestsUnchanged(unittest.TestCase):
    """Clearances and preshapes far above the floor behave exactly as before."""

    def test_default_clearance_is_unaffected(self):
        g = ParallelJawGripper(finger_length=0.08, max_aperture=0.3)
        self.assertEqual(len(g.grasp_sphere(0.03)), 3)
        self.assertEqual(len(g.grasp_cylinder(0.03, 0.12)), 12)

    @settings(max_examples=50, deadline=None)
    @given(radius=st.floats(0.005, 0.05), frac=st.floats(0.01, 0.3), scale=st.sampled_from(SCALES))
    def test_ordinary_grasps_stay_feasible_and_sound(self, radius, frac, scale):
        # The clearance is a fraction of the radius, so only the straddle rule is
        # exercised: a clearance above the radius would empty the radial depth band.
        r = radius * scale
        clearance = frac * r
        g = ParallelJawGripper(finger_length=0.2 * scale, max_aperture=0.5 * scale)
        templates = g.grasp_sphere(r, clearance=clearance, k=1)
        self.assertTrue(templates)
        self.assertEqual(_pose_failures(g, r, clearance, templates[0], n=50), 0)


if __name__ == "__main__":
    unittest.main()
