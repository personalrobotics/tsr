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

from ._grasp_matrix import GraspCase, GripperSpec, soundness_failures
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


def _lattice_step(span, margin, direction):
    """Neighbour of ``margin`` on the lattice of REALIZED margins (spacing ulp(span)).

    A default preshape is ``span + c``, so realized margins are quantized by
    ``ulp(span)`` -- coarser than ``ulp(c)`` by many orders of magnitude. Stepping the
    clearance itself by one ulp is therefore invisible; the meaningful neighbour is one
    step of this lattice.
    """
    return math.nextafter(span + margin, direction) - span


class TestDefaultPreshapePath(unittest.TestCase):
    """A public clearance request is met by the nearest representable geometry (#131)."""

    def _sphere(self, scale):
        r = 0.002 * scale
        return ParallelJawGripper(finger_length=2.0 * scale, max_aperture=3.0 * scale), r, 4 * length_atol(r)

    @given(scale=st.sampled_from(SCALES))
    @settings(max_examples=3, deadline=None)
    def test_floor_and_above_emit_below_is_rejected(self, scale):
        g, r, floor = self._sphere(scale)
        span = 2 * r
        self.assertEqual(g.grasp_sphere(r, clearance=_lattice_step(span, floor, 0.0), k=1), [])
        self.assertTrue(g.grasp_sphere(r, clearance=floor, k=1))
        self.assertTrue(g.grasp_sphere(r, clearance=_lattice_step(span, floor, math.inf), k=1))

    @given(scale=st.sampled_from(SCALES))
    @settings(max_examples=3, deadline=None)
    def test_templates_at_the_floor_certify_at_rotated_samples(self, scale):
        g, r, floor = self._sphere(scale)
        templates = g.grasp_sphere(r, clearance=floor, k=1)
        self.assertTrue(templates)
        self.assertEqual(_pose_failures(g, r, floor, templates[0], n=200), 0)

    def test_requested_clearance_is_never_weakened(self):
        # The realized margin is at least what was asked for, never a hair less.
        for scale in SCALES:
            g, r, floor = self._sphere(scale)
            for c in (floor, 3 * floor, 0.1 * r):
                t = g.grasp_sphere(r, clearance=c, k=1)[0]
                self.assertGreaterEqual(float(t.preshape[0]) - 2 * r, c, (scale, c))

    def test_explicit_preshape_is_not_altered(self):
        # Only DEFAULTS are constructed conservatively; explicit input is untouched.
        g, r, floor = self._sphere(1.0)
        explicit = 2 * r + 5 * floor
        self.assertEqual(float(g.grasp_sphere(r, preshape=explicit, clearance=floor, k=1)[0].preshape[0]), explicit)

    def test_aperture_still_binds_after_rounding_up(self):
        # If no representable preshape realizing the request fits the hardware, [] is
        # correct -- the aperture check applies to the constructed preshape.
        g, r, floor = self._sphere(1.0)
        span = 2 * r
        tight = ParallelJawGripper(finger_length=g.finger_length, max_aperture=_lattice_step(span, floor, 0.0) + span)
        self.assertEqual(tight.grasp_sphere(r, clearance=floor, k=1), [])
        with self.assertLogs("tsr.hands.base", level="DEBUG") as cm:
            tight.grasp_sphere(r, clearance=floor, k=1)
        self.assertTrue(any("exceeds_aperture" in m for m in cm.output))


# (factory, dims, object span, characteristic scale, family) -- the span and scale each
# factory is contracted to use, made explicit so a wrong one is caught (#133).
FACTORY_BOUNDARY_CASES = (
    ("grasp_sphere", {"object_radius": 0.02}, 0.04, 0.02, lambda p: True),
    ("grasp_cylinder_side", {"cylinder_radius": 0.02, "cylinder_height": 0.12}, 0.04, 0.12, lambda p: True),
    ("grasp_cylinder_top", {"cylinder_radius": 0.02, "cylinder_height": 0.12}, 0.04, 0.12, lambda p: True),
    ("grasp_cylinder_bottom", {"cylinder_radius": 0.02, "cylinder_height": 0.12}, 0.04, 0.12, lambda p: True),
    ("grasp_cylinder", {"cylinder_radius": 0.02, "cylinder_height": 0.12}, 0.04, 0.12, lambda p: True),
    ("grasp_box_top", {"box_x": 0.05, "box_y": 0.05, "box_z": 0.05}, 0.05, 0.05, lambda p: True),
    ("grasp_box_bottom", {"box_x": 0.05, "box_y": 0.05, "box_z": 0.05}, 0.05, 0.05, lambda p: True),
    ("grasp_box_face_x", {"box_x": 0.05, "box_y": 0.05, "box_z": 0.05}, 0.05, 0.05, lambda p: True),
    ("grasp_box_face_y", {"box_x": 0.05, "box_y": 0.05, "box_z": 0.05}, 0.05, 0.05, lambda p: True),
    ("grasp_box", {"box_x": 0.05, "box_y": 0.05, "box_z": 0.05}, 0.05, 0.05, lambda p: True),
    ("grasp_torus_side", {"torus_radius": 0.06, "tube_radius": 0.015}, 0.03, 0.075, lambda p: p.mode == "side"),
    ("grasp_torus_span", {"torus_radius": 0.06, "tube_radius": 0.015}, 0.15, 0.075, lambda p: p.mode == "span"),
    # Combined torus: the side and span families have different spans, so this pins the
    # side family (the span family needs preshape >= 2(R+r) and is absent here).
    ("grasp_torus", {"torus_radius": 0.06, "tube_radius": 0.015}, 0.03, 0.075, lambda p: p.mode == "side"),
)


class TestEveryFactoryHonoursTheBoundary(unittest.TestCase):
    """The exact straddle transition, per factory, with its own span and scale (#133).

    A wiring test, not a campaign: it catches a factory passing the wrong span or
    scale into the shared rule, which the general matrix can miss because an
    unexpectedly empty result satisfies soundness vacuously.
    """

    GRIPPER = GripperSpec("ParallelJawGripper", 0.08, 0.3)

    def test_factory_boundary_cases_cover_every_factory(self):
        from ._grasp_matrix import FACTORIES

        self.assertEqual({c[0] for c in FACTORY_BOUNDARY_CASES}, set(FACTORIES))

    def test_exact_boundary_transition_per_factory(self):
        for factory, dims, span, scale, family in FACTORY_BOUNDARY_CASES:
            with self.subTest(factory=factory):
                at = _floor_preshape(span, scale)
                clearance = 0.1 * min(dims.values())
                below = math.nextafter(at, 0.0)
                self.assertLess(below - span, 4 * length_atol(scale))

                def run(preshape):
                    case = GraspCase(
                        self.GRIPPER, factory, dims, {"k": 2, "clearance": clearance, "preshape": preshape}
                    )
                    return case, [t for t in case.run() if family(t.provenance)]

                _, none_expected = run(below)
                self.assertEqual(none_expected, [], f"{factory}: below the floor must be rejected")
                for preshape in (at, math.nextafter(at, math.inf)):
                    case, templates = run(preshape)
                    self.assertTrue(templates, f"{factory}: expected templates at {preshape!r}")
                    self.assertEqual(soundness_failures(case, templates), [], factory)


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
