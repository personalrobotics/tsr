#!/usr/bin/env python
# SPDX-License-Identifier: BSD-2-Clause
# Authors: Siddhartha Srinivasa and contributors to TSR

"""Unit tests for the independent analytic grasp oracle (issue #67).

Every pose here is **hand-derived** with ``pose_from`` -- no template is produced
by the grasp factory under test, so the oracle is validated standalone (the
oracle-over-generator wiring lives in the #68-#73 generator-repair work). Positive
fixtures are canonical parallel-jaw pre-grasps; negative controls flip the
approach axis, move the face origin, oversize the span, push the palm inside, add
excess clearance, and undersize the aperture -- and we assert the *clause* each one
violates, not just a Boolean.
"""

import unittest

import numpy as np

from ._grasp_oracle import (
    Box,
    Cylinder,
    Sphere,
    Torus,
    certify,
    length_atol,
    outward_normal,
    pose_from,
)

L, A = 0.08, 0.14  # idealized finger length / max aperture used across fixtures


def _clauses(witness):
    return {clause for clause, _ in witness.failed}


class TestPrimitiveSDF(unittest.TestCase):
    """The independent SDFs are correct in sign and zero on the surface."""

    def test_sphere_sdf(self):
        s = Sphere(0.03)
        self.assertAlmostEqual(s.sdf(np.array([0.03, 0, 0])), 0.0, places=12)
        self.assertLess(s.sdf(np.array([0.0, 0, 0])), 0)  # center inside
        self.assertGreater(s.sdf(np.array([0.05, 0, 0])), 0)  # outside

    def test_cylinder_sdf(self):
        c = Cylinder(0.03, 0.12)  # z in [0, 0.12]
        self.assertAlmostEqual(c.sdf(np.array([0.03, 0, 0.06])), 0.0, places=12)  # lateral
        self.assertAlmostEqual(c.sdf(np.array([0.0, 0, 0.12])), 0.0, places=12)  # top cap
        self.assertLess(c.sdf(np.array([0.0, 0, 0.06])), 0)  # axis inside
        self.assertGreater(c.sdf(np.array([0.0, 0, 0.20])), 0)  # above top

    def test_box_sdf(self):
        b = Box(0.05, 0.06, 0.07)  # x,y centered; z in [0,0.07]
        self.assertAlmostEqual(b.sdf(np.array([0.025, 0, 0.035])), 0.0, places=12)  # +x face
        self.assertAlmostEqual(b.sdf(np.array([0.0, 0.03, 0.035])), 0.0, places=12)  # +y face
        self.assertLess(b.sdf(np.array([0.0, 0, 0.035])), 0)  # center inside
        self.assertGreater(b.sdf(np.array([0.0, 0, 0.10])), 0)  # above top

    def test_torus_sdf(self):
        t = Torus(0.06, 0.02)  # tube ring radius 0.06, tube radius 0.02
        self.assertAlmostEqual(t.sdf(np.array([0.08, 0, 0])), 0.0, places=12)  # outer equator
        self.assertAlmostEqual(t.sdf(np.array([0.04, 0, 0])), 0.0, places=12)  # inner equator
        self.assertLess(t.sdf(np.array([0.06, 0, 0])), 0)  # tube center inside
        self.assertGreater(t.sdf(np.array([0.0, 0, 0])), 0)  # hole (center) outside

    def test_outward_normal_points_out(self):
        s = Sphere(0.03)
        n = outward_normal(s, np.array([0.03, 0, 0]))
        np.testing.assert_allclose(n, [1, 0, 0], atol=1e-6)


class TestOraclePositive(unittest.TestCase):
    """Canonical hand-derived grasps are certified sound for every family."""

    def test_sphere_surface(self):
        r, c = 0.03, 0.006
        pose = pose_from([0.06, 0, 0], [-1, 0, 0], [0, 1, 0])  # approach through center
        w = certify(Sphere(r), pose, finger_length=L, max_aperture=A, preshape=2 * r + c, clearance=c, mode="surface")
        self.assertTrue(w.ok, w.failed)

    def test_cylinder_side(self):
        r, h, c = 0.03, 0.12, 0.006
        pose = pose_from([0.06, 0, h / 2], [-1, 0, 0], [0, 1, 0])  # radial approach, diameter close
        w = certify(Cylinder(r, h), pose, finger_length=L, max_aperture=A, preshape=2 * r + c, clearance=c, mode="side")
        self.assertTrue(w.ok, w.failed)

    def test_cylinder_top_and_bottom(self):
        r, h, c = 0.03, 0.12, 0.006
        top = pose_from([0, 0, h + (L - 0.02)], [0, 0, -1], [0, 1, 0])
        bot = pose_from([0, 0, -(L - 0.02)], [0, 0, 1], [0, 1, 0])
        for pose, mode in ((top, "top"), (bot, "bottom")):
            w = certify(
                Cylinder(r, h), pose, finger_length=L, max_aperture=A, preshape=2 * r + c, clearance=c, mode=mode
            )
            self.assertTrue(w.ok, (mode, w.failed))

    def test_box_top_both_orientations(self):
        bx, by, bz, c = 0.05, 0.06, 0.07, 0.006
        span_y = pose_from([0, 0, bz + (L - 0.02)], [0, 0, -1], [0, 1, 0])
        span_x = pose_from([0, 0, bz + (L - 0.02)], [0, 0, -1], [1, 0, 0])
        wy = certify(Box(bx, by, bz), span_y, finger_length=L, max_aperture=A, preshape=by + c, clearance=c, mode="top")
        wx = certify(Box(bx, by, bz), span_x, finger_length=L, max_aperture=A, preshape=bx + c, clearance=c, mode="top")
        self.assertTrue(wy.ok, wy.failed)
        self.assertTrue(wx.ok, wx.failed)

    def test_box_face(self):
        bx, by, bz, c = 0.05, 0.06, 0.07, 0.006
        pose = pose_from([bx / 2 + (L - 0.02), 0, bz / 2], [-1, 0, 0], [0, 0, 1])  # +x face, span z
        w = certify(Box(bx, by, bz), pose, finger_length=L, max_aperture=A, preshape=bz + c, clearance=c, mode="face")
        self.assertTrue(w.ok, w.failed)

    def test_torus_span(self):
        R, r, c = 0.06, 0.02, 0.004
        pose = pose_from([0, 0, 0.05], [0, 0, -1], [0, 1, 0])  # descend to equator, close across outer diameter
        w = certify(
            Torus(R, r), pose, finger_length=L, max_aperture=0.30, preshape=2 * (R + r) + c, clearance=c, mode="span"
        )
        self.assertTrue(w.ok, w.failed)

    def test_torus_side(self):
        R, r, c = 0.06, 0.02, 0.004
        pose = pose_from([0.12, 0, 0], [-1, 0, 0], [0, 0, 1])  # radial approach, close along tube axis
        w = certify(Torus(R, r), pose, finger_length=L, max_aperture=A, preshape=2 * r + c, clearance=c, mode="side")
        self.assertTrue(w.ok, w.failed)


class TestOracleWitness(unittest.TestCase):
    """A canonical grasp's witness reports the correct geometry."""

    def test_sphere_witness_geometry(self):
        r, c = 0.03, 0.006
        pose = pose_from([0.06, 0, 0], [-1, 0, 0], [0, 1, 0])
        w = certify(Sphere(r), pose, finger_length=L, max_aperture=A, preshape=2 * r + c, clearance=c, mode="surface")
        atol = 10 * length_atol(r)
        self.assertAlmostEqual(w.span, 2 * r, delta=atol)  # diameter grasp
        self.assertAlmostEqual(w.palm_clearance, 0.06 - r, delta=atol)  # ro - r
        np.testing.assert_allclose(w.contact_pos, [0, r, 0], atol=atol)
        np.testing.assert_allclose(w.contact_neg, [0, -r, 0], atol=atol)
        np.testing.assert_allclose(w.normal_pos, [0, 1, 0], atol=1e-5)  # opposes +close
        np.testing.assert_allclose(w.normal_neg, [0, -1, 0], atol=1e-5)


class TestOracleNegativeControls(unittest.TestCase):
    """Sign / axis / origin / extent / clearance / aperture errors are detected,
    each pinned to the soundness clause it violates."""

    def test_undersized_aperture_fails_clause_2(self):
        r, c = 0.03, 0.006
        pose = pose_from([0.06, 0, 0], [-1, 0, 0], [0, 1, 0])
        w = certify(
            Sphere(r), pose, finger_length=L, max_aperture=A, preshape=2 * r - 0.001, clearance=c, mode="surface"
        )
        self.assertFalse(w.ok)
        self.assertIn(2, _clauses(w))

    def test_oversized_span_fails_clause_2(self):
        # A box whose closed dimension (0.20) exceeds the preshape it is grasped with.
        bx, by, bz, c = 0.20, 0.06, 0.07, 0.006
        pose = pose_from([0, 0, bz + (L - 0.02)], [0, 0, -1], [1, 0, 0])  # close along the 0.20 axis
        w = certify(Box(bx, by, bz), pose, finger_length=L, max_aperture=A, preshape=bx + c, clearance=c, mode="top")
        self.assertFalse(w.ok)
        self.assertIn(2, _clauses(w))

    def test_palm_inside_fails_clause_3(self):
        # Radius exceeds finger reach: the palm sits inside the solid (#69 family).
        pose = pose_from([0.05, 0, 0], [-1, 0, 0], [0, 1, 0])  # palm at 0.05 from center of a 0.10 sphere
        w = certify(
            Sphere(0.10), pose, finger_length=L, max_aperture=A, preshape=0.2 + 0.006, clearance=0.006, mode="surface"
        )
        self.assertFalse(w.ok)
        self.assertIn(3, _clauses(w))
        self.assertLess(w.palm_clearance, 0)

    def test_flipped_approach_fails_clause_5(self):
        r, c = 0.03, 0.006
        pose = pose_from([0.06, 0, 0], [1, 0, 0], [0, 1, 0])  # approach points AWAY from the sphere
        w = certify(Sphere(r), pose, finger_length=L, max_aperture=A, preshape=2 * r + c, clearance=c, mode="surface")
        self.assertFalse(w.ok)
        self.assertIn(5, _clauses(w))

    def test_wrong_face_origin_fails_clause_5(self):
        bx, by, bz, c = 0.05, 0.06, 0.07, 0.006
        pose = pose_from([0.5, 0, bz + 0.02], [0, 0, -1], [0, 1, 0])  # palm far off the top face
        w = certify(Box(bx, by, bz), pose, finger_length=L, max_aperture=A, preshape=by + c, clearance=c, mode="top")
        self.assertFalse(w.ok)
        self.assertIn(5, _clauses(w))

    def test_excess_clearance_unreachable_center_fails(self):
        # Palm so far from a sphere that the fingertips cannot reach the diameter
        # plane; the only reachable contacts are off-center, so normals don't oppose.
        r, c = 0.03, 0.006
        pose = pose_from([0.10, 0, 0], [-1, 0, 0], [0, 1, 0])  # center at depth 0.10 > L
        w = certify(Sphere(r), pose, finger_length=L, max_aperture=A, preshape=2 * r + c, clearance=c, mode="surface")
        self.assertFalse(w.ok)
        self.assertIn(6, _clauses(w))

    def test_tilted_closing_axis_breaks_normal_opposition_clause_6(self):
        # A cylinder side grasp whose closing axis is tilted out of the plane normal
        # to the cylinder axis: the lateral-surface normals (purely radial, no z)
        # cannot oppose a closing direction that has an axial (z) component.
        r, h, c = 0.03, 0.12, 0.006
        pose = pose_from([0.06, 0, h / 2], [-1, 0, 0], [0, 1, 1])  # close tilted toward +z
        w = certify(Cylinder(r, h), pose, finger_length=L, max_aperture=A, preshape=2 * r + c, clearance=c, mode="side")
        self.assertFalse(w.ok)
        self.assertIn(6, _clauses(w))

    def test_nonfinite_pose_fails_clause_1(self):
        pose = pose_from([0.06, 0, 0], [-1, 0, 0], [0, 1, 0])
        pose[0, 3] = np.nan
        w = certify(
            Sphere(0.03), pose, finger_length=L, max_aperture=A, preshape=0.066, clearance=0.006, mode="surface"
        )
        self.assertFalse(w.ok)
        self.assertIn(1, _clauses(w))


if __name__ == "__main__":
    unittest.main()
