#!/usr/bin/env python
# SPDX-License-Identifier: BSD-2-Clause
# Authors: Siddhartha Srinivasa and contributors to TSR

"""Unit tests for the independent analytic grasp oracle (issue #67, #93/#94/#95).

Every pose is **hand-derived** with ``pose_from`` -- no template is produced by the
grasp factory under test, so the oracle is validated standalone. The suite covers:
exact/scale-robust contact geometry (#93), mode-specific clearance and derived
semantics (#94), and full model/pose validation (#95).
"""

import math
import unittest

import numpy as np

from ._grasp_oracle import (
    ANGLE_ATOL,
    Box,
    Cylinder,
    Sphere,
    Torus,
    certify,
    length_atol,
    pose_from,
)

L, A = 0.08, 0.14  # idealized finger length / max aperture used across fixtures


def _clauses(w):
    return {clause for clause, _ in w.failed}


def _certify(prim, pose, *, finger_length=L, max_aperture=A, preshape, clearance, mode, **kw):
    return certify(
        prim,
        pose,
        finger_length=finger_length,
        max_aperture=max_aperture,
        preshape=preshape,
        clearance=clearance,
        mode=mode,
        **kw,
    )


class TestPrimitiveSDF(unittest.TestCase):
    """The independent SDFs (used as verifiers) are correct in sign and on-surface."""

    def test_sphere(self):
        self.assertAlmostEqual(Sphere(0.03).sdf(np.array([0.03, 0, 0])), 0.0, places=12)
        self.assertLess(Sphere(0.03).sdf(np.zeros(3)), 0)

    def test_cylinder(self):
        c = Cylinder(0.03, 0.12)
        self.assertAlmostEqual(c.sdf(np.array([0.03, 0, 0.06])), 0.0, places=12)
        self.assertAlmostEqual(c.sdf(np.array([0.0, 0, 0.12])), 0.0, places=12)

    def test_box(self):
        b = Box(0.05, 0.06, 0.07)
        self.assertAlmostEqual(b.sdf(np.array([0.0, 0.03, 0.035])), 0.0, places=12)
        self.assertGreater(b.sdf(np.array([0.0, 0, 0.10])), 0)

    def test_torus(self):
        t = Torus(0.06, 0.02)
        self.assertAlmostEqual(t.sdf(np.array([0.08, 0, 0])), 0.0, places=12)
        self.assertGreater(t.sdf(np.zeros(3)), 0)  # hole is outside


class TestModelValidation(unittest.TestCase):
    """Malformed MODELS raise ValueError before any geometry (clause-1 policy, #95)."""

    GOOD = dict(finger_length=L, max_aperture=A, preshape=0.066, clearance=0.006, mode="surface")

    def _pose(self):
        return pose_from([0.06, 0, 0], [-1, 0, 0], [0, 1, 0])

    def test_bad_primitive_dimensions_raise(self):
        for bad in (0.0, -0.03, np.nan, np.inf, -np.inf):
            with self.assertRaises(ValueError):
                certify(Sphere(bad), self._pose(), **self.GOOD)
        with self.assertRaises(ValueError):
            certify(
                Box(0.05, 0.0, 0.07),
                self._pose(),
                finger_length=L,
                max_aperture=A,
                preshape=0.06,
                clearance=0.006,
                mode="top",
            )
        with self.assertRaises(ValueError):
            certify(
                Torus(0.06, 0.06),
                self._pose(),
                finger_length=L,
                max_aperture=A,
                preshape=0.06,
                clearance=0.006,
                mode="side",
            )  # major !> minor

    def test_bad_gripper_and_clearance_raise(self):
        for key, bad in [
            ("finger_length", 0.0),
            ("finger_length", -1.0),
            ("finger_length", np.inf),
            ("max_aperture", 0.0),
            ("max_aperture", np.nan),
            ("preshape", 0.0),
            ("preshape", -0.1),
            ("preshape", np.inf),
            ("clearance", -0.001),
            ("clearance", np.nan),
            ("clearance", np.inf),
        ]:
            kw = dict(self.GOOD)
            kw[key] = bad
            with self.assertRaises(ValueError, msg=f"{key}={bad}"):
                certify(Sphere(0.03), self._pose(), **kw)

    def test_clearance_zero_is_allowed(self):
        kw = dict(self.GOOD)
        kw["clearance"] = 0.0
        w = certify(Sphere(0.03), self._pose(), **kw)
        self.assertTrue(w.ok, w.failed)

    def test_unsupported_mode_raises(self):
        with self.assertRaises(ValueError):
            certify(
                Sphere(0.03), self._pose(), finger_length=L, max_aperture=A, preshape=0.066, clearance=0.006, mode="top"
            )
        with self.assertRaises(ValueError):
            certify(
                Cylinder(0.03, 0.12),
                self._pose(),
                finger_length=L,
                max_aperture=A,
                preshape=0.066,
                clearance=0.006,
                mode="surface",
            )


class TestPoseValidation(unittest.TestCase):
    """Malformed POSES return a structured clause-1 failure, not an exception (#95)."""

    def _cert(self, pose):
        return _certify(Sphere(0.03), pose, preshape=0.066, clearance=0.006, mode="surface")

    def test_reflection_det_minus_one_fails_clause_1(self):
        pose = pose_from([0.06, 0, 0], [-1, 0, 0], [0, 1, 0])
        pose[:3, 0] *= -1  # det(R) = -1
        w = self._cert(pose)
        self.assertFalse(w.ok)
        self.assertEqual(_clauses(w), {1})

    def test_nonfinite_pose_fails_clause_1(self):
        pose = pose_from([0.06, 0, 0], [-1, 0, 0], [0, 1, 0])
        pose[0, 3] = np.inf
        self.assertIn(1, _clauses(self._cert(pose)))

    def test_bad_shape_fails_clause_1(self):
        self.assertIn(1, _clauses(self._cert(np.eye(3))))

    def test_bad_homogeneous_row_fails_clause_1(self):
        pose = pose_from([0.06, 0, 0], [-1, 0, 0], [0, 1, 0])
        pose[3, 0] = 0.5
        self.assertIn(1, _clauses(self._cert(pose)))

    def test_nonorthonormal_fails_clause_1(self):
        pose = pose_from([0.06, 0, 0], [-1, 0, 0], [0, 1, 0])
        pose[:3, :3] *= 1.5  # scaled rotation
        self.assertIn(1, _clauses(self._cert(pose)))


class TestOraclePositive(unittest.TestCase):
    """Canonical hand-derived grasps are certified sound for every family/mode."""

    def test_all_families(self):
        cases = [
            (Sphere(0.03), pose_from([0.06, 0, 0], [-1, 0, 0], [0, 1, 0]), "surface", 0.066, A),
            (Cylinder(0.03, 0.12), pose_from([0.06, 0, 0.06], [-1, 0, 0], [0, 1, 0]), "side", 0.066, A),
            (Cylinder(0.03, 0.12), pose_from([0, 0, 0.18], [0, 0, -1], [0, 1, 0]), "top", 0.066, A),
            (Cylinder(0.03, 0.12), pose_from([0, 0, -0.06], [0, 0, 1], [0, 1, 0]), "bottom", 0.066, A),
            (Box(0.05, 0.06, 0.07), pose_from([0, 0, 0.13], [0, 0, -1], [0, 1, 0]), "top", 0.066, A),
            (Box(0.05, 0.06, 0.07), pose_from([0, 0, -0.06], [0, 0, 1], [1, 0, 0]), "bottom", 0.056, A),
            (Box(0.05, 0.06, 0.07), pose_from([0.085, 0, 0.035], [-1, 0, 0], [0, 0, 1]), "face", 0.076, A),
            (Torus(0.06, 0.02), pose_from([0, 0, 0.05], [0, 0, -1], [0, 1, 0]), "span", 0.164, 0.30),
            (Torus(0.06, 0.02), pose_from([0.12, 0, 0], [-1, 0, 0], [0, 0, 1]), "side", 0.044, A),
        ]
        for prim, pose, mode, preshape, ap in cases:
            w = _certify(prim, pose, max_aperture=ap, preshape=preshape, clearance=0.006, mode=mode)
            self.assertTrue(w.ok, (type(prim).__name__, mode, w.failed))


class TestWitnessGeometry(unittest.TestCase):
    """The witness reports correct pose-derived geometry."""

    def test_sphere_diameter_and_normals(self):
        r, c = 0.03, 0.006
        w = _certify(
            Sphere(r), pose_from([0.06, 0, 0], [-1, 0, 0], [0, 1, 0]), preshape=2 * r + c, clearance=c, mode="surface"
        )
        atol = 10 * length_atol(r)
        self.assertAlmostEqual(w.span, 2 * r, delta=atol)
        np.testing.assert_allclose(w.contact_pos, [0, r, 0], atol=atol)
        np.testing.assert_allclose(w.normal_pos, [0, 1, 0], atol=1e-6)
        # #93: contact-normal error within the 1e-6 rad contract even off-grid.
        angle = math.acos(np.clip(w.normal_pos @ np.array([0, 1, 0]), -1, 1))
        self.assertLess(angle, ANGLE_ATOL)

    def test_realized_depth_matches_provenance_frame(self):
        # Sphere side generator: palm at ro = r + L - d, so realized insertion = d.
        r, L0, c = 0.03, 0.08, 0.006
        for d in (0.03, 0.045, 0.06):
            ro = r + L0 - d
            w = _certify(
                Sphere(r),
                pose_from([ro, 0, 0], [-1, 0, 0], [0, 1, 0]),
                finger_length=L0,
                preshape=2 * r + c,
                clearance=c,
                mode="surface",
            )
            self.assertAlmostEqual(w.realized_depth, d, delta=1e-6, msg=f"d={d}")

    def test_torus_side_minor_angle_is_pose_derived(self):
        # Closing along +z at the tube contacts the tube top/bottom -> minor angle pi/2.
        w = _certify(
            Torus(0.06, 0.02),
            pose_from([0.12, 0, 0], [-1, 0, 0], [0, 0, 1]),
            preshape=0.044,
            clearance=0.006,
            mode="side",
        )
        self.assertIsNotNone(w.minor_angle)
        self.assertAlmostEqual(abs(w.minor_angle), math.pi / 2, delta=1e-6)


class TestModeSemantics(unittest.TestCase):
    """mode participates in dispatch; clearance bands and mislabels are enforced (#94)."""

    def test_mislabeled_mode_fails_clause_8(self):
        # A radial (side) pose declared as a top grasp.
        pose = pose_from([0.06, 0, 0.06], [-1, 0, 0], [0, 1, 0])
        w = _certify(Cylinder(0.03, 0.12), pose, preshape=0.066, clearance=0.006, mode="top")
        self.assertFalse(w.ok)
        self.assertIn(8, _clauses(w))

    def test_cylinder_side_end_clearance_band(self):
        cyl, c = Cylinder(0.03, 0.12), 0.006
        atol = length_atol(cyl.scale)

        def side(z0):
            return _certify(
                cyl, pose_from([0.06, 0, z0], [-1, 0, 0], [0, 1, 0]), preshape=0.066, clearance=c, mode="side"
            )

        self.assertFalse(side(1e-5).ok)  # 10 microns from the end (#94 reproduction)
        self.assertIn(4, _clauses(side(1e-5)))
        self.assertFalse(side(np.nextafter(c, -np.inf) - atol).ok)  # just inside the band
        self.assertTrue(side(c + atol).ok)  # just outside the band, otherwise sound
        self.assertTrue(side(0.06).ok)  # comfortably clear

    def test_box_top_edge_clearance_band(self):
        box, c = Box(0.05, 0.06, 0.07), 0.006

        # Slide axis for a top/span-y grasp is x (half-extent 0.025); edge band is |x| <= 0.025 - c.
        def top(x0):
            return _certify(
                box, pose_from([x0, 0, 0.13], [0, 0, -1], [0, 1, 0]), preshape=0.066, clearance=c, mode="top"
            )

        self.assertFalse(top(0.025 - 1e-5).ok)  # 10 microns from the lateral edge (#94)
        self.assertIn(4, _clauses(top(0.025 - 1e-5)))
        self.assertTrue(top(0.0).ok)  # centered

    def test_declared_selectors_checked(self):
        # A valid box face grasp with a wrong declared finger_orientation fails clause 8.
        pose = pose_from([0.085, 0, 0.035], [-1, 0, 0], [0, 0, 1])  # closes along z
        ok = _certify(Box(0.05, 0.06, 0.07), pose, preshape=0.076, clearance=0.006, mode="face", finger_orientation="z")
        self.assertTrue(ok.ok, ok.failed)
        bad = _certify(
            Box(0.05, 0.06, 0.07), pose, preshape=0.076, clearance=0.006, mode="face", finger_orientation="y"
        )
        self.assertFalse(bad.ok)
        self.assertIn(8, _clauses(bad))


class TestScaleRobustness(unittest.TestCase):
    """Exact geometry: results do not depend on any sampling grid (#93)."""

    def test_off_grid_sphere_certifies(self):
        # The #93 reproduction: center depth 0.0601 is not commensurate with L/128.
        w = _certify(
            Sphere(0.03),
            pose_from([0.0601, 0, 0], [-1, 0, 0], [0, 1, 0]),
            preshape=0.066,
            clearance=0.006,
            mode="surface",
        )
        self.assertTrue(w.ok, w.failed)

    def test_ordinary_library_sphere_depths_all_certify(self):
        # r=0.04, L=0.055: all three generated depths must certify (#93 acceptance).
        r, L0 = 0.04, 0.055
        c = 0.1 * min(L0, 2 * r)
        for d in np.linspace(r, min(L0, 2 * r) - c, 3):
            ro = r + L0 - d
            w = _certify(
                Sphere(r),
                pose_from([ro, 0, 0], [-1, 0, 0], [0, 1, 0]),
                finger_length=L0,
                preshape=2 * r + c,
                clearance=c,
                mode="surface",
            )
            self.assertTrue(w.ok, (float(d), w.failed))

    def test_scale_equivariance(self):
        for k in (1e-3, 1.0, 1e3):
            w = _certify(
                Sphere(0.03 * k),
                pose_from([0.06 * k, 0, 0], [-1, 0, 0], [0, 1, 0]),
                finger_length=0.08 * k,
                max_aperture=0.14 * k,
                preshape=0.066 * k,
                clearance=0.006 * k,
                mode="surface",
            )
            self.assertTrue(w.ok, (k, w.failed))
            self.assertAlmostEqual(w.span / k, 0.06, delta=1e-6)

    def test_thin_torus_side(self):
        w = _certify(
            Torus(0.06, 0.003),
            pose_from([0.06 + 0.003 + 0.05, 0, 0], [-1, 0, 0], [0, 0, 1]),
            preshape=2 * 0.003 + 0.001,
            clearance=0.001,
            mode="side",
        )
        self.assertTrue(w.ok, w.failed)


class TestNegativeControls(unittest.TestCase):
    """Sign / axis / origin / extent / clearance / aperture errors are detected."""

    def test_undersized_aperture_clause_2(self):
        w = _certify(
            Sphere(0.03),
            pose_from([0.06, 0, 0], [-1, 0, 0], [0, 1, 0]),
            preshape=0.059,
            clearance=0.006,
            mode="surface",
        )
        self.assertIn(2, _clauses(w))

    def test_oversized_span_clause_2(self):
        w = _certify(
            Box(0.20, 0.06, 0.07),
            pose_from([0, 0, 0.13], [0, 0, -1], [1, 0, 0]),
            preshape=0.206,
            clearance=0.006,
            mode="top",
        )
        self.assertIn(2, _clauses(w))

    def test_palm_inside_clause_3(self):
        w = _certify(
            Sphere(0.10),
            pose_from([0.05, 0, 0], [-1, 0, 0], [0, 1, 0]),
            preshape=0.206,
            clearance=0.006,
            mode="surface",
        )
        self.assertIn(3, _clauses(w))
        self.assertLess(w.palm_clearance, 0)

    def test_flipped_approach_clause_5(self):
        w = _certify(
            Sphere(0.03), pose_from([0.06, 0, 0], [1, 0, 0], [0, 1, 0]), preshape=0.066, clearance=0.006, mode="surface"
        )
        self.assertIn(5, _clauses(w))

    def test_wrong_face_origin_clause_5(self):
        w = _certify(
            Box(0.05, 0.06, 0.07),
            pose_from([0.5, 0, 0.13], [0, 0, -1], [0, 1, 0]),
            preshape=0.066,
            clearance=0.006,
            mode="top",
        )
        self.assertFalse(w.ok)

    def test_tilted_closing_axis_clause_6(self):
        w = _certify(
            Cylinder(0.03, 0.12),
            pose_from([0.06, 0, 0.06], [-1, 0, 0], [0, 1, 1]),
            preshape=0.066,
            clearance=0.006,
            mode="side",
        )
        self.assertIn(6, _clauses(w))


if __name__ == "__main__":
    unittest.main()
