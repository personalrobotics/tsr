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

    def test_torus_side_minor_angle_is_the_approach_angle(self):
        # #99: minor_angle is the APPROACH minor angle (the approached tube surface),
        # matching the generator convention -- not the closing-contact angle. A
        # radial (equatorial) approach is alpha = 0.
        w = _certify(
            Torus(0.06, 0.02),
            pose_from([0.12, 0, 0], [-1, 0, 0], [0, 0, 1]),
            preshape=0.044,
            clearance=0.006,
            mode="side",
        )
        self.assertIsNotNone(w.minor_angle)
        self.assertAlmostEqual(w.minor_angle, 0.0, delta=1e-6)


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


class TestInsertionClearance(unittest.TestCase):
    """Clearance-aware insertion bands for caps and faces (#96)."""

    def test_cylinder_cap_insertion_band(self):
        cyl, c = Cylinder(0.03, 0.12), 0.006
        atol = length_atol(cyl.scale)

        def top(depth):  # insertion depth past the top face
            return _certify(
                cyl, pose_from([0, 0, 0.12 + L - depth], [0, 0, -1], [0, 1, 0]), preshape=0.066, clearance=c, mode="top"
            )

        self.assertFalse(top(1e-5).ok)  # 10 microns past the cap (#96 reproduction)
        self.assertIn(4, _clauses(top(1e-5)))
        self.assertFalse(top(np.nextafter(c, -np.inf) - atol).ok)  # just inside the near-cap band
        self.assertTrue(top(c + atol).ok)  # just clear of the near cap
        self.assertTrue(top(0.02).ok)  # comfortably inserted

    def test_cylinder_far_cap_band(self):
        # A short cylinder: inserting deep leaves < c to the far cap -> clause 4.
        cyl, c = Cylinder(0.03, 0.05), 0.006
        w = _certify(
            cyl, pose_from([0, 0, 0.05 + L - 0.048], [0, 0, -1], [0, 1, 0]), preshape=0.066, clearance=c, mode="top"
        )
        self.assertFalse(w.ok)
        self.assertIn(4, _clauses(w))

    def test_box_face_insertion_band(self):
        box, c = Box(0.05, 0.06, 0.07), 0.006
        w = _certify(
            box, pose_from([0, 0, 0.07 + L - 1e-5], [0, 0, -1], [0, 1, 0]), preshape=0.066, clearance=c, mode="top"
        )
        self.assertFalse(w.ok)  # 10 microns past the top face (#96 reproduction)
        self.assertIn(4, _clauses(w))


class TestApertureIsPoseRelative(unittest.TestCase):
    """Aperture fit is pose-relative: contacts must lie between the open jaws (#97)."""

    def test_one_metre_offset_fails_clause_2(self):
        # Span (60 mm) fits 66 mm, but the object is 1 m away along y_EE.
        w = _certify(
            Box(0.05, 0.06, 0.07),
            pose_from([0.0, 1.0, 0.13], [0, 0, -1], [0, 1, 0]),
            preshape=0.066,
            clearance=0.006,
            mode="top",
        )
        self.assertFalse(w.ok)
        self.assertIn(2, _clauses(w))

    def test_centered_ok_and_offset_beyond_one_jaw_fails(self):
        box = Box(0.05, 0.06, 0.07)  # span-y = 0.06; jaw at +/-0.033 for preshape 0.066
        self.assertTrue(
            _certify(
                box, pose_from([0, 0, 0.13], [0, 0, -1], [0, 1, 0]), preshape=0.066, clearance=0.006, mode="top"
            ).ok
        )
        # Shift the palm so the +y face falls outside the +jaw: y-contacts at +/-0.03,
        # palm at y=+0.01 -> contacts at +0.02/-0.04 relative to palm; -0.04 < -0.033.
        w = _certify(
            box, pose_from([0, 0.01, 0.13], [0, 0, -1], [0, 1, 0]), preshape=0.066, clearance=0.006, mode="top"
        )
        self.assertFalse(w.ok)
        self.assertIn(2, _clauses(w))

    def test_asymmetric_but_enclosed_is_accepted(self):
        # Palm offset within the jaw margin keeps both contacts inside [-a/2, a/2].
        box = Box(0.05, 0.06, 0.07)
        w = _certify(
            box, pose_from([0, 0.002, 0.13], [0, 0, -1], [0, 1, 0]), preshape=0.070, clearance=0.006, mode="top"
        )
        self.assertTrue(w.ok, w.failed)


class TestSelectors(unittest.TestCase):
    """Structural selectors use hand-occupied approach semantics (#98)."""

    BOX = Box(0.05, 0.06, 0.07)

    def _face(self, approach_label, palm, approach_vec):
        return _certify(
            self.BOX,
            pose_from(palm, approach_vec, [0, 0, 1]),
            preshape=0.076,
            clearance=0.006,
            mode="face",
            approach=approach_label,
        )

    def test_box_faces_accept_hand_occupied_labels(self):
        # Hand on +x side => z_EE = -x, declared approach "+x".
        self.assertTrue(self._face("+x", [0.085, 0, 0.035], [-1, 0, 0]).ok)
        self.assertTrue(self._face("-x", [-0.085, 0, 0.035], [1, 0, 0]).ok)
        self.assertTrue(
            _certify(
                self.BOX,
                pose_from([0, 0.085, 0.035], [0, -1, 0], [0, 0, 1]),
                preshape=0.076,
                clearance=0.006,
                mode="face",
                approach="+y",
            ).ok
        )
        self.assertTrue(
            _certify(
                self.BOX,
                pose_from([0, -0.085, 0.035], [0, 1, 0], [0, 0, 1]),
                preshape=0.076,
                clearance=0.006,
                mode="face",
                approach="-y",
            ).ok
        )

    def test_box_top_bottom_hand_occupied_labels(self):
        self.assertTrue(
            _certify(
                self.BOX,
                pose_from([0, 0, 0.13], [0, 0, -1], [0, 1, 0]),
                preshape=0.066,
                clearance=0.006,
                mode="top",
                approach="+z",
                finger_orientation="y",
            ).ok
        )
        self.assertTrue(
            _certify(
                self.BOX,
                pose_from([0, 0, -0.06], [0, 0, 1], [0, 1, 0]),
                preshape=0.066,
                clearance=0.006,
                mode="bottom",
                approach="-z",
                finger_orientation="y",
            ).ok
        )

    def test_reversed_sign_label_fails_clause_8(self):
        w = self._face("-x", [0.085, 0, 0.035], [-1, 0, 0])  # actually a +x-side grasp
        self.assertFalse(w.ok)
        self.assertIn(8, _clauses(w))

    def test_unknown_selector_raises(self):
        with self.assertRaises(ValueError):
            self._face("nonsense", [0.085, 0, 0.035], [-1, 0, 0])
        with self.assertRaises(ValueError):
            _certify(
                Cylinder(0.03, 0.12),
                pose_from([0.06, 0, 0.06], [-1, 0, 0], [0, 1, 0]),
                preshape=0.066,
                clearance=0.006,
                mode="side",
                finger_orientation="z",
            )

    def test_family_labels_all_modes(self):
        # Each family label is a real geometric predicate, not a dead parameter.
        self.assertTrue(
            _certify(
                Sphere(0.03),
                pose_from([0.06, 0, 0], [-1, 0, 0], [0, 1, 0]),
                preshape=0.066,
                clearance=0.006,
                mode="surface",
                approach="radial",
                finger_orientation="diameter",
            ).ok
        )
        self.assertTrue(
            _certify(
                Cylinder(0.03, 0.12),
                pose_from([0.06, 0, 0.06], [-1, 0, 0], [0, 1, 0]),
                preshape=0.066,
                clearance=0.006,
                mode="side",
                approach="radial",
                finger_orientation="tangential",
            ).ok
        )
        self.assertTrue(
            _certify(
                Torus(0.06, 0.02),
                pose_from([0.12, 0, 0], [-1, 0, 0], [0, 0, 1]),
                preshape=0.044,
                clearance=0.006,
                mode="side",
                approach="tube",
                finger_orientation="tangential",
            ).ok
        )

    def test_omitting_selectors_is_geometry_only(self):
        # No selectors -> geometry-only certification still succeeds.
        self.assertTrue(
            _certify(
                self.BOX,
                pose_from([0.085, 0, 0.035], [-1, 0, 0], [0, 0, 1]),
                preshape=0.076,
                clearance=0.006,
                mode="face",
            ).ok
        )


class TestBuiltinProvenanceIntegration(unittest.TestCase):
    """Real factory provenance selectors agree with the oracle's mapping (#98)."""

    def test_box_face_templates_accept_their_own_provenance(self):
        from tsr.hands import ParallelJawGripper

        g = ParallelJawGripper(finger_length=0.08, max_aperture=0.30)
        templates = g.grasp_box_face_x(0.05, 0.06, 0.07)
        self.assertTrue(templates)
        for t in templates:
            prov = t.provenance
            # Reconstruct the object-frame EE pose at the template's Bw midpoint.
            tsr = t.instantiate(np.eye(4))
            pose = tsr.to_transform(np.zeros(6))
            w = certify(
                Box(0.05, 0.06, 0.07),
                pose,
                finger_length=0.08,
                max_aperture=0.30,
                preshape=float(t.preshape[0]),
                clearance=0.006,
                mode=prov.mode,
                approach=prov.approach,
                finger_orientation=prov.finger_orientation,
            )
            # The selector mapping must agree (no clause-8 mismatch), even if other
            # clauses are the province of the #68-#72 generator repairs.
            self.assertNotIn(8, _clauses(w), (prov.approach, prov.finger_orientation, w.failed))


class TestTorusSideMinorAngles(unittest.TestCase):
    """Torus side across the full minor-angle range, with off-ray rejection (#99)."""

    @staticmethod
    def _side_pose(R, r, alpha, standoff, flip=False):
        normal = np.array([np.cos(alpha), 0.0, np.sin(alpha)])  # outward at (phi=0, alpha)
        approach = -normal
        tube_center = np.array([R, 0.0, 0.0])
        palm = tube_center - approach * (r + standoff)
        tangent = np.array([-np.sin(alpha), 0.0, np.cos(alpha)])
        close = -tangent if flip else tangent
        return pose_from(palm, approach, close)

    def test_certifies_across_minor_angles_both_flips(self):
        R, r, c = 0.06, 0.02, 0.006
        for alpha in (-np.pi / 2, -np.pi / 4, 0.0, np.pi / 4, np.pi / 2):
            for flip in (False, True):
                pose = self._side_pose(R, r, alpha, standoff=0.03, flip=flip)
                w = _certify(Torus(R, r), pose, preshape=2 * r + c, clearance=c, mode="side")
                self.assertTrue(w.ok, (alpha, flip, w.failed))
                self.assertAlmostEqual(w.minor_angle, alpha, delta=1e-6)

    def test_off_ray_contact_is_rejected(self):
        # #99 reproduction: the palm ray x=0.12 along -y never passes through the
        # inferred tube centre; the contact plane is behind the palm.
        w = _certify(
            Torus(0.06, 0.02),
            pose_from([0.12, 0, 0], [0, -1, 0], [0, 0, 1]),
            preshape=0.044,
            clearance=0.006,
            mode="side",
        )
        self.assertFalse(w.ok)
        self.assertIn(5, _clauses(w))

    def test_contact_plane_is_on_the_forward_segment(self):
        w = _certify(
            Torus(0.06, 0.02),
            self._side_pose(0.06, 0.02, np.pi / 4, 0.03),
            preshape=0.046,
            clearance=0.006,
            mode="side",
        )
        self.assertGreater(w.contact_plane_distance, 0.0)
        self.assertLessEqual(w.contact_plane_distance, L + 1e-9)


def _rotz(theta):
    """4x4 rotation about the object z-axis (a symmetry of sphere/cylinder/torus)."""
    c, s = np.cos(theta), np.sin(theta)
    T = np.eye(4)
    T[:2, :2] = [[c, -s], [s, c]]
    return T


class TestAxisTiltAndForwardReach(unittest.TestCase):
    """Axis-aligned modes reject tilt; witnesses require forward, finite planes (#100, #101)."""

    def test_cylinder_cap_tilt_rejected(self):
        theta = 1e-3
        a = [np.sin(theta), 0.0, -np.cos(theta)]
        w = _certify(
            Cylinder(0.03, 0.12),
            pose_from([0, 0, 0.18], a, [0, 1, 0]),
            max_aperture=0.30,
            preshape=0.066,
            clearance=0.006,
            mode="top",
        )
        self.assertFalse(w.ok)  # #100 reproduction

    def test_box_tilt_and_slide_rejected(self):
        theta = 1e-3
        a = [np.sin(theta), 0.0, -np.cos(theta)]
        w = _certify(
            Box(0.05, 0.06, 0.07),
            pose_from([0.019, 0, 0.13], a, [0, 1, 0]),
            max_aperture=0.30,
            preshape=0.066,
            clearance=0.006,
            mode="top",
        )
        self.assertFalse(w.ok)  # #100 reproduction (5.92 mm true slide clearance)

    def test_exact_axis_alignment_still_certifies(self):
        # The angular boundary: the exact canonical pose is accepted.
        self.assertTrue(
            _certify(
                Cylinder(0.03, 0.12),
                pose_from([0, 0, 0.18], [0, 0, -1], [0, 1, 0]),
                preshape=0.066,
                clearance=0.006,
                mode="top",
            ).ok
        )

    def test_angular_alignment_boundary(self):
        # Axis matching is in radians at ANGLE_ATOL: a tilt at the boundary (and
        # just inside) is accepted, just outside is rejected (#100, #102).
        for factor, expect in ((0.99, True), (1.0, True), (1.01, False)):
            theta = ANGLE_ATOL * factor
            a = [np.sin(theta), 0.0, -np.cos(theta)]
            wc = _certify(
                Cylinder(0.03, 0.12),
                pose_from([0, 0, 0.18], a, [0, 1, 0]),
                max_aperture=0.30,
                preshape=0.066,
                clearance=0.006,
                mode="top",
            )
            self.assertEqual(wc.ok, expect, ("cyl", factor, wc.failed))
            wb = _certify(
                Box(0.05, 0.06, 0.07),
                pose_from([0, 0, 0.13], a, [0, 1, 0]),
                max_aperture=0.30,
                preshape=0.066,
                clearance=0.006,
                mode="top",
            )
            self.assertEqual(wb.ok, expect, ("box", factor, wb.failed))

    def test_cylinder_side_missing_ray_rejected(self):
        # #101 reproduction: approach ray y=0.06 misses the r=0.03 cylinder.
        w = _certify(
            Cylinder(0.03, 0.12),
            pose_from([0, 0.06, 0.06], [1, 0, 0], [0, 1, 0]),
            max_aperture=0.30,
            preshape=0.20,
            clearance=0.006,
            mode="side",
            approach="radial",
            finger_orientation="tangential",
        )
        self.assertFalse(w.ok)
        self.assertIn(5, _clauses(w))

    def test_torus_span_equatorial_palm_rejected(self):
        # #101 reproduction: palm in the equatorial plane, not on the +z side.
        w = _certify(
            Torus(0.06, 0.02),
            pose_from([0, 0.10, 0], [0, 0, -1], [0, 1, 0]),
            max_aperture=0.50,
            preshape=0.40,
            clearance=0.006,
            mode="span",
            approach="+z",
            finger_orientation="diameter",
        )
        self.assertFalse(w.ok)

    def test_successful_witnesses_have_finite_forward_planes(self):
        cases = [
            (Sphere(0.03), pose_from([0.06, 0, 0], [-1, 0, 0], [0, 1, 0]), "surface", 0.066, A),
            (Cylinder(0.03, 0.12), pose_from([0.06, 0, 0.06], [-1, 0, 0], [0, 1, 0]), "side", 0.066, A),
            (Cylinder(0.03, 0.12), pose_from([0, 0, 0.18], [0, 0, -1], [0, 1, 0]), "top", 0.066, A),
            (Box(0.05, 0.06, 0.07), pose_from([0, 0, 0.13], [0, 0, -1], [0, 1, 0]), "top", 0.066, A),
            (Torus(0.06, 0.02), pose_from([0, 0, 0.05], [0, 0, -1], [0, 1, 0]), "span", 0.164, 0.30),
            (Torus(0.06, 0.02), pose_from([0.12, 0, 0], [-1, 0, 0], [0, 0, 1]), "side", 0.044, A),
        ]
        for prim, pose, mode, preshape, ap in cases:
            w = _certify(prim, pose, max_aperture=ap, preshape=preshape, clearance=0.006, mode=mode)
            self.assertTrue(w.ok, (mode, w.failed))
            self.assertTrue(np.isfinite(w.realized_depth))
            self.assertTrue(0.0 < w.contact_plane_distance <= L + 1e-9)

    def test_azimuthal_equivariance(self):
        # Rotating the whole config about the object z-axis is a symmetry.
        base = {
            "cyl_side": (Cylinder(0.03, 0.12), pose_from([0.06, 0, 0.06], [-1, 0, 0], [0, 1, 0]), "side", 0.066, A),
            "torus_span": (Torus(0.06, 0.02), pose_from([0, 0, 0.05], [0, 0, -1], [0, 1, 0]), "span", 0.164, 0.30),
            "sphere": (Sphere(0.03), pose_from([0.06, 0, 0], [-1, 0, 0], [0, 1, 0]), "surface", 0.066, A),
        }
        for prim, pose, mode, preshape, ap in base.values():
            for theta in (0.3, 1.1, -2.0):
                w = _certify(prim, _rotz(theta) @ pose, max_aperture=ap, preshape=preshape, clearance=0.006, mode=mode)
                self.assertTrue(w.ok, (mode, theta, w.failed))


class TestInsertionClearanceMatrix(unittest.TestCase):
    """Insertion/clearance bands across caps, faces, boundaries, and scale (#96, #102)."""

    def test_cylinder_bottom_near_and_far_caps(self):
        cyl, c = Cylinder(0.03, 0.12), 0.006
        atol = length_atol(cyl.scale)

        def bottom(depth):  # insertion past the bottom face (z=0), palm below
            return _certify(
                cyl, pose_from([0, 0, -(L - depth)], [0, 0, 1], [0, 1, 0]), preshape=0.066, clearance=c, mode="bottom"
            )

        self.assertFalse(bottom(np.nextafter(c, -np.inf) - atol).ok)
        self.assertTrue(bottom(c + atol).ok)
        self.assertTrue(bottom(0.02).ok)

    def test_every_box_face_slide_boundary(self):
        box, c = Box(0.05, 0.06, 0.07), 0.006
        atol = length_atol(box.scale)
        # (mode, palm, approach, close, slide_axis, slide_half, preshape)
        faces = [
            ("top", [0, 0, 0.13], [0, 0, -1], [0, 1, 0], 0, 0.025, 0.066),  # slide x, span y
            ("bottom", [0, 0, -0.06], [0, 0, 1], [0, 1, 0], 0, 0.025, 0.066),
            ("face", [0.085, 0, 0.035], [-1, 0, 0], [0, 0, 1], 1, 0.03, 0.076),  # +x face, span z, slide y
            ("face", [-0.085, 0, 0.035], [1, 0, 0], [0, 0, 1], 1, 0.03, 0.076),  # -x face
            ("face", [0, 0.085, 0.035], [0, -1, 0], [0, 0, 1], 0, 0.025, 0.076),  # +y face, slide x
            ("face", [0, -0.085, 0.035], [0, 1, 0], [0, 0, 1], 0, 0.025, 0.076),  # -y face
        ]
        for mode, palm, ap, cl, slide_ax, half, preshape in faces:
            for shift, expect_ok in ((half - c - atol, True), (half - c + 2 * atol, False)):
                pp = list(palm)
                pp[slide_ax] = shift
                w = _certify(box, pose_from(pp, ap, cl), preshape=preshape, clearance=c, mode=mode)
                self.assertEqual(w.ok, expect_ok, (mode, palm, shift, w.failed))

    def test_scale_regime(self):
        for k in (0.1, 10.0):
            w = _certify(
                Cylinder(0.03 * k, 0.12 * k),
                pose_from([0, 0, 0.18 * k], [0, 0, -1], [0, 1, 0]),
                finger_length=L * k,
                max_aperture=A * k,
                preshape=0.066 * k,
                clearance=0.006 * k,
                mode="top",
            )
            self.assertTrue(w.ok, (k, w.failed))

    def test_cylinder_cap_approach_and_far_boundaries(self):
        c = 0.006
        tall, short = Cylinder(0.03, 0.12), Cylinder(0.03, 0.05)  # far cap reachable only when h < L
        atT, atS = length_atol(tall.scale), length_atol(short.scale)

        def cert(cyl, mode, pz):
            ap = [0, 0, -1] if mode == "top" else [0, 0, 1]
            return _certify(
                cyl, pose_from([0, 0, pz], ap, [0, 1, 0]), max_aperture=0.30, preshape=0.066, clearance=c, mode=mode
            )

        # approached-cap margin = realized_depth (tall cylinder)
        self.assertTrue(cert(tall, "top", 0.12 + L - (c + atT)).ok)
        self.assertIn(4, _clauses(cert(tall, "top", 0.12 + L - (c - 2 * atT))))
        self.assertTrue(cert(tall, "bottom", (c + atT) - L).ok)
        self.assertIn(4, _clauses(cert(tall, "bottom", (c - 2 * atT) - L)))
        # far-cap margin = height - realized_depth (short cylinder, h < L)
        self.assertTrue(cert(short, "top", L + (c + atS)).ok)
        self.assertIn(4, _clauses(cert(short, "top", L + (c - 2 * atS))))
        self.assertTrue(cert(short, "bottom", (0.05 - (c + atS)) - L).ok)
        self.assertIn(4, _clauses(cert(short, "bottom", (0.05 - (c - 2 * atS)) - L)))

    def test_box_approach_and_far_boundaries(self):
        c = 0.006
        box = Box(0.05, 0.06, 0.07)  # dz = 0.07 < L, so the far face is reachable
        at = length_atol(box.scale)

        def top(pz):
            return _certify(
                box,
                pose_from([0, 0, pz], [0, 0, -1], [0, 1, 0]),
                max_aperture=0.30,
                preshape=0.066,
                clearance=c,
                mode="top",
            )

        def facex(px):
            return _certify(
                box,
                pose_from([px, 0, 0.035], [-1, 0, 0], [0, 0, 1]),
                max_aperture=0.30,
                preshape=0.076,
                clearance=c,
                mode="face",
            )

        # box top approached-face and far-face margins
        self.assertTrue(top(0.07 + L - (c + at)).ok)
        self.assertIn(4, _clauses(top(0.07 + L - (c - 2 * at))))
        self.assertTrue(top(L + (c + at)).ok)
        self.assertIn(4, _clauses(top(L + (c - 2 * at))))
        # box +x face approached-face and far-face margins
        self.assertTrue(facex(0.025 + L - (c + at)).ok)
        self.assertIn(4, _clauses(facex(0.025 + L - (c - 2 * at))))
        self.assertTrue(facex(0.055 + (c + at)).ok)
        self.assertIn(4, _clauses(facex(0.055 + (c - 2 * at))))


class TestApertureMatrix(unittest.TestCase):
    """Pose-relative aperture across primitives, both jaw sides, boundaries (#97, #102)."""

    def test_both_jaw_sides_across_primitives(self):
        # Shifting the palm along +/- y_EE pushes one contact past the corresponding jaw.
        specs = [
            (Sphere(0.03), [0.06, 0, 0], [-1, 0, 0], [0, 1, 0], 0.066, A),
            (Cylinder(0.03, 0.12), [0.06, 0, 0.06], [-1, 0, 0], [0, 1, 0], 0.066, A),
            (Box(0.05, 0.06, 0.07), [0, 0, 0.13], [0, 0, -1], [0, 1, 0], 0.066, A),
        ]
        for prim, palm, ap, cl, preshape, aperture in specs:
            self.assertTrue(
                _certify(
                    prim,
                    pose_from(palm, ap, cl),
                    max_aperture=aperture,
                    preshape=preshape,
                    clearance=0.006,
                    mode=_mode_for(prim),
                ).ok
            )
            for sign in (+1, -1):
                shifted = list(palm)
                # Move palm along the closing axis so one contact exits that jaw.
                shifted[1] += sign * 0.02
                w = _certify(
                    prim,
                    pose_from(shifted, ap, cl),
                    max_aperture=aperture,
                    preshape=preshape,
                    clearance=0.006,
                    mode=_mode_for(prim),
                )
                self.assertFalse(w.ok, (type(prim).__name__, sign))
                self.assertIn(2, _clauses(w))

    def test_aperture_boundary_nextafter(self):
        r, c = 0.03, 0.006
        pose = pose_from([0.06, 0, 0], [-1, 0, 0], [0, 1, 0])
        atol = length_atol(r)
        # span = 2r = 0.06; strict fit needs preshape > 0.06 + 2*atol.
        self.assertTrue(_certify(Sphere(r), pose, preshape=2 * r + 4 * atol, clearance=c, mode="surface").ok)
        self.assertFalse(
            _certify(Sphere(r), pose, preshape=np.nextafter(2 * r, -np.inf), clearance=c, mode="surface").ok
        )

    def test_boundary_neighbours_all_centered_modes(self):
        # Exact fit boundary (preshape = span + 2*atol is accepted) and both
        # neighbours, for every centred mode including torus side and span (#97, #102).
        specs = [
            (Sphere(0.03), pose_from([0.06, 0, 0], [-1, 0, 0], [0, 1, 0]), "surface", 0.06, A),
            (Cylinder(0.03, 0.12), pose_from([0.06, 0, 0.06], [-1, 0, 0], [0, 1, 0]), "side", 0.06, A),
            (Torus(0.06, 0.02), pose_from([0.12, 0, 0], [-1, 0, 0], [0, 0, 1]), "side", 0.04, A),
            (Torus(0.06, 0.02), pose_from([0, 0, 0.05], [0, 0, -1], [0, 1, 0]), "span", 0.16, 0.50),
        ]
        for prim, pose, mode, span, ap in specs:
            at = length_atol(prim.scale)
            self.assertTrue(
                _certify(prim, pose, max_aperture=ap, preshape=span + 4 * at, clearance=0.006, mode=mode).ok, mode
            )
            self.assertTrue(
                _certify(prim, pose, max_aperture=ap, preshape=span + 2 * at, clearance=0.006, mode=mode).ok, mode
            )
            self.assertFalse(_certify(prim, pose, max_aperture=ap, preshape=span, clearance=0.006, mode=mode).ok, mode)

    def test_torus_span_both_jaw_sides_and_asymmetric(self):
        R, r, c = 0.06, 0.02, 0.006
        preshape = 2 * (R + r) + c  # jaw half = R + r + c/2
        # Asymmetric but fully enclosed (palm shifted < c/2 along y_EE) still certifies.
        self.assertTrue(
            _certify(
                Torus(R, r),
                pose_from([0, c * 0.25, 0.05], [0, 0, -1], [0, 1, 0]),
                max_aperture=0.50,
                preshape=preshape,
                clearance=c,
                mode="span",
            ).ok
        )
        # Shifting past c/2 pushes the corresponding contact outside that jaw.
        for dy in (c, -c):
            w = _certify(
                Torus(R, r),
                pose_from([0, dy, 0.05], [0, 0, -1], [0, 1, 0]),
                max_aperture=0.50,
                preshape=preshape,
                clearance=c,
                mode="span",
            )
            self.assertFalse(w.ok, dy)
            self.assertIn(2, _clauses(w))


def _mode_for(prim):
    return {"Sphere": "surface", "Cylinder": "side", "Box": "top", "Torus": "span"}[type(prim).__name__]


class TestSelectorMatrix(unittest.TestCase):
    """Every built-in (primitive, mode) accepts its labels and rejects wrong ones (#98, #102)."""

    CASES = [
        (Sphere(0.03), pose_from([0.06, 0, 0], [-1, 0, 0], [0, 1, 0]), "surface", "radial", "diameter", 0.066, A),
        (
            Cylinder(0.03, 0.12),
            pose_from([0.06, 0, 0.06], [-1, 0, 0], [0, 1, 0]),
            "side",
            "radial",
            "tangential",
            0.066,
            A,
        ),
        (Cylinder(0.03, 0.12), pose_from([0, 0, 0.18], [0, 0, -1], [0, 1, 0]), "top", "+z", "diameter", 0.066, A),
        (Cylinder(0.03, 0.12), pose_from([0, 0, -0.06], [0, 0, 1], [0, 1, 0]), "bottom", "-z", "diameter", 0.066, A),
        (Box(0.05, 0.06, 0.07), pose_from([0, 0, 0.13], [0, 0, -1], [0, 1, 0]), "top", "+z", "y", 0.066, A),
        (Box(0.05, 0.06, 0.07), pose_from([0, 0, -0.06], [0, 0, 1], [1, 0, 0]), "bottom", "-z", "x", 0.056, A),
        (Box(0.05, 0.06, 0.07), pose_from([0.085, 0, 0.035], [-1, 0, 0], [0, 0, 1]), "face", "+x", "z", 0.076, A),
        (Torus(0.06, 0.02), pose_from([0, 0, 0.05], [0, 0, -1], [0, 1, 0]), "span", "+z", "diameter", 0.164, 0.30),
        (Torus(0.06, 0.02), pose_from([0.12, 0, 0], [-1, 0, 0], [0, 0, 1]), "side", "tube", "tangential", 0.044, A),
    ]

    def test_allowed_labels_accepted(self):
        for prim, pose, mode, approach, fo, preshape, ap in self.CASES:
            w = _certify(
                prim,
                pose,
                max_aperture=ap,
                preshape=preshape,
                clearance=0.006,
                mode=mode,
                approach=approach,
                finger_orientation=fo,
            )
            self.assertTrue(w.ok, (type(prim).__name__, mode, w.failed))

    def test_unknown_label_raises_valueerror(self):
        for prim, pose, mode, _a, _fo, preshape, ap in self.CASES:
            with self.assertRaises(ValueError):
                certify(
                    prim,
                    pose,
                    finger_length=L,
                    max_aperture=ap,
                    preshape=preshape,
                    clearance=0.006,
                    mode=mode,
                    approach="bogus",
                )

    def test_non_string_selector_raises_valueerror(self):
        prim, pose, mode, _a, _fo, preshape, ap = self.CASES[0]
        for bad in (5, ["radial"], object()):
            with self.assertRaises(ValueError):
                certify(
                    prim,
                    pose,
                    finger_length=L,
                    max_aperture=ap,
                    preshape=preshape,
                    clearance=0.006,
                    mode=mode,
                    approach=bad,
                )

    def test_every_factory_family_provenance_agrees(self):
        from tsr.hands import ParallelJawGripper

        g = ParallelJawGripper(finger_length=0.08, max_aperture=0.30)
        families = {
            "cylinder": (Cylinder(0.03, 0.12), g.grasp_cylinder(0.03, 0.12)),
            "box": (Box(0.05, 0.06, 0.07), g.grasp_box(0.05, 0.06, 0.07)),
            "sphere": (Sphere(0.03), g.grasp_sphere(0.03)),
            "torus": (Torus(0.06, 0.02), g.grasp_torus(0.06, 0.02)),
        }
        for name, (prim, templates) in families.items():
            self.assertTrue(templates, name)
            for t in templates:
                prov = t.provenance
                pose = t.instantiate(np.eye(4)).to_transform(np.zeros(6))
                w = certify(
                    prim,
                    pose,
                    finger_length=0.08,
                    max_aperture=0.30,
                    preshape=float(t.preshape[0]),
                    clearance=0.006,
                    mode=prov.mode,
                    approach=prov.approach,
                    finger_orientation=prov.finger_orientation,
                )
                self.assertNotIn(8, _clauses(w), (name, prov.mode, prov.approach, prov.finger_orientation, w.failed))


class TestTorusSideMatrix(unittest.TestCase):
    """Torus side across azimuths x minor angles x flips, with reach checks (#99, #102)."""

    @staticmethod
    def _pose(R, r, phi, alpha, standoff, flip=False):
        radial = np.array([np.cos(phi), np.sin(phi), 0.0])
        normal = np.cos(alpha) * radial + np.sin(alpha) * np.array([0, 0, 1.0])
        approach = -normal
        tube_center = R * radial
        palm = tube_center - approach * (r + standoff)
        tangent = -np.sin(alpha) * radial + np.cos(alpha) * np.array([0, 0, 1.0])
        close = -tangent if flip else tangent
        return pose_from(palm, approach, close)

    def test_azimuth_minor_flip_grid(self):
        R, r, c = 0.06, 0.02, 0.006
        for phi in (0.0, 0.7, 2.5, -1.3):
            for alpha in (-np.pi / 2, -np.pi / 4, 0.0, np.pi / 4, np.pi / 2):
                for flip in (False, True):
                    w = _certify(
                        Torus(R, r),
                        self._pose(R, r, phi, alpha, 0.03, flip),
                        preshape=2 * r + c,
                        clearance=c,
                        mode="side",
                    )
                    self.assertTrue(w.ok, (phi, alpha, flip, w.failed))
                    self.assertAlmostEqual(w.minor_angle, alpha, delta=1e-6)

    def test_scale_regimes(self):
        for k in (0.1, 10.0):
            R, r = 0.06 * k, 0.02 * k
            w = _certify(
                Torus(R, r),
                self._pose(R, r, 0.4, np.pi / 4, 0.03 * k),
                finger_length=L * k,
                max_aperture=A * k,
                preshape=(2 * r + 0.006 * k),
                clearance=0.006 * k,
                mode="side",
            )
            self.assertTrue(w.ok, (k, w.failed))

    def test_reach_boundary(self):
        # t_center = r + standoff must lie in the forward interval (0, L]. At exactly
        # L it is accepted; just beyond, clause 5 (#99, #102).
        R, r, c = 0.06, 0.02, 0.006
        at = length_atol(R + r)
        self.assertTrue(
            _certify(Torus(R, r), self._pose(R, r, 0.0, 0.0, L - r), preshape=2 * r + c, clearance=c, mode="side").ok
        )
        self.assertTrue(
            _certify(
                Torus(R, r), self._pose(R, r, 0.0, 0.0, L - r - 2 * at), preshape=2 * r + c, clearance=c, mode="side"
            ).ok
        )
        w = _certify(
            Torus(R, r), self._pose(R, r, 0.0, 0.0, L - r + 2 * at), preshape=2 * r + c, clearance=c, mode="side"
        )
        self.assertFalse(w.ok)
        self.assertIn(5, _clauses(w))

    def test_vertical_over_axis_degenerate(self):
        # A vertical approach over the torus axis has an undefined azimuth (#99).
        w = _certify(
            Torus(0.06, 0.02),
            pose_from([0, 0, 0.05], [0, 0, -1], [1, 0, 0]),
            preshape=0.046,
            clearance=0.006,
            mode="side",
        )
        self.assertFalse(w.ok)
        self.assertIn(8, _clauses(w))

    def test_near_vertical_minor_angle(self):
        # alpha just inside +pi/2 (near-vertical approach) still certifies with the
        # azimuth recovered from the horizontal approach component.
        R, r = 0.06, 0.02
        w = _certify(
            Torus(R, r),
            self._pose(R, r, 0.5, np.pi / 2 - 1e-3, 0.03),
            preshape=2 * r + 0.006,
            clearance=0.006,
            mode="side",
        )
        self.assertTrue(w.ok, w.failed)
        self.assertAlmostEqual(w.minor_angle, np.pi / 2 - 1e-3, delta=1e-6)


if __name__ == "__main__":
    unittest.main()
