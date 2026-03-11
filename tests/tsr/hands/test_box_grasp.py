"""Tests for ParallelJawGripper box grasp primitives.

Box dimensions used:
  - SMALL cube: 0.06 × 0.06 × 0.06 m — all orientations fit (preshape < 0.14)
  - TALL box:   0.08 × 0.06 × 0.20 m — BOX_Z = 0.20 > max_aperture = 0.14
                so "span-z" orientations on x/y faces are filtered out.
"""
import unittest
import numpy as np

from tsr.hands import ParallelJawGripper
from tsr.template import TSRTemplate

# Small cube — all 2 orientations per face valid
SX, SY, SZ = 0.060, 0.060, 0.060

# Tall box — span-z orientations on ±x and ±y faces exceed max_aperture = 0.140
TX, TY, TZ = 0.080, 0.060, 0.200


def _check_se3(tc: unittest.TestCase, R: np.ndarray) -> None:
    tc.assertEqual(R.shape, (3, 3))
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)
    np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-10)


class TestBoxFaceTemplatesHelper(unittest.TestCase):
    """Core correctness checks on _box_face_templates via the public API."""

    def setUp(self):
        self.gripper = ParallelJawGripper(finger_length=0.055, max_aperture=0.140)

    def test_all_tw_e_valid_se3_small_cube(self):
        for t in self.gripper.grasp_box(SX, SY, SZ):
            _check_se3(self, t.Tw_e[:3, :3])

    def test_all_tw_e_valid_se3_tall_box(self):
        for t in self.gripper.grasp_box(TX, TY, TZ):
            _check_se3(self, t.Tw_e[:3, :3])

    def test_bw_exactly_one_translational_dof(self):
        """Each template has freedom in exactly one Bw row (the slide direction)."""
        for t in self.gripper.grasp_box(SX, SY, SZ):
            free_rows = [i for i in range(3) if t.Bw[i, 0] != t.Bw[i, 1]]
            self.assertEqual(len(free_rows), 1,
                             msg=f"Expected 1 free Bw row, got {free_rows} in {t.name}")

    def test_bw_no_rotational_dof(self):
        for t in self.gripper.grasp_box(SX, SY, SZ):
            for row in (3, 4, 5):
                self.assertEqual(t.Bw[row, 0], t.Bw[row, 1])

    def test_palm_offset_is_h_palm_outside_face(self):
        """Translation in Tw_e is -z_EE * h_palm (palm outside the face)."""
        for t in self.gripper.grasp_box(SX, SY, SZ):
            z_ee = t.Tw_e[:3, 2]
            trans = t.Tw_e[:3, 3]
            # trans should be antiparallel to z_ee
            h_palm = np.linalg.norm(trans)
            if h_palm > 1e-12:
                np.testing.assert_allclose(trans / h_palm, -z_ee, atol=1e-10,
                                           err_msg=f"Translation not along -z_EE in {t.name}")

    def test_preshape_matches_span_dim(self):
        """Auto-computed preshape = span_dim + clearance for each orientation."""
        clearance = 0.1 * self.gripper.finger_length
        for t in self.gripper.grasp_box(SX, SY, SZ):
            ps = t.preshape[0]
            # preshape should be close to one of the box face dimensions + clearance
            candidates = [SX + clearance, SY + clearance, SZ + clearance]
            self.assertTrue(
                any(abs(ps - c) < 1e-9 for c in candidates),
                msg=f"preshape {ps:.4f} not near any face dim + clearance in {t.name}"
            )


class TestParallelJawGripperBoxTop(unittest.TestCase):

    def setUp(self):
        self.gripper = ParallelJawGripper(finger_length=0.055, max_aperture=0.140)

    def test_small_cube_returns_2k_templates(self):
        # Both orientations (span-x and span-y) fit within max_aperture
        self.assertEqual(len(self.gripper.grasp_box_top(SX, SY, SZ)), 6)  # 2*k=2*3

    def test_k1_returns_two_templates(self):
        self.assertEqual(len(self.gripper.grasp_box_top(SX, SY, SZ, k=1)), 2)

    def test_z_axis_points_down(self):
        for t in self.gripper.grasp_box_top(SX, SY, SZ):
            np.testing.assert_allclose(t.Tw_e[:3, 2], [0., 0., -1.], atol=1e-10)

    def test_tsr_origin_at_box_top(self):
        for t in self.gripper.grasp_box_top(SX, SY, SZ):
            self.assertAlmostEqual(t.T_ref_tsr[2, 3], SZ)

    def test_preshape_exceeds_aperture_raises(self):
        with self.assertRaises(ValueError):
            self.gripper.grasp_box_top(SX, SY, SZ,
                                       preshape=self.gripper.max_aperture + 0.01)

    def test_zero_dim_raises(self):
        with self.assertRaises(ValueError):
            self.gripper.grasp_box_top(0.0, SY, SZ)

    def test_tw_e_valid_se3(self):
        for t in self.gripper.grasp_box_top(SX, SY, SZ):
            _check_se3(self, t.Tw_e[:3, :3])

    def test_palm_offset_decreases_with_depth(self):
        # h_palm = finger_length - d; as depth index increases, h_palm decreases
        names_zs = [(t.name, t.Tw_e[2, 3])
                    for t in self.gripper.grasp_box_top(SX, SY, SZ, k=3)
                    if "span-x" in t.name]
        z_vals = [z for _, z in names_zs]
        self.assertEqual(sorted(z_vals, reverse=True), z_vals)

    def test_names_contain_span_labels(self):
        names = [t.name for t in self.gripper.grasp_box_top(SX, SY, SZ)]
        self.assertTrue(any("span-x" in n for n in names))
        self.assertTrue(any("span-y" in n for n in names))

    def test_span_x_slides_in_y(self):
        clearance = 0.1 * self.gripper.finger_length
        span_x_templates = [t for t in self.gripper.grasp_box_top(SX, SY, SZ)
                             if "span-x" in t.name]
        for t in span_x_templates:
            self.assertEqual(t.Bw[0, 0], t.Bw[0, 1])   # x fixed
            self.assertAlmostEqual(t.Bw[1, 0], -(SY / 2 - clearance))  # y slides
            self.assertAlmostEqual(t.Bw[1, 1],  (SY / 2 - clearance))

    def test_span_y_slides_in_x(self):
        clearance = 0.1 * self.gripper.finger_length
        span_y_templates = [t for t in self.gripper.grasp_box_top(SX, SY, SZ)
                             if "span-y" in t.name]
        for t in span_y_templates:
            self.assertEqual(t.Bw[1, 0], t.Bw[1, 1])   # y fixed
            self.assertAlmostEqual(t.Bw[0, 0], -(SX / 2 - clearance))  # x slides
            self.assertAlmostEqual(t.Bw[0, 1],  (SX / 2 - clearance))

    def test_wide_box_span_x_filtered(self):
        """If box_x > max_aperture - clearance, span-x orientation is skipped."""
        wide = self.gripper.max_aperture + 0.01   # definitely too wide
        templates = self.gripper.grasp_box_top(wide, SY, SZ)
        self.assertTrue(all("span-x" not in t.name for t in templates))
        # span-y should still be there (SY=0.06 fits)
        self.assertTrue(any("span-y" in t.name for t in templates))

    def test_instantiate_and_sample_valid(self):
        for t in self.gripper.grasp_box_top(SX, SY, SZ):
            pose = t.instantiate(np.eye(4)).sample()
            _check_se3(self, pose[:3, :3])


class TestParallelJawGripperBoxBottom(unittest.TestCase):

    def setUp(self):
        self.gripper = ParallelJawGripper(finger_length=0.055, max_aperture=0.140)

    def test_small_cube_returns_2k_templates(self):
        self.assertEqual(len(self.gripper.grasp_box_bottom(SX, SY, SZ)), 6)

    def test_z_axis_points_up(self):
        for t in self.gripper.grasp_box_bottom(SX, SY, SZ):
            np.testing.assert_allclose(t.Tw_e[:3, 2], [0., 0., 1.], atol=1e-10)

    def test_tsr_origin_at_z_zero(self):
        for t in self.gripper.grasp_box_bottom(SX, SY, SZ):
            self.assertAlmostEqual(t.T_ref_tsr[2, 3], 0.0)

    def test_tw_e_valid_se3(self):
        for t in self.gripper.grasp_box_bottom(SX, SY, SZ):
            _check_se3(self, t.Tw_e[:3, :3])

    def test_instantiate_and_sample_valid(self):
        for t in self.gripper.grasp_box_bottom(SX, SY, SZ):
            pose = t.instantiate(np.eye(4)).sample()
            _check_se3(self, pose[:3, :3])


class TestParallelJawGripperBoxFaceX(unittest.TestCase):

    def setUp(self):
        self.gripper = ParallelJawGripper(finger_length=0.055, max_aperture=0.140)

    def test_small_cube_returns_4k_templates(self):
        # 2 faces × 2 orientations × k=3 = 12
        self.assertEqual(len(self.gripper.grasp_box_face_x(SX, SY, SZ)), 12)

    def test_k1_returns_four_templates(self):
        self.assertEqual(len(self.gripper.grasp_box_face_x(SX, SY, SZ, k=1)), 4)

    def test_tall_box_span_z_filtered(self):
        """Span-z on ±x face of tall box exceeds max_aperture → filtered out."""
        templates = self.gripper.grasp_box_face_x(TX, TY, TZ)
        self.assertTrue(all("span-z" not in t.name for t in templates))
        # span-y (TY=0.06) should still be present for both ±x faces
        self.assertTrue(any("+x (span-y)" in t.name for t in templates))
        self.assertTrue(any("-x (span-y)" in t.name for t in templates))

    def test_approach_directions(self):
        templates = self.gripper.grasp_box_face_x(SX, SY, SZ, k=1)
        pos_x = [t for t in templates if "+x" in t.name]
        neg_x = [t for t in templates if "-x" in t.name]
        for t in pos_x:
            np.testing.assert_allclose(t.Tw_e[:3, 2], [-1., 0., 0.], atol=1e-10)
        for t in neg_x:
            np.testing.assert_allclose(t.Tw_e[:3, 2], [+1., 0., 0.], atol=1e-10)

    def test_tw_e_valid_se3(self):
        for t in self.gripper.grasp_box_face_x(SX, SY, SZ):
            _check_se3(self, t.Tw_e[:3, :3])

    def test_span_y_slides_in_z_only(self):
        clearance = 0.1 * self.gripper.finger_length
        span_y = [t for t in self.gripper.grasp_box_face_x(SX, SY, SZ)
                  if "span-y" in t.name]
        for t in span_y:
            self.assertEqual(t.Bw[0, 0], t.Bw[0, 1])  # x fixed (radial)
            self.assertEqual(t.Bw[1, 0], t.Bw[1, 1])  # y fixed (span)
            self.assertAlmostEqual(t.Bw[2, 0], -(SZ / 2 - clearance))  # z slides
            self.assertAlmostEqual(t.Bw[2, 1],  (SZ / 2 - clearance))

    def test_span_z_slides_in_y_only(self):
        clearance = 0.1 * self.gripper.finger_length
        span_z = [t for t in self.gripper.grasp_box_face_x(SX, SY, SZ)
                  if "span-z" in t.name]
        for t in span_z:
            self.assertEqual(t.Bw[0, 0], t.Bw[0, 1])  # x fixed (radial)
            self.assertAlmostEqual(t.Bw[1, 0], -(SY / 2 - clearance))  # y slides
            self.assertAlmostEqual(t.Bw[1, 1],  (SY / 2 - clearance))
            self.assertEqual(t.Bw[2, 0], t.Bw[2, 1])  # z fixed (span)

    def test_preshape_exceeds_aperture_raises(self):
        with self.assertRaises(ValueError):
            self.gripper.grasp_box_face_x(SX, SY, SZ,
                                          preshape=self.gripper.max_aperture + 0.01)

    def test_instantiate_and_sample_valid(self):
        for t in self.gripper.grasp_box_face_x(SX, SY, SZ):
            pose = t.instantiate(np.eye(4)).sample()
            _check_se3(self, pose[:3, :3])


class TestParallelJawGripperBoxFaceY(unittest.TestCase):

    def setUp(self):
        self.gripper = ParallelJawGripper(finger_length=0.055, max_aperture=0.140)

    def test_small_cube_returns_4k_templates(self):
        self.assertEqual(len(self.gripper.grasp_box_face_y(SX, SY, SZ)), 12)

    def test_tall_box_span_z_filtered(self):
        templates = self.gripper.grasp_box_face_y(TX, TY, TZ)
        self.assertTrue(all("span-z" not in t.name for t in templates))
        self.assertTrue(any("+y (span-x)" in t.name for t in templates))
        self.assertTrue(any("-y (span-x)" in t.name for t in templates))

    def test_approach_directions(self):
        templates = self.gripper.grasp_box_face_y(SX, SY, SZ, k=1)
        pos_y = [t for t in templates if "+y" in t.name]
        neg_y = [t for t in templates if "-y" in t.name]
        for t in pos_y:
            np.testing.assert_allclose(t.Tw_e[:3, 2], [0., -1., 0.], atol=1e-10)
        for t in neg_y:
            np.testing.assert_allclose(t.Tw_e[:3, 2], [0., +1., 0.], atol=1e-10)

    def test_tw_e_valid_se3(self):
        for t in self.gripper.grasp_box_face_y(SX, SY, SZ):
            _check_se3(self, t.Tw_e[:3, :3])

    def test_span_x_slides_in_z_only(self):
        clearance = 0.1 * self.gripper.finger_length
        span_x = [t for t in self.gripper.grasp_box_face_y(SX, SY, SZ)
                  if "span-x" in t.name]
        for t in span_x:
            self.assertEqual(t.Bw[1, 0], t.Bw[1, 1])  # y fixed (radial)
            self.assertEqual(t.Bw[0, 0], t.Bw[0, 1])  # x fixed (span)
            self.assertAlmostEqual(t.Bw[2, 0], -(SZ / 2 - clearance))
            self.assertAlmostEqual(t.Bw[2, 1],  (SZ / 2 - clearance))

    def test_instantiate_and_sample_valid(self):
        for t in self.gripper.grasp_box_face_y(SX, SY, SZ):
            pose = t.instantiate(np.eye(4)).sample()
            _check_se3(self, pose[:3, :3])


class TestParallelJawGripperBoxCombined(unittest.TestCase):

    def setUp(self):
        self.gripper = ParallelJawGripper(finger_length=0.055, max_aperture=0.140)

    def test_small_cube_returns_12k_templates(self):
        # 6 faces × 2 orientations × k=3 = 36, but small cube all fit
        # face_x: 4k=12, face_y: 4k=12, top: 2k=6, bottom: 2k=6 → 36 total
        self.assertEqual(len(self.gripper.grasp_box(SX, SY, SZ)), 36)

    def test_k1_small_cube(self):
        self.assertEqual(len(self.gripper.grasp_box(SX, SY, SZ, k=1)), 12)

    def test_tall_box_span_z_filtered(self):
        """For tall box, span-z grasps on ±x and ±y are filtered."""
        templates = self.gripper.grasp_box(TX, TY, TZ)
        # face_x: 2 faces × 1 valid orient × k = 2k=6
        # face_y: 2 faces × 1 valid orient × k = 2k=6
        # top: TX=0.08 preshape=0.0855 ✓, TY=0.06 preshape=0.0655 ✓ → 2k=6
        # bottom: same → 2k=6
        # total: 6+6+6+6 = 24
        self.assertEqual(len(templates), 24)

    def test_all_templates_are_tsrtemplate(self):
        for t in self.gripper.grasp_box(SX, SY, SZ):
            self.assertIsInstance(t, TSRTemplate)

    def test_all_tw_e_valid_se3(self):
        for t in self.gripper.grasp_box(SX, SY, SZ):
            _check_se3(self, t.Tw_e[:3, :3])

    def test_task_subject_reference(self):
        templates = self.gripper.grasp_box(SX, SY, SZ, subject="ee", reference="book")
        for t in templates:
            self.assertEqual(t.task, "grasp")
            self.assertEqual(t.subject, "ee")
            self.assertEqual(t.reference, "book")

    def test_instantiate_and_sample_all_valid(self):
        for t in self.gripper.grasp_box(SX, SY, SZ):
            pose = t.instantiate(np.eye(4)).sample()
            _check_se3(self, pose[:3, :3])
