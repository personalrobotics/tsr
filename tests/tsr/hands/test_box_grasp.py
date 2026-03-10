"""Tests for ParallelJawGripper box grasp primitives."""
import unittest
import numpy as np
from numpy import pi

from tsr.hands import ParallelJawGripper
from tsr.template import TSRTemplate

BX = 0.12   # box x dimension [m]
BY = 0.08   # box y dimension [m]
BZ = 0.20   # box z dimension [m]


def _check_se3(tc: unittest.TestCase, R: np.ndarray) -> None:
    tc.assertEqual(R.shape, (3, 3))
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)
    np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-10)


class TestParallelJawGripperBoxTop(unittest.TestCase):

    def setUp(self):
        self.gripper = ParallelJawGripper(finger_length=0.055, max_aperture=0.140)

    def test_default_returns_k_templates(self):
        self.assertEqual(len(self.gripper.grasp_box_top(BX, BY, BZ)), 3)

    def test_k1_returns_one_template(self):
        self.assertEqual(len(self.gripper.grasp_box_top(BX, BY, BZ, k=1)), 1)

    def test_k2_returns_two_templates(self):
        self.assertEqual(len(self.gripper.grasp_box_top(BX, BY, BZ, k=2)), 2)

    def test_preshape_exceeds_aperture_raises(self):
        with self.assertRaises(ValueError):
            self.gripper.grasp_box_top(BX, BY, BZ,
                                       preshape=self.gripper.max_aperture + 0.01)

    def test_zero_dim_raises(self):
        with self.assertRaises(ValueError):
            self.gripper.grasp_box_top(0.0, BY, BZ)
        with self.assertRaises(ValueError):
            self.gripper.grasp_box_top(BX, 0.0, BZ)
        with self.assertRaises(ValueError):
            self.gripper.grasp_box_top(BX, BY, 0.0)

    def test_all_templates_are_tsrtemplate(self):
        for t in self.gripper.grasp_box_top(BX, BY, BZ):
            self.assertIsInstance(t, TSRTemplate)

    def test_bw_shape_no_rotation(self):
        for t in self.gripper.grasp_box_top(BX, BY, BZ):
            self.assertEqual(t.Bw.shape, (6, 2))
            # no rotational freedom
            for row in (3, 4, 5):
                self.assertEqual(t.Bw[row, 0], t.Bw[row, 1])

    def test_bw_translational_freedom(self):
        clearance = 0.1 * self.gripper.finger_length
        for t in self.gripper.grasp_box_top(BX, BY, BZ):
            # x slides within face
            self.assertAlmostEqual(t.Bw[0, 0], -(BX / 2 - clearance))
            self.assertAlmostEqual(t.Bw[0, 1],  (BX / 2 - clearance))
            # y slides within face
            self.assertAlmostEqual(t.Bw[1, 0], -(BY / 2 - clearance))
            self.assertAlmostEqual(t.Bw[1, 1],  (BY / 2 - clearance))
            # z fixed
            self.assertEqual(t.Bw[2, 0], t.Bw[2, 1])

    def test_tsr_origin_at_box_top(self):
        for t in self.gripper.grasp_box_top(BX, BY, BZ):
            self.assertAlmostEqual(t.T_ref_tsr[2, 3], BZ)

    def test_z_axis_points_down(self):
        for t in self.gripper.grasp_box_top(BX, BY, BZ):
            np.testing.assert_allclose(t.Tw_e[:3, 2], [0., 0., -1.], atol=1e-10)

    def test_tw_e_is_valid_se3(self):
        for t in self.gripper.grasp_box_top(BX, BY, BZ):
            _check_se3(self, t.Tw_e[:3, :3])

    def test_palm_offset_decreases_with_depth(self):
        z_offsets = [t.Tw_e[2, 3] for t in self.gripper.grasp_box_top(BX, BY, BZ, k=3)]
        self.assertEqual(sorted(z_offsets, reverse=True), z_offsets)

    def test_names_contain_depth_labels(self):
        names = [t.name for t in self.gripper.grasp_box_top(BX, BY, BZ)]
        self.assertTrue(any("shallow" in n for n in names))
        self.assertTrue(any("mid"     in n for n in names))
        self.assertTrue(any("deep"    in n for n in names))

    def test_instantiate_and_sample_valid(self):
        for t in self.gripper.grasp_box_top(BX, BY, BZ):
            pose = t.instantiate(np.eye(4)).sample()
            self.assertEqual(pose.shape, (4, 4))
            np.testing.assert_allclose(pose[3, :], [0, 0, 0, 1], atol=1e-10)
            _check_se3(self, pose[:3, :3])


class TestParallelJawGripperBoxBottom(unittest.TestCase):

    def setUp(self):
        self.gripper = ParallelJawGripper(finger_length=0.055, max_aperture=0.140)

    def test_default_returns_k_templates(self):
        self.assertEqual(len(self.gripper.grasp_box_bottom(BX, BY, BZ)), 3)

    def test_k1_returns_one_template(self):
        self.assertEqual(len(self.gripper.grasp_box_bottom(BX, BY, BZ, k=1)), 1)

    def test_preshape_exceeds_aperture_raises(self):
        with self.assertRaises(ValueError):
            self.gripper.grasp_box_bottom(BX, BY, BZ,
                                          preshape=self.gripper.max_aperture + 0.01)

    def test_zero_dim_raises(self):
        with self.assertRaises(ValueError):
            self.gripper.grasp_box_bottom(0.0, BY, BZ)

    def test_bw_shape_no_rotation(self):
        for t in self.gripper.grasp_box_bottom(BX, BY, BZ):
            self.assertEqual(t.Bw.shape, (6, 2))
            for row in (3, 4, 5):
                self.assertEqual(t.Bw[row, 0], t.Bw[row, 1])

    def test_tsr_origin_at_z_zero(self):
        for t in self.gripper.grasp_box_bottom(BX, BY, BZ):
            self.assertAlmostEqual(t.T_ref_tsr[2, 3], 0.0)

    def test_z_axis_points_up(self):
        for t in self.gripper.grasp_box_bottom(BX, BY, BZ):
            np.testing.assert_allclose(t.Tw_e[:3, 2], [0., 0., 1.], atol=1e-10)

    def test_tw_e_is_valid_se3(self):
        for t in self.gripper.grasp_box_bottom(BX, BY, BZ):
            _check_se3(self, t.Tw_e[:3, :3])

    def test_palm_offset_decreases_with_depth(self):
        z_offsets = [abs(t.Tw_e[2, 3])
                     for t in self.gripper.grasp_box_bottom(BX, BY, BZ, k=3)]
        self.assertEqual(sorted(z_offsets, reverse=True), z_offsets)

    def test_instantiate_and_sample_valid(self):
        for t in self.gripper.grasp_box_bottom(BX, BY, BZ):
            pose = t.instantiate(np.eye(4)).sample()
            _check_se3(self, pose[:3, :3])


class TestParallelJawGripperBoxFaceX(unittest.TestCase):

    def setUp(self):
        self.gripper = ParallelJawGripper(finger_length=0.055, max_aperture=0.140)

    def test_default_returns_2k_templates(self):
        self.assertEqual(len(self.gripper.grasp_box_face_x(BX, BY, BZ)), 6)

    def test_k1_returns_two_templates(self):
        self.assertEqual(len(self.gripper.grasp_box_face_x(BX, BY, BZ, k=1)), 2)

    def test_k2_returns_four_templates(self):
        self.assertEqual(len(self.gripper.grasp_box_face_x(BX, BY, BZ, k=2)), 4)

    def test_preshape_exceeds_aperture_raises(self):
        with self.assertRaises(ValueError):
            self.gripper.grasp_box_face_x(BX, BY, BZ,
                                          preshape=self.gripper.max_aperture + 0.01)

    def test_zero_dim_raises(self):
        with self.assertRaises(ValueError):
            self.gripper.grasp_box_face_x(0.0, BY, BZ)

    def test_bw_shape_no_rotation(self):
        for t in self.gripper.grasp_box_face_x(BX, BY, BZ):
            self.assertEqual(t.Bw.shape, (6, 2))
            for row in (3, 4, 5):
                self.assertEqual(t.Bw[row, 0], t.Bw[row, 1])

    def test_bw_y_slides_z_slides_x_fixed(self):
        clearance = 0.1 * self.gripper.finger_length
        for t in self.gripper.grasp_box_face_x(BX, BY, BZ):
            # x fixed (no radial freedom)
            self.assertEqual(t.Bw[0, 0], t.Bw[0, 1])
            # y slides within face
            self.assertAlmostEqual(t.Bw[1, 0], -(BY / 2 - clearance))
            self.assertAlmostEqual(t.Bw[1, 1],  (BY / 2 - clearance))
            # z slides within graspable band
            self.assertLess(t.Bw[2, 0], 0.)
            self.assertGreater(t.Bw[2, 1], 0.)

    def test_tw_e_is_valid_se3(self):
        for t in self.gripper.grasp_box_face_x(BX, BY, BZ):
            _check_se3(self, t.Tw_e[:3, :3])

    def test_approach_directions(self):
        templates = self.gripper.grasp_box_face_x(BX, BY, BZ, k=1)
        # First template: +x face, z_EE should be [-1, 0, 0]
        np.testing.assert_allclose(templates[0].Tw_e[:3, 2], [-1., 0., 0.], atol=1e-10)
        # Second template: -x face, z_EE should be [+1, 0, 0]
        np.testing.assert_allclose(templates[1].Tw_e[:3, 2], [+1., 0., 0.], atol=1e-10)

    def test_names_contain_face_and_depth_labels(self):
        names = [t.name for t in self.gripper.grasp_box_face_x(BX, BY, BZ)]
        self.assertTrue(any("+x" in n for n in names))
        self.assertTrue(any("-x" in n for n in names))
        self.assertTrue(any("shallow" in n for n in names))
        self.assertTrue(any("deep"    in n for n in names))

    def test_instantiate_and_sample_valid(self):
        for t in self.gripper.grasp_box_face_x(BX, BY, BZ):
            pose = t.instantiate(np.eye(4)).sample()
            _check_se3(self, pose[:3, :3])


class TestParallelJawGripperBoxFaceY(unittest.TestCase):

    def setUp(self):
        self.gripper = ParallelJawGripper(finger_length=0.055, max_aperture=0.140)

    def test_default_returns_2k_templates(self):
        self.assertEqual(len(self.gripper.grasp_box_face_y(BX, BY, BZ)), 6)

    def test_k1_returns_two_templates(self):
        self.assertEqual(len(self.gripper.grasp_box_face_y(BX, BY, BZ, k=1)), 2)

    def test_preshape_exceeds_aperture_raises(self):
        with self.assertRaises(ValueError):
            self.gripper.grasp_box_face_y(BX, BY, BZ,
                                          preshape=self.gripper.max_aperture + 0.01)

    def test_zero_dim_raises(self):
        with self.assertRaises(ValueError):
            self.gripper.grasp_box_face_y(BX, 0.0, BZ)

    def test_bw_shape_no_rotation(self):
        for t in self.gripper.grasp_box_face_y(BX, BY, BZ):
            self.assertEqual(t.Bw.shape, (6, 2))
            for row in (3, 4, 5):
                self.assertEqual(t.Bw[row, 0], t.Bw[row, 1])

    def test_bw_x_slides_z_slides_y_fixed(self):
        clearance = 0.1 * self.gripper.finger_length
        for t in self.gripper.grasp_box_face_y(BX, BY, BZ):
            # x slides within face
            self.assertAlmostEqual(t.Bw[0, 0], -(BX / 2 - clearance))
            self.assertAlmostEqual(t.Bw[0, 1],  (BX / 2 - clearance))
            # y fixed (no radial freedom)
            self.assertEqual(t.Bw[1, 0], t.Bw[1, 1])
            # z slides within graspable band
            self.assertLess(t.Bw[2, 0], 0.)
            self.assertGreater(t.Bw[2, 1], 0.)

    def test_tw_e_is_valid_se3(self):
        for t in self.gripper.grasp_box_face_y(BX, BY, BZ):
            _check_se3(self, t.Tw_e[:3, :3])

    def test_approach_directions(self):
        templates = self.gripper.grasp_box_face_y(BX, BY, BZ, k=1)
        # First template: +y face, z_EE should be [0, -1, 0]
        np.testing.assert_allclose(templates[0].Tw_e[:3, 2], [0., -1., 0.], atol=1e-10)
        # Second template: -y face, z_EE should be [0, +1, 0]
        np.testing.assert_allclose(templates[1].Tw_e[:3, 2], [0., +1., 0.], atol=1e-10)

    def test_names_contain_face_and_depth_labels(self):
        names = [t.name for t in self.gripper.grasp_box_face_y(BX, BY, BZ)]
        self.assertTrue(any("+y" in n for n in names))
        self.assertTrue(any("-y" in n for n in names))
        self.assertTrue(any("shallow" in n for n in names))
        self.assertTrue(any("deep"    in n for n in names))

    def test_instantiate_and_sample_valid(self):
        for t in self.gripper.grasp_box_face_y(BX, BY, BZ):
            pose = t.instantiate(np.eye(4)).sample()
            _check_se3(self, pose[:3, :3])


class TestParallelJawGripperBoxCombined(unittest.TestCase):

    def setUp(self):
        self.gripper = ParallelJawGripper(finger_length=0.055, max_aperture=0.140)

    def test_default_returns_6k_templates(self):
        # 2*k face_x + 2*k face_y + k top + k bottom = 6*k
        self.assertEqual(len(self.gripper.grasp_box(BX, BY, BZ)), 18)

    def test_k1_returns_six_templates(self):
        self.assertEqual(len(self.gripper.grasp_box(BX, BY, BZ, k=1)), 6)

    def test_all_templates_are_tsrtemplate(self):
        for t in self.gripper.grasp_box(BX, BY, BZ):
            self.assertIsInstance(t, TSRTemplate)

    def test_all_tw_e_are_valid_se3(self):
        for t in self.gripper.grasp_box(BX, BY, BZ):
            _check_se3(self, t.Tw_e[:3, :3])

    def test_instantiate_and_sample_all_valid(self):
        for t in self.gripper.grasp_box(BX, BY, BZ):
            pose = t.instantiate(np.eye(4)).sample()
            _check_se3(self, pose[:3, :3])

    def test_task_subject_reference(self):
        templates = self.gripper.grasp_box(BX, BY, BZ, subject="ee", reference="cereal")
        for t in templates:
            self.assertEqual(t.task, "grasp")
            self.assertEqual(t.subject, "ee")
            self.assertEqual(t.reference, "cereal")
