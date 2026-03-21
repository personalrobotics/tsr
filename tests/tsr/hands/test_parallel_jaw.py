"""Tests for ParallelJawGripper."""
import unittest
import numpy as np
from numpy import pi

from tsr.hands import ParallelJawGripper
from tsr.template import TSRTemplate

R = 0.040   # cylinder radius
H = 0.120   # cylinder height


class TestParallelJawGripperCylinderSide(unittest.TestCase):

    def setUp(self):
        self.gripper = ParallelJawGripper(finger_length=0.055, max_aperture=0.140)

    def test_default_returns_six_templates(self):
        self.assertEqual(len(self.gripper.grasp_cylinder_side(R, H)), 6)

    def test_k1_returns_two_templates(self):
        self.assertEqual(len(self.gripper.grasp_cylinder_side(R, H, k=1)), 2)

    def test_k2_returns_four_templates(self):
        self.assertEqual(len(self.gripper.grasp_cylinder_side(R, H, k=2)), 4)

    def test_preshape_too_small_returns_empty(self):
        self.assertEqual(
            self.gripper.grasp_cylinder_side(R, H, preshape=2 * R - 0.001), [])

    def test_preshape_exceeds_aperture_raises(self):
        with self.assertRaises(ValueError):
            self.gripper.grasp_cylinder_side(R, H,
                                             preshape=self.gripper.max_aperture + 0.01)

    def test_zero_radius_raises(self):
        with self.assertRaises(ValueError):
            self.gripper.grasp_cylinder_side(0.0, H)

    def test_narrow_height_raises(self):
        with self.assertRaises(ValueError):
            self.gripper.grasp_cylinder_side(R, 0.001)

    def test_default_preshape_is_2r_plus_clearance(self):
        clearance = 0.3 * min(self.gripper.finger_length, R)
        expected  = 2 * R + clearance
        templates = self.gripper.grasp_cylinder_side(R, H)
        np.testing.assert_allclose(templates[0].preshape[0], expected)

    def test_all_templates_are_tsrtemplate(self):
        for t in self.gripper.grasp_cylinder_side(R, H):
            self.assertIsInstance(t, TSRTemplate)

    def test_template_task_subject_reference(self):
        templates = self.gripper.grasp_cylinder_side(R, H, subject="ee", reference="mug")
        for t in templates:
            self.assertEqual(t.task, "grasp")
            self.assertEqual(t.subject, "ee")
            self.assertEqual(t.reference, "mug")

    def test_bw_shape_and_yaw_free(self):
        for t in self.gripper.grasp_cylinder_side(R, H):
            self.assertEqual(t.Bw.shape, (6, 2))
            self.assertAlmostEqual(t.Bw[5, 0], 0.)
            self.assertAlmostEqual(t.Bw[5, 1], 2 * pi)
            for row in (0, 1, 3, 4):
                self.assertEqual(t.Bw[row, 0], t.Bw[row, 1])

    def test_tw_e_is_valid_se3(self):
        for t in self.gripper.grasp_cylinder_side(R, H):
            Rot = t.Tw_e[:3, :3]
            np.testing.assert_allclose(Rot @ Rot.T, np.eye(3), atol=1e-10)
            np.testing.assert_allclose(np.linalg.det(Rot), 1.0, atol=1e-10)

    def test_names_contain_depth_and_roll_labels(self):
        names = [t.name for t in self.gripper.grasp_cylinder_side(R, H)]
        self.assertTrue(any("shallow" in n for n in names))
        self.assertTrue(any("mid"     in n for n in names))
        self.assertTrue(any("deep"    in n for n in names))
        self.assertTrue(any("roll 0°"   in n for n in names))
        self.assertTrue(any("roll 180°" in n for n in names))

    def test_instantiate_and_sample_produces_valid_pose(self):
        object_pose = np.eye(4)
        object_pose[:3, 3] = [0.5, 0.0, 0.0]
        for t in self.gripper.grasp_cylinder_side(R, H):
            pose = t.instantiate(object_pose).sample()
            self.assertEqual(pose.shape, (4, 4))
            np.testing.assert_allclose(pose[3, :], [0, 0, 0, 1], atol=1e-10)
            Rot = pose[:3, :3]
            np.testing.assert_allclose(Rot @ Rot.T, np.eye(3), atol=1e-8)

    def test_renderer_returns_callable(self):
        try:
            self.assertTrue(callable(self.gripper.renderer()))
        except ImportError:
            pass


class TestParallelJawGripperCylinderTop(unittest.TestCase):

    def setUp(self):
        self.gripper = ParallelJawGripper(finger_length=0.055, max_aperture=0.140)

    def test_default_returns_k_templates(self):
        self.assertEqual(len(self.gripper.grasp_cylinder_top(R, H)), 3)

    def test_k1_returns_one_template(self):
        self.assertEqual(len(self.gripper.grasp_cylinder_top(R, H, k=1)), 1)

    def test_k2_returns_two_templates(self):
        self.assertEqual(len(self.gripper.grasp_cylinder_top(R, H, k=2)), 2)

    def test_preshape_too_small_returns_empty(self):
        self.assertEqual(
            self.gripper.grasp_cylinder_top(R, H, preshape=2 * R - 0.001), [])

    def test_preshape_exceeds_aperture_raises(self):
        with self.assertRaises(ValueError):
            self.gripper.grasp_cylinder_top(R, H,
                                            preshape=self.gripper.max_aperture + 0.01)

    def test_zero_radius_raises(self):
        with self.assertRaises(ValueError):
            self.gripper.grasp_cylinder_top(0.0, H)

    def test_bw_shape_and_yaw_free(self):
        for t in self.gripper.grasp_cylinder_top(R, H):
            self.assertEqual(t.Bw.shape, (6, 2))
            self.assertAlmostEqual(t.Bw[5, 0], 0.)
            self.assertAlmostEqual(t.Bw[5, 1], 2 * pi)
            for row in (0, 1, 2, 3, 4):
                self.assertEqual(t.Bw[row, 0], t.Bw[row, 1])

    def test_tw_e_is_valid_se3(self):
        for t in self.gripper.grasp_cylinder_top(R, H):
            Rot = t.Tw_e[:3, :3]
            np.testing.assert_allclose(Rot @ Rot.T, np.eye(3), atol=1e-10)
            np.testing.assert_allclose(np.linalg.det(Rot), 1.0, atol=1e-10)

    def test_z_axis_points_down(self):
        for t in self.gripper.grasp_cylinder_top(R, H):
            np.testing.assert_allclose(t.Tw_e[:3, 2], [0., 0., -1.], atol=1e-10)

    def test_tsr_origin_at_cylinder_top(self):
        for t in self.gripper.grasp_cylinder_top(R, H):
            self.assertAlmostEqual(t.T_ref_tsr[2, 3], H)

    def test_palm_offset_decreases_with_depth(self):
        z_offsets = [t.Tw_e[2, 3] for t in self.gripper.grasp_cylinder_top(R, H, k=3)]
        self.assertEqual(sorted(z_offsets, reverse=True), z_offsets)

    def test_names_contain_depth_labels(self):
        names = [t.name for t in self.gripper.grasp_cylinder_top(R, H)]
        self.assertTrue(any("shallow" in n for n in names))
        self.assertTrue(any("mid"     in n for n in names))
        self.assertTrue(any("deep"    in n for n in names))

    def test_instantiate_and_sample_produces_valid_pose(self):
        for t in self.gripper.grasp_cylinder_top(R, H):
            pose = t.instantiate(np.eye(4)).sample()
            self.assertEqual(pose.shape, (4, 4))
            np.testing.assert_allclose(pose[3, :], [0, 0, 0, 1], atol=1e-10)
            Rot = pose[:3, :3]
            np.testing.assert_allclose(Rot @ Rot.T, np.eye(3), atol=1e-8)


class TestParallelJawGripperCylinderBottom(unittest.TestCase):

    def setUp(self):
        self.gripper = ParallelJawGripper(finger_length=0.055, max_aperture=0.140)

    def test_default_returns_k_templates(self):
        self.assertEqual(len(self.gripper.grasp_cylinder_bottom(R, H)), 3)

    def test_k1_returns_one_template(self):
        self.assertEqual(len(self.gripper.grasp_cylinder_bottom(R, H, k=1)), 1)

    def test_preshape_too_small_returns_empty(self):
        self.assertEqual(
            self.gripper.grasp_cylinder_bottom(R, H, preshape=2 * R - 0.001), [])

    def test_preshape_exceeds_aperture_raises(self):
        with self.assertRaises(ValueError):
            self.gripper.grasp_cylinder_bottom(R, H,
                                               preshape=self.gripper.max_aperture + 0.01)

    def test_zero_radius_raises(self):
        with self.assertRaises(ValueError):
            self.gripper.grasp_cylinder_bottom(0.0, H)

    def test_bw_shape_and_yaw_free(self):
        for t in self.gripper.grasp_cylinder_bottom(R, H):
            self.assertEqual(t.Bw.shape, (6, 2))
            self.assertAlmostEqual(t.Bw[5, 0], 0.)
            self.assertAlmostEqual(t.Bw[5, 1], 2 * pi)
            for row in (0, 1, 2, 3, 4):
                self.assertEqual(t.Bw[row, 0], t.Bw[row, 1])

    def test_tw_e_is_valid_se3(self):
        for t in self.gripper.grasp_cylinder_bottom(R, H):
            Rot = t.Tw_e[:3, :3]
            np.testing.assert_allclose(Rot @ Rot.T, np.eye(3), atol=1e-10)
            np.testing.assert_allclose(np.linalg.det(Rot), 1.0, atol=1e-10)

    def test_z_axis_points_up(self):
        for t in self.gripper.grasp_cylinder_bottom(R, H):
            np.testing.assert_allclose(t.Tw_e[:3, 2], [0., 0., 1.], atol=1e-10)

    def test_tsr_origin_at_z_zero(self):
        for t in self.gripper.grasp_cylinder_bottom(R, H):
            self.assertAlmostEqual(t.T_ref_tsr[2, 3], 0.0)

    def test_palm_offset_decreases_with_depth(self):
        z_offsets = [abs(t.Tw_e[2, 3]) for t in self.gripper.grasp_cylinder_bottom(R, H, k=3)]
        self.assertEqual(sorted(z_offsets, reverse=True), z_offsets)

    def test_names_contain_depth_labels(self):
        names = [t.name for t in self.gripper.grasp_cylinder_bottom(R, H)]
        self.assertTrue(any("shallow" in n for n in names))
        self.assertTrue(any("mid"     in n for n in names))
        self.assertTrue(any("deep"    in n for n in names))

    def test_instantiate_and_sample_produces_valid_pose(self):
        for t in self.gripper.grasp_cylinder_bottom(R, H):
            pose = t.instantiate(np.eye(4)).sample()
            self.assertEqual(pose.shape, (4, 4))
            np.testing.assert_allclose(pose[3, :], [0, 0, 0, 1], atol=1e-10)
            Rot = pose[:3, :3]
            np.testing.assert_allclose(Rot @ Rot.T, np.eye(3), atol=1e-8)


class TestParallelJawGripperCylinderCombined(unittest.TestCase):

    def setUp(self):
        self.gripper = ParallelJawGripper(finger_length=0.055, max_aperture=0.140)

    def test_default_returns_4k_templates(self):
        # 2*k side + k top + k bottom = 4*k
        self.assertEqual(len(self.gripper.grasp_cylinder(R, H)), 12)

    def test_k1_returns_four_templates(self):
        self.assertEqual(len(self.gripper.grasp_cylinder(R, H, k=1)), 4)

    def test_all_templates_are_tsrtemplate(self):
        for t in self.gripper.grasp_cylinder(R, H):
            self.assertIsInstance(t, TSRTemplate)

    def test_preshape_too_small_returns_empty(self):
        self.assertEqual(
            self.gripper.grasp_cylinder(R, H, preshape=2 * R - 0.001), [])

    def test_instantiate_and_sample_all_valid(self):
        for t in self.gripper.grasp_cylinder(R, H):
            pose = t.instantiate(np.eye(4)).sample()
            Rot = pose[:3, :3]
            np.testing.assert_allclose(Rot @ Rot.T, np.eye(3), atol=1e-8)


class TestParallelJawGripperAngleRange(unittest.TestCase):

    def test_restricted_angle_range(self):
        gripper = ParallelJawGripper(finger_length=0.055, max_aperture=0.140)
        for t in gripper.grasp_cylinder_side(R, H, angle_range=(0., np.pi)):
            self.assertAlmostEqual(t.Bw[5, 0], 0.)
            self.assertAlmostEqual(t.Bw[5, 1], np.pi)
