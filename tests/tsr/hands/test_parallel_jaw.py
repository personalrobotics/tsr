"""Tests for ParallelJawGripper."""
import unittest
import numpy as np
from numpy import pi

from tsr.hands import ParallelJawGripper
from tsr.template import TSRTemplate


class TestParallelJawGripperCylinder(unittest.TestCase):

    def setUp(self):
        self.gripper = ParallelJawGripper(finger_length=0.055, max_aperture=0.140)
        self.radius  = 0.040
        self.height  = (0.02, 0.10)

    def test_default_returns_six_templates(self):
        templates = self.gripper.grasp_cylinder(self.radius, self.height)
        self.assertEqual(len(templates), 6)  # k=3 × 2 rolls

    def test_k1_returns_two_templates(self):
        templates = self.gripper.grasp_cylinder(self.radius, self.height, k=1)
        self.assertEqual(len(templates), 2)

    def test_k2_returns_four_templates(self):
        templates = self.gripper.grasp_cylinder(self.radius, self.height, k=2)
        self.assertEqual(len(templates), 4)

    def test_preshape_too_small_returns_empty(self):
        # preshape <= 2 * radius → can't span the cylinder
        templates = self.gripper.grasp_cylinder(self.radius, self.height,
                                                preshape=2 * self.radius - 0.001)
        self.assertEqual(templates, [])

    def test_preshape_exceeds_aperture_raises(self):
        with self.assertRaises(ValueError):
            self.gripper.grasp_cylinder(self.radius, self.height,
                                        preshape=self.gripper.max_aperture + 0.01)

    def test_zero_radius_raises(self):
        with self.assertRaises(ValueError):
            self.gripper.grasp_cylinder(0.0, self.height)

    def test_narrow_height_raises(self):
        # clearance eats into both ends; 1mm band is too narrow
        with self.assertRaises(ValueError):
            self.gripper.grasp_cylinder(self.radius, (0.0, 0.001))

    def test_default_preshape_is_2r_plus_clearance(self):
        clearance = 0.1 * self.gripper.finger_length
        expected_preshape = 2 * self.radius + clearance
        templates = self.gripper.grasp_cylinder(self.radius, self.height)
        np.testing.assert_allclose(templates[0].preshape[0], expected_preshape)

    def test_all_templates_are_tsrtemplate(self):
        templates = self.gripper.grasp_cylinder(self.radius, self.height)
        for t in templates:
            self.assertIsInstance(t, TSRTemplate)

    def test_template_task_subject_reference(self):
        templates = self.gripper.grasp_cylinder(self.radius, self.height,
                                                subject="ee", reference="mug")
        for t in templates:
            self.assertEqual(t.task, "grasp")
            self.assertEqual(t.subject, "ee")
            self.assertEqual(t.reference, "mug")

    def test_bw_shape_and_yaw_free(self):
        templates = self.gripper.grasp_cylinder(self.radius, self.height)
        for t in templates:
            self.assertEqual(t.Bw.shape, (6, 2))
            # yaw (row 5) spans full circle
            self.assertAlmostEqual(t.Bw[5, 0], 0.)
            self.assertAlmostEqual(t.Bw[5, 1], 2 * pi)
            # x, y, roll, pitch fixed
            for row in (0, 1, 3, 4):
                self.assertEqual(t.Bw[row, 0], t.Bw[row, 1])

    def test_tw_e_is_valid_se3(self):
        templates = self.gripper.grasp_cylinder(self.radius, self.height)
        for t in templates:
            R = t.Tw_e[:3, :3]
            np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)
            np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-10)
            self.assertAlmostEqual(t.Tw_e[3, 3], 1.0)

    def test_names_contain_depth_and_roll_labels(self):
        templates = self.gripper.grasp_cylinder(self.radius, self.height)
        names = [t.name for t in templates]
        self.assertTrue(any("shallow" in n for n in names))
        self.assertTrue(any("mid"     in n for n in names))
        self.assertTrue(any("deep"    in n for n in names))
        self.assertTrue(any("roll 0°"   in n for n in names))
        self.assertTrue(any("roll 180°" in n for n in names))

    def test_instantiate_and_sample_produces_valid_pose(self):
        templates = self.gripper.grasp_cylinder(self.radius, self.height)
        object_pose = np.eye(4)
        object_pose[:3, 3] = [0.5, 0.0, 0.0]
        for t in templates:
            tsr  = t.instantiate(object_pose)
            pose = tsr.sample()
            self.assertEqual(pose.shape, (4, 4))
            np.testing.assert_allclose(pose[3, :], [0, 0, 0, 1], atol=1e-10)
            R = pose[:3, :3]
            np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-8)

    def test_renderer_returns_callable(self):
        # renderer() should return a callable without importing pyvista at this
        # point (the import happens inside the callable, not here)
        try:
            r = self.gripper.renderer()
            self.assertTrue(callable(r))
        except ImportError:
            pass  # pyvista not installed — acceptable in CI without viz extra


class TestParallelJawGripperAngleRange(unittest.TestCase):

    def test_restricted_angle_range(self):
        gripper = ParallelJawGripper(finger_length=0.055, max_aperture=0.140)
        templates = gripper.grasp_cylinder(0.040, (0.02, 0.10),
                                           angle_range=(0., np.pi))
        for t in templates:
            self.assertAlmostEqual(t.Bw[5, 0], 0.)
            self.assertAlmostEqual(t.Bw[5, 1], np.pi)


class TestParallelJawGripperCylinderTop(unittest.TestCase):

    def setUp(self):
        self.gripper = ParallelJawGripper(finger_length=0.055, max_aperture=0.140)
        self.radius  = 0.040
        self.height  = 0.120

    def test_default_returns_k_templates(self):
        templates = self.gripper.grasp_cylinder_top(self.radius, self.height)
        self.assertEqual(len(templates), 3)

    def test_k1_returns_one_template(self):
        templates = self.gripper.grasp_cylinder_top(self.radius, self.height, k=1)
        self.assertEqual(len(templates), 1)

    def test_k2_returns_two_templates(self):
        templates = self.gripper.grasp_cylinder_top(self.radius, self.height, k=2)
        self.assertEqual(len(templates), 2)

    def test_preshape_too_small_returns_empty(self):
        templates = self.gripper.grasp_cylinder_top(self.radius, self.height,
                                                    preshape=2 * self.radius - 0.001)
        self.assertEqual(templates, [])

    def test_preshape_exceeds_aperture_raises(self):
        with self.assertRaises(ValueError):
            self.gripper.grasp_cylinder_top(self.radius, self.height,
                                            preshape=self.gripper.max_aperture + 0.01)

    def test_zero_radius_raises(self):
        with self.assertRaises(ValueError):
            self.gripper.grasp_cylinder_top(0.0, self.height)

    def test_bw_shape_and_yaw_free(self):
        templates = self.gripper.grasp_cylinder_top(self.radius, self.height)
        for t in templates:
            self.assertEqual(t.Bw.shape, (6, 2))
            # yaw spans full circle
            self.assertAlmostEqual(t.Bw[5, 0], 0.)
            self.assertAlmostEqual(t.Bw[5, 1], 2 * pi)
            # x, y, z, roll, pitch fixed
            for row in (0, 1, 2, 3, 4):
                self.assertEqual(t.Bw[row, 0], t.Bw[row, 1])

    def test_tw_e_is_valid_se3(self):
        templates = self.gripper.grasp_cylinder_top(self.radius, self.height)
        for t in templates:
            R = t.Tw_e[:3, :3]
            np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)
            np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-10)

    def test_z_axis_points_down(self):
        """z_EE should point in -z direction (approach from above)."""
        templates = self.gripper.grasp_cylinder_top(self.radius, self.height)
        for t in templates:
            z_col = t.Tw_e[:3, 2]  # third column = z_EE in reference frame
            np.testing.assert_allclose(z_col, [0., 0., -1.], atol=1e-10)

    def test_tsr_origin_at_cylinder_top(self):
        templates = self.gripper.grasp_cylinder_top(self.radius, self.height)
        for t in templates:
            self.assertAlmostEqual(t.T_ref_tsr[2, 3], self.height)

    def test_palm_above_rim_decreases_with_depth(self):
        templates = self.gripper.grasp_cylinder_top(self.radius, self.height, k=3)
        # deeper template → smaller palm height above rim
        z_offsets = [t.Tw_e[2, 3] for t in templates]
        self.assertEqual(sorted(z_offsets, reverse=True), z_offsets)

    def test_names_contain_depth_labels(self):
        templates = self.gripper.grasp_cylinder_top(self.radius, self.height)
        names = [t.name for t in templates]
        self.assertTrue(any("shallow" in n for n in names))
        self.assertTrue(any("mid"     in n for n in names))
        self.assertTrue(any("deep"    in n for n in names))

    def test_instantiate_and_sample_produces_valid_pose(self):
        templates = self.gripper.grasp_cylinder_top(self.radius, self.height)
        object_pose = np.eye(4)
        for t in templates:
            tsr  = t.instantiate(object_pose)
            pose = tsr.sample()
            self.assertEqual(pose.shape, (4, 4))
            np.testing.assert_allclose(pose[3, :], [0, 0, 0, 1], atol=1e-10)
            R = pose[:3, :3]
            np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-8)


class TestParallelJawGripperCylinderBottom(unittest.TestCase):

    def setUp(self):
        self.gripper = ParallelJawGripper(finger_length=0.055, max_aperture=0.140)
        self.radius  = 0.040
        self.bottom  = 0.0

    def test_default_returns_k_templates(self):
        templates = self.gripper.grasp_cylinder_bottom(self.radius, self.bottom)
        self.assertEqual(len(templates), 3)

    def test_k1_returns_one_template(self):
        templates = self.gripper.grasp_cylinder_bottom(self.radius, self.bottom, k=1)
        self.assertEqual(len(templates), 1)

    def test_preshape_too_small_returns_empty(self):
        templates = self.gripper.grasp_cylinder_bottom(self.radius, self.bottom,
                                                       preshape=2 * self.radius - 0.001)
        self.assertEqual(templates, [])

    def test_preshape_exceeds_aperture_raises(self):
        with self.assertRaises(ValueError):
            self.gripper.grasp_cylinder_bottom(self.radius, self.bottom,
                                               preshape=self.gripper.max_aperture + 0.01)

    def test_zero_radius_raises(self):
        with self.assertRaises(ValueError):
            self.gripper.grasp_cylinder_bottom(0.0, self.bottom)

    def test_bw_shape_and_yaw_free(self):
        templates = self.gripper.grasp_cylinder_bottom(self.radius, self.bottom)
        for t in templates:
            self.assertEqual(t.Bw.shape, (6, 2))
            self.assertAlmostEqual(t.Bw[5, 0], 0.)
            self.assertAlmostEqual(t.Bw[5, 1], 2 * pi)
            for row in (0, 1, 2, 3, 4):
                self.assertEqual(t.Bw[row, 0], t.Bw[row, 1])

    def test_tw_e_is_valid_se3(self):
        templates = self.gripper.grasp_cylinder_bottom(self.radius, self.bottom)
        for t in templates:
            R = t.Tw_e[:3, :3]
            np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)
            np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-10)

    def test_z_axis_points_up(self):
        """z_EE should point in +z direction (approach from below)."""
        templates = self.gripper.grasp_cylinder_bottom(self.radius, self.bottom)
        for t in templates:
            z_col = t.Tw_e[:3, 2]
            np.testing.assert_allclose(z_col, [0., 0., 1.], atol=1e-10)

    def test_tsr_origin_at_cylinder_bottom(self):
        templates = self.gripper.grasp_cylinder_bottom(self.radius, cylinder_bottom=0.05)
        for t in templates:
            self.assertAlmostEqual(t.T_ref_tsr[2, 3], 0.05)

    def test_palm_below_rim_decreases_with_depth(self):
        templates = self.gripper.grasp_cylinder_bottom(self.radius, self.bottom, k=3)
        # deeper template → smaller |z offset| (palm closer to rim)
        z_offsets = [abs(t.Tw_e[2, 3]) for t in templates]
        self.assertEqual(sorted(z_offsets, reverse=True), z_offsets)

    def test_names_contain_depth_labels(self):
        templates = self.gripper.grasp_cylinder_bottom(self.radius, self.bottom)
        names = [t.name for t in templates]
        self.assertTrue(any("shallow" in n for n in names))
        self.assertTrue(any("mid"     in n for n in names))
        self.assertTrue(any("deep"    in n for n in names))

    def test_instantiate_and_sample_produces_valid_pose(self):
        templates = self.gripper.grasp_cylinder_bottom(self.radius, self.bottom)
        object_pose = np.eye(4)
        for t in templates:
            tsr  = t.instantiate(object_pose)
            pose = tsr.sample()
            self.assertEqual(pose.shape, (4, 4))
            np.testing.assert_allclose(pose[3, :], [0, 0, 0, 1], atol=1e-10)
            R = pose[:3, :3]
            np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-8)
