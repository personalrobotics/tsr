"""Tests for ParallelJawGripper sphere grasp primitives."""
import unittest
import numpy as np

from tsr.hands import ParallelJawGripper

RADIUS = 0.040   # sphere radius [m]
FL     = 0.055   # finger length
MA     = 0.140   # max aperture


class TestGraspSphere(unittest.TestCase):

    def setUp(self):
        self.gripper = ParallelJawGripper(finger_length=FL, max_aperture=MA)
        self.templates = self.gripper.grasp_sphere(object_radius=RADIUS)

    # ── Template count ─────────────────────────────────────────────────────

    def test_default_k3_returns_6_templates(self):
        self.assertEqual(len(self.templates), 6)  # 2 rolls × 3 depths

    def test_k1_returns_2_templates(self):
        ts = self.gripper.grasp_sphere(RADIUS, k=1)
        self.assertEqual(len(ts), 2)

    def test_k5_returns_10_templates(self):
        ts = self.gripper.grasp_sphere(RADIUS, k=5)
        self.assertEqual(len(ts), 10)

    # ── Geometry: Tw_e ────────────────────────────────────────────────────

    def test_tw_e_is_valid_se3(self):
        for t in self.templates:
            R = t.Tw_e[:3, :3]
            np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)
            np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-10)

    def test_z_ee_points_inward(self):
        """z_EE column of Tw_e should point from gripper toward sphere center."""
        for t in self.templates:
            z_ee = t.Tw_e[:3, 2]
            trans = t.Tw_e[:3, 3]
            # standoff is positive (gripper outside sphere), z_ee antiparallel to trans
            ro = np.linalg.norm(trans)
            self.assertGreater(ro, 0)
            np.testing.assert_allclose(trans / ro, -z_ee, atol=1e-10,
                                       err_msg=f"z_EE not inward in {t.name}")

    def test_standoff_within_expected_range(self):
        clearance = 0.1 * FL
        # shallowest: ro = r + fl - clearance
        # deepest:    ro = r + fl - (min(fl, r) - clearance) = r + fl - min(fl,r) + clearance
        ro_max = RADIUS + FL - clearance
        ro_min = RADIUS + FL - (min(FL, RADIUS) - clearance)
        for t in self.templates:
            ro = np.linalg.norm(t.Tw_e[:3, 3])
            self.assertGreaterEqual(ro, ro_min - 1e-9)
            self.assertLessEqual(ro, ro_max + 1e-9)

    # ── Geometry: Bw ─────────────────────────────────────────────────────

    def test_bw_no_translational_dof(self):
        """Sphere grasp has no positional freedom — all Bw trans rows fixed."""
        for t in self.templates:
            for row in range(3):
                self.assertEqual(t.Bw[row, 0], t.Bw[row, 1],
                                 msg=f"Bw row {row} not fixed in {t.name}")

    def test_bw_full_yaw_freedom(self):
        for t in self.templates:
            self.assertAlmostEqual(t.Bw[5, 0], 0.)
            self.assertAlmostEqual(t.Bw[5, 1], 2 * np.pi)

    def test_bw_no_roll_or_pitch_freedom(self):
        for t in self.templates:
            self.assertEqual(t.Bw[3, 0], t.Bw[3, 1])
            self.assertEqual(t.Bw[4, 0], t.Bw[4, 1])

    # ── Preshape ──────────────────────────────────────────────────────────

    def test_default_preshape_is_diameter_plus_clearance(self):
        clearance = 0.1 * FL
        expected = 2 * RADIUS + clearance
        for t in self.templates:
            self.assertAlmostEqual(t.preshape[0], expected)

    def test_custom_preshape_propagates(self):
        ps = 0.095
        ts = self.gripper.grasp_sphere(RADIUS, preshape=ps)
        for t in ts:
            self.assertAlmostEqual(t.preshape[0], ps)

    # ── Filtering / errors ────────────────────────────────────────────────

    def test_returns_empty_when_preshape_too_small(self):
        # preshape <= 2*r → can't straddle
        ts = self.gripper.grasp_sphere(RADIUS, preshape=2 * RADIUS - 0.001)
        self.assertEqual(ts, [])

    def test_raises_when_preshape_exceeds_max_aperture(self):
        with self.assertRaises(ValueError):
            self.gripper.grasp_sphere(RADIUS, preshape=MA + 0.001)

    def test_raises_for_nonpositive_radius(self):
        with self.assertRaises(ValueError):
            self.gripper.grasp_sphere(0.)

    def test_large_sphere_raises(self):
        # auto-computed preshape = 2r + clearance > max_aperture → ValueError
        # (same behavior as grasp_cylinder_side for oversized objects)
        with self.assertRaises(ValueError):
            self.gripper.grasp_sphere(object_radius=MA)  # 2r = 0.28 > 0.14

    # ── TSR origin ────────────────────────────────────────────────────────

    def test_tsr_origin_at_sphere_center(self):
        """T_ref_tsr should be identity (origin at sphere center)."""
        for t in self.templates:
            np.testing.assert_allclose(t.T_ref_tsr, np.eye(4), atol=1e-10)

    # ── angle_range ───────────────────────────────────────────────────────

    def test_custom_angle_range(self):
        ts = self.gripper.grasp_sphere(RADIUS, angle_range=(0., np.pi))
        for t in ts:
            self.assertAlmostEqual(t.Bw[5, 0], 0.)
            self.assertAlmostEqual(t.Bw[5, 1], np.pi)


if __name__ == "__main__":
    unittest.main()
