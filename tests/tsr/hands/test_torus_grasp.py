"""Tests for ParallelJawGripper torus grasp primitives.

Torus fixtures:
  SMALL torus: R=0.04, r=0.015 → outer=0.055m, span preshape≈0.116m < MA=0.14 → span included
  LARGE torus: R=0.06, r=0.025 → outer=0.085m, span preshape≈0.175m > MA=0.14 → span excluded
"""
import unittest
import numpy as np

from tsr.hands import ParallelJawGripper

FL = 0.055   # finger length
MA = 0.140   # max aperture

# Small torus: span fits (2*(R+r)+clearance = 2*0.055+0.0165 = 0.1265 < 0.14)
SR, Sr = 0.040, 0.015

# Large torus: span doesn't fit (2*(R+r)+clearance = 2*0.085+0.0165 = 0.1865 > 0.14)
LR, Lr = 0.060, 0.025


def _check_se3(tc, R):
    tc.assertEqual(R.shape, (3, 3))
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)
    np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-10)


class TestGraspTorusSide(unittest.TestCase):

    def setUp(self):
        self.gripper = ParallelJawGripper(finger_length=FL, max_aperture=MA)
        self.templates = self.gripper.grasp_torus_side(SR, Sr)

    # ── Template count ─────────────────────────────────────────────────────

    def test_default_k3_n5_returns_30_templates(self):
        # 5 minor angles × 3 depths × 2 hand flips = 30
        self.assertEqual(len(self.templates), 30)

    def test_k1_n1_returns_2_templates(self):
        # equatorial only (n_minor=1), single depth, 2 hand flips
        ts = self.gripper.grasp_torus_side(SR, Sr, k=1, n_minor=1)
        self.assertEqual(len(ts), 2)

    def test_k1_n3_returns_6_templates(self):
        ts = self.gripper.grasp_torus_side(SR, Sr, k=1, n_minor=3)
        self.assertEqual(len(ts), 6)   # 3 angles × 1 depth × 2 flips

    def test_k3_n1_returns_6_templates(self):
        ts = self.gripper.grasp_torus_side(SR, Sr, k=3, n_minor=1)
        self.assertEqual(len(ts), 6)   # 1 angle × 3 depths × 2 flips

    # ── Geometry: Tw_e ────────────────────────────────────────────────────

    def test_tw_e_valid_se3(self):
        for t in self.templates:
            _check_se3(self, t.Tw_e[:3, :3])

    def test_z_ee_in_xz_plane_of_tsr_frame(self):
        """z_EE = (−cosα, 0, −sinα): y-component always zero."""
        for t in self.templates:
            self.assertAlmostEqual(t.Tw_e[1, 2], 0.,
                                   msg=f"z_EE y-component nonzero in {t.name}")

    def test_z_ee_equatorial_is_minus_x(self):
        """Middle minor angle of n_minor=3 is α=0: z_EE = (−1, 0, 0)."""
        ts = self.gripper.grasp_torus_side(SR, Sr, k=1, n_minor=3)
        # Templates order: α=−π/2 (idx 0,1), α=0 (idx 2,3), α=+π/2 (idx 4,5)
        for t in ts[2:4]:
            np.testing.assert_allclose(t.Tw_e[:3, 2], [-1., 0., 0.], atol=1e-10)

    def test_z_ee_at_alpha_pos_pi2_points_down(self):
        """Last minor angle is +π/2: z_EE = (0, 0, −1) (approach from above)."""
        ts = self.gripper.grasp_torus_side(SR, Sr, k=1, n_minor=5)
        for t in ts[-2:]:   # last 2 = α=+π/2, both hand flips
            np.testing.assert_allclose(t.Tw_e[:3, 2], [0., 0., -1.], atol=1e-10)

    def test_z_ee_at_alpha_neg_pi2_points_up(self):
        """First minor angle is −π/2: z_EE = (0, 0, +1) (approach from below)."""
        ts = self.gripper.grasp_torus_side(SR, Sr, k=1, n_minor=5)
        for t in ts[:2]:    # first 2 = α=−π/2, both hand flips
            np.testing.assert_allclose(t.Tw_e[:3, 2], [0., 0., 1.], atol=1e-10)

    def test_gripper_outside_tube_surface(self):
        """Distance from tube center to gripper must exceed tube_radius."""
        for t in self.templates:
            tx, tz = t.Tw_e[0, 3], t.Tw_e[2, 3]
            # Tube center in TSR frame is at (R, 0, 0)
            dist_from_tube_center = np.sqrt((tx - SR) ** 2 + tz ** 2)
            self.assertGreater(dist_from_tube_center, Sr,
                               msg=f"Gripper inside tube surface in {t.name}")

    def test_y_ee_in_radial_vertical_plane(self):
        """y_EE (finger opening) lies in the radial-vertical plane: y-component = 0."""
        for t in self.templates:
            self.assertAlmostEqual(t.Tw_e[1, 1], 0.,
                                   msg=f"y_EE y-component nonzero in {t.name}")

    # ── Geometry: Bw ─────────────────────────────────────────────────────

    def test_bw_no_translational_dof(self):
        for t in self.templates:
            for row in range(3):
                self.assertEqual(t.Bw[row, 0], t.Bw[row, 1])

    def test_bw_full_yaw_freedom(self):
        for t in self.templates:
            self.assertAlmostEqual(t.Bw[5, 0], 0.)
            self.assertAlmostEqual(t.Bw[5, 1], 2 * np.pi)

    def test_bw_no_roll_pitch_freedom(self):
        for t in self.templates:
            self.assertEqual(t.Bw[3, 0], t.Bw[3, 1])
            self.assertEqual(t.Bw[4, 0], t.Bw[4, 1])

    # ── Preshape ──────────────────────────────────────────────────────────

    def test_default_preshape_is_tube_diameter_plus_clearance(self):
        clearance = self.gripper.clearance_fraction * min(FL, Sr)
        expected = 2 * Sr + clearance
        for t in self.templates:
            self.assertAlmostEqual(t.preshape[0], expected)

    # ── Filtering / errors ────────────────────────────────────────────────

    def test_returns_empty_when_preshape_too_small(self):
        ts = self.gripper.grasp_torus_side(SR, Sr, preshape=2 * Sr - 0.001)
        self.assertEqual(ts, [])

    def test_raises_when_preshape_exceeds_max_aperture(self):
        with self.assertRaises(ValueError):
            self.gripper.grasp_torus_side(SR, Sr, preshape=MA + 0.001)

    def test_returns_empty_when_finger_too_short_to_reach_centerline(self):
        # finger_length must be > tube_radius to reach the tube center from outside
        short_gripper = ParallelJawGripper(finger_length=Sr - 0.001, max_aperture=MA)
        ts = short_gripper.grasp_torus_side(SR, Sr)
        self.assertEqual(ts, [])

    def test_side_depth_shallow_fingertip_at_tube_center(self):
        """Shallowest depth (d=tube_radius): fingertip exactly at tube center."""
        # k=2, n_minor=3 → α=0 block at idx 4-7; shallow=4,5
        ts = self.gripper.grasp_torus_side(SR, Sr, k=2, n_minor=3)
        for t in ts[4:6]:
            ro_minor = t.Tw_e[0, 3] - SR
            np.testing.assert_allclose(ro_minor - FL, 0., atol=1e-10)

    def test_side_depth_deep_fingertip_at_inner_surface(self):
        """Deepest depth (d=2*tube_radius): fingertip at inner tube surface."""
        # k=2, n_minor=3 → α=0 block at idx 4-7; deep=6,7
        ts = self.gripper.grasp_torus_side(SR, Sr, k=2, n_minor=3)
        for t in ts[6:8]:
            ro_minor = t.Tw_e[0, 3] - SR
            np.testing.assert_allclose(ro_minor - FL, -Sr, atol=1e-10)

    def test_raises_for_invalid_torus(self):
        with self.assertRaises(ValueError):
            self.gripper.grasp_torus_side(0.01, 0.02)  # tube_r >= torus_r

    # ── TSR origin ────────────────────────────────────────────────────────

    def test_tsr_origin_at_torus_center(self):
        for t in self.templates:
            np.testing.assert_allclose(t.T_ref_tsr, np.eye(4), atol=1e-10)


class TestGraspTorusSpan(unittest.TestCase):

    def setUp(self):
        self.gripper = ParallelJawGripper(finger_length=FL, max_aperture=MA)

    # ── Conditional inclusion ─────────────────────────────────────────────

    def test_small_torus_generates_span_templates(self):
        ts = self.gripper.grasp_torus_span(SR, Sr)
        self.assertGreater(len(ts), 0)

    def test_large_torus_returns_empty(self):
        # auto-preshape = 2*(R+r)+clearance > MA → silently skip
        ts = self.gripper.grasp_torus_span(LR, Lr)
        self.assertEqual(ts, [])

    def test_default_k3_returns_6_templates_for_small_torus(self):
        ts = self.gripper.grasp_torus_span(SR, Sr)
        self.assertEqual(len(ts), 6)   # k top + k bottom = 6

    def test_k1_returns_2_templates(self):
        ts = self.gripper.grasp_torus_span(SR, Sr, k=1)
        self.assertEqual(len(ts), 2)

    # ── Geometry ──────────────────────────────────────────────────────────

    def test_tw_e_valid_se3(self):
        for t in self.gripper.grasp_torus_span(SR, Sr):
            _check_se3(self, t.Tw_e[:3, :3])

    def test_top_z_ee_points_down(self):
        """Top templates: z_EE = (0,0,-1) in TSR frame."""
        ts = self.gripper.grasp_torus_span(SR, Sr)
        top_templates = ts[::2]   # every other starting at 0 = top
        for t in top_templates:
            np.testing.assert_allclose(t.Tw_e[:3, 2], [0., 0., -1.], atol=1e-10)

    def test_bottom_z_ee_points_up(self):
        """Bottom templates: z_EE = (0,0,+1) in TSR frame."""
        ts = self.gripper.grasp_torus_span(SR, Sr)
        bot_templates = ts[1::2]   # every other starting at 1 = bottom
        for t in bot_templates:
            np.testing.assert_allclose(t.Tw_e[:3, 2], [0., 0., 1.], atol=1e-10)

    def test_top_tsr_origin_at_tube_top(self):
        ts = self.gripper.grasp_torus_span(SR, Sr)
        for t in ts[::2]:
            self.assertAlmostEqual(t.T_ref_tsr[2, 3], Sr)

    def test_bottom_tsr_origin_at_tube_bottom(self):
        ts = self.gripper.grasp_torus_span(SR, Sr)
        for t in ts[1::2]:
            self.assertAlmostEqual(t.T_ref_tsr[2, 3], -Sr)

    def test_bw_full_yaw_freedom(self):
        for t in self.gripper.grasp_torus_span(SR, Sr):
            self.assertAlmostEqual(t.Bw[5, 0], 0.)
            self.assertAlmostEqual(t.Bw[5, 1], 2 * np.pi)

    def test_bw_no_roll_pitch_freedom(self):
        for t in self.gripper.grasp_torus_span(SR, Sr):
            self.assertEqual(t.Bw[3, 0], t.Bw[3, 1])
            self.assertEqual(t.Bw[4, 0], t.Bw[4, 1])

    def test_preshape_spans_outer_diameter(self):
        clearance = self.gripper.clearance_fraction * FL
        expected = 2 * (SR + Sr) + clearance
        for t in self.gripper.grasp_torus_span(SR, Sr):
            self.assertAlmostEqual(t.preshape[0], expected)


class TestGraspTorusCombiner(unittest.TestCase):

    def setUp(self):
        self.gripper = ParallelJawGripper(finger_length=FL, max_aperture=MA)

    def test_small_torus_returns_side_plus_span(self):
        side = self.gripper.grasp_torus_side(SR, Sr)
        span = self.gripper.grasp_torus_span(SR, Sr)
        combined = self.gripper.grasp_torus(SR, Sr)
        self.assertEqual(len(combined), len(side) + len(span))

    def test_large_torus_returns_side_only(self):
        side = self.gripper.grasp_torus_side(LR, Lr)
        combined = self.gripper.grasp_torus(LR, Lr)
        self.assertEqual(len(combined), len(side))
        self.assertGreater(len(combined), 0)


if __name__ == "__main__":
    unittest.main()
