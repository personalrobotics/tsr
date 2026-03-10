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

# Small torus: span fits (2*(R+r)+clearance = 2*0.055+0.0055 = 0.1155 < 0.14)
SR, Sr = 0.040, 0.015

# Large torus: span doesn't fit (2*(R+r)+clearance = 2*0.085+0.0055 = 0.1755 > 0.14)
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

    def test_default_k3_returns_6_templates(self):
        self.assertEqual(len(self.templates), 6)   # 2 rolls × 3 depths

    def test_k1_returns_2_templates(self):
        ts = self.gripper.grasp_torus_side(SR, Sr, k=1)
        self.assertEqual(len(ts), 2)

    # ── Geometry: Tw_e ────────────────────────────────────────────────────

    def test_tw_e_valid_se3(self):
        for t in self.templates:
            _check_se3(self, t.Tw_e[:3, :3])

    def test_z_ee_points_inward(self):
        """z_EE should be (-1,0,0) in TSR frame (radially inward)."""
        for t in self.templates:
            np.testing.assert_allclose(t.Tw_e[:3, 2], [-1., 0., 0.], atol=1e-10)

    def test_standoff_includes_major_radius(self):
        """Standoff = R + r + fl - d; must be > R+r (outside tube surface)."""
        for t in self.templates:
            ro = t.Tw_e[0, 3]   # x-translation = standoff from torus axis
            self.assertGreater(ro, SR + Sr,
                               msg=f"standoff {ro} not > R+r in {t.name}")

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
        clearance = 0.1 * FL
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
        clearance = 0.1 * FL
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
