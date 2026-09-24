#!/usr/bin/env python
# SPDX-License-Identifier: BSD-2-Clause
# Authors: Siddhartha Srinivasa and contributors to TSR

"""The experimental Viser backend (issue #77).

Only the pure geometry and sampling helpers are tested here: **no server is started**
and no browser is involved, so the default suite neither requires the optional extra
nor opens a port. The interactive behaviour this backend exists to evaluate is judged
by a human, not asserted here — visualization is diagnostic, never part of the
template-correctness argument.
"""

import unittest

import numpy as np

from tsr.hands import ParallelJawGripper

viser_backend = __import__("importlib").util.find_spec("viser")
if viser_backend is not None:
    from tsr.viser import _wxyz as wxyz
    from tsr.viser import free_coordinates, gripper_segments, sample_poses

GRIPPER = ParallelJawGripper(finger_length=0.08, max_aperture=0.14)


@unittest.skipIf(viser_backend is None, "optional extra 'viser' is not installed")
class TestSampling(unittest.TestCase):
    def setUp(self):
        self.templates = GRIPPER.grasp_cylinder_side(0.03, 0.12)

    def test_seed_makes_the_view_reproducible(self):
        a = sample_poses(self.templates, 3, seed=7)
        b = sample_poses(self.templates, 3, seed=7)
        self.assertEqual(len(a), 3 * len(self.templates))
        for pa, pb in zip(a, b):
            np.testing.assert_array_equal(pa, pb)

    def test_an_explicit_rng_is_honoured(self):
        poses = sample_poses(self.templates, 2, rng=np.random.default_rng(1))
        other = sample_poses(self.templates, 2, rng=np.random.default_rng(2))
        self.assertTrue(any(not np.array_equal(p, q) for p, q in zip(poses, other)))

    def test_sampled_poses_lie_in_their_template(self):
        # The viewer must show poses the TSR actually admits, at the requested frame.
        T = np.eye(4)
        T[:3, 3] = [0.4, -0.2, 0.1]
        for template in self.templates:
            tsr = template.instantiate(T)
            for pose in sample_poses([template], 4, T_ref_world=T, seed=3):
                self.assertTrue(tsr.contains(pose))


@unittest.skipIf(viser_backend is None, "optional extra 'viser' is not installed")
class TestFreeCoordinates(unittest.TestCase):
    """The sliders are driven by the template's own non-degenerate Bw rows."""

    def test_cylinder_side_is_free_in_z_and_yaw(self):
        template = GRIPPER.grasp_cylinder_side(0.03, 0.12)[0]
        free = free_coordinates(template)
        self.assertEqual([row for row, _, _, _ in free], [2, 5])
        self.assertEqual([label for _, label, _, _ in free], ["z [m]", "yaw [rad]"])
        for row, _label, lo, hi in free:
            self.assertEqual((lo, hi), (template.Bw[row, 0], template.Bw[row, 1]))
            self.assertLess(lo, hi)

    def test_cylinder_top_is_free_in_yaw_only(self):
        template = GRIPPER.grasp_cylinder_top(0.03, 0.12)[0]
        self.assertEqual([label for _, label, _, _ in free_coordinates(template)], ["yaw [rad]"])

    def test_degenerate_rows_are_never_offered(self):
        # A zero-width row is not a freedom; putting a slider on it would be a lie.
        for template in GRIPPER.grasp_box_top(0.05, 0.05, 0.05):
            for row, _label, lo, hi in free_coordinates(template):
                self.assertGreater(hi, lo, row)


@unittest.skipIf(viser_backend is None, "optional extra 'viser' is not installed")
class TestGeometry(unittest.TestCase):
    def test_jaw_segments_follow_the_library_frame_convention(self):
        segments = gripper_segments(finger_length=0.08, aperture=0.06)
        self.assertEqual(segments.shape, (4, 2, 3))
        crossbar, finger_neg, finger_pos, stick = segments
        # Fingers extend along +z (approach) from the palm plane at z = 0.
        np.testing.assert_allclose([finger_neg[0][2], finger_pos[0][2]], [0.0, 0.0])
        np.testing.assert_allclose([finger_neg[1][2], finger_pos[1][2]], [0.08, 0.08])
        # The crossbar spans the opening along y, at the requested aperture.
        np.testing.assert_allclose(crossbar[:, 1], [-0.03, 0.03])
        self.assertLess(stick[1][2], 0.0)  # the approach stick sits behind the palm

    def test_quaternion_matches_the_rotation(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
            if np.linalg.det(q) < 0:
                q[:, 0] *= -1.0
            w, x, y, z = wxyz(q)
            # Rebuild the matrix from the quaternion and compare.
            R = np.array(
                [
                    [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                    [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                    [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
                ]
            )
            np.testing.assert_allclose(R, q, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
