# SPDX-License-Identifier: BSD-2-Clause
# Authors: Siddhartha Srinivasa and contributors to TSR

"""Tests for FrankaHand."""

import unittest

import numpy as np

from tsr.hands import FrankaHand, ParallelJawGripper


class TestFrankaHand(unittest.TestCase):
    def setUp(self):
        self.gripper = FrankaHand()

    def test_fixed_params(self):
        # finger_length = palm(hand-body forward edge) -> pad tip, from the
        # menagerie hand.xml (same palm->pad-tip convention as the Robotiq hands).
        self.assertAlmostEqual(self.gripper.finger_length, 0.037)
        self.assertAlmostEqual(self.gripper.max_aperture, 0.080)
        self.assertAlmostEqual(self.gripper.PALM_OFFSET_FROM_HAND, 0.0753)

    def test_is_subclass_of_parallel_jaw(self):
        self.assertIsInstance(self.gripper, ParallelJawGripper)

    def test_tw_e_is_valid_se3(self):
        for t in self.gripper.grasp_cylinder_side(0.030, 0.10):
            R = t.Tw_e[:3, :3]
            np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)
            np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
