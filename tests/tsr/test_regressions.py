# SPDX-License-Identifier: BSD-2-Clause
# Authors: Siddhartha Srinivasa and contributors to TSR

"""Regression tests for bugs fixed in the v2.0.0 review (see docs/REVIEW.md)."""

import unittest

import numpy as np
import pytest
from numpy import pi

from tsr import ParallelJawGripper, StablePlacer
from tsr.core.tsr import TSR


class TestC1GimbalConsistency(unittest.TestCase):
    """sample() poses must be contained; round-trips accurate near gimbal lock."""

    def test_near_gimbal_pitch_sample_is_contained(self):
        # pitch ~ -1.602 rad (just inside the old EPSILON snap band) used to make
        # contains() reject a pose that sample()/is_valid() accepted.
        Bw = np.array([[0, 0.3], [0, 0.3], [0, 0.3], [-0.3, 0.0], [-1.65, -1.55], [-2.0, -1.5]])
        tsr = TSR(Bw=Bw)
        for _ in range(200):
            pose = tsr.sample()
            self.assertTrue(tsr.contains(pose))
            self.assertLess(abs(tsr.distance(pose)[0]), 1e-6)

    def test_roundtrip_accurate_just_outside_singularity(self):
        # rot[2,0] ~ 0.9995 (pitch ~2.5deg from +/-90) previously snapped and lost ~0.04 rad.
        R = TSR.rpy_to_rot([0.3, np.arcsin(0.9995), 0.7])
        R2 = TSR.rpy_to_rot(TSR.rot_to_rpy(R))
        np.testing.assert_allclose(R, R2, atol=1e-6)


class TestC2PlacementRestingHeight(unittest.TestCase):
    """place_mesh: the resting face must sit at z=0 for any COM (not just centered)."""

    def _unit_cube_corner_origin(self):
        return np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], float)

    def test_off_origin_com_rests_on_table(self):
        verts = self._unit_cube_corner_origin()  # origin at a corner
        com = np.array([0.5, 0.5, 0.5])  # geometric center, != origin
        placer = StablePlacer(0.3, 0.3)
        for t in placer.place_mesh(verts, com):
            pose = t.instantiate(np.eye(4)).to_transform(np.zeros(6))
            world = (pose[:3, :3] @ verts.T).T + pose[:3, 3]
            self.assertAlmostEqual(world[:, 2].min(), 0.0, places=6)

    def test_random_offcenter_com_meshes_rest_on_table(self):
        rng = np.random.default_rng(0)
        placer = StablePlacer(0.3, 0.3)
        for _ in range(50):
            pts = rng.normal(size=(12, 3))
            com = pts.mean(0) + rng.normal(scale=0.1, size=3)
            for t in placer.place_mesh(pts, com):
                pose = t.instantiate(np.eye(4)).to_transform(np.zeros(6))
                world = (pose[:3, :3] @ pts.T).T + pose[:3, 3]
                self.assertAlmostEqual(world[:, 2].min(), 0.0, places=6)


class TestC3DepthCountValidation(unittest.TestCase):
    """k < 1 is invalid input -> ValueError (used to IndexError on the depth label)."""

    def setUp(self):
        self.g = ParallelJawGripper(finger_length=0.055, max_aperture=0.140)

    def test_k_zero_raises(self):
        for fn in (
            lambda: self.g.grasp_cylinder_top(0.03, 0.10, k=0),
            lambda: self.g.grasp_cylinder_side(0.03, 0.10, k=0),
            lambda: self.g.grasp_box_top(0.05, 0.05, 0.05, k=0),
            lambda: self.g.grasp_sphere(0.03, k=0),
            lambda: self.g.grasp_torus_side(0.06, 0.02, k=0),
        ):
            with self.assertRaises(ValueError):
                fn()


class TestC4AngleRangeValidation(unittest.TestCase):
    """Reversed angle_range (max, min) is invalid input -> ValueError."""

    def setUp(self):
        self.g = ParallelJawGripper(finger_length=0.055, max_aperture=0.140)

    def test_reversed_angle_range_raises(self):
        with self.assertRaises(ValueError):
            self.g.grasp_cylinder_top(0.03, 0.10, angle_range=(2 * pi, 0.0))


class TestC5DegenerateMesh(unittest.TestCase):
    """Degenerate meshes raise a friendly ValueError, not a raw QhullError."""

    def setUp(self):
        self.placer = StablePlacer(0.3, 0.3)

    def test_too_few_points(self):
        with pytest.raises(ValueError):
            self.placer.place_mesh(np.zeros((3, 3)), np.zeros(3))

    def test_coplanar_points(self):
        flat = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], float)  # all z=0
        with pytest.raises(ValueError):
            self.placer.place_mesh(flat, flat.mean(0))


if __name__ == "__main__":
    unittest.main()
