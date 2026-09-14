#!/usr/bin/env python
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Tests for SphereConstraint."""

import unittest

import numpy as np
from gafro import Motor, Point, Sphere

from tests.tsr._motor_helpers import motor_split
from tsr import Constraint, SphereConstraint


def _pose_at(x, y, z, biv=(0.0, 0.0, 0.0)) -> Motor:
    """World-frame EE Motor at position (x,y,z) with rotor-bivector log ``biv``."""
    from tests.tsr._motor_helpers import motor_from_split

    return motor_from_split(np.array([x, y, z], float), np.array(biv, float))


def _unit_sphere() -> Sphere:
    """Unit sphere centered at the origin."""
    return Sphere(Point(0.0, 0.0, 0.0), 1.0)


class TestConstraintInterface(unittest.TestCase):
    def test_sphere_is_constraint(self):
        self.assertIsInstance(SphereConstraint(_unit_sphere()), Constraint)


class TestSphereDistance(unittest.TestCase):
    def setUp(self):
        self.c = SphereConstraint(_unit_sphere())

    def test_on_shell_is_zero(self):
        for pt in [(1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 0.0, -1.0)]:
            self.assertAlmostEqual(self.c.distance(_pose_at(*pt))[0], 0.0, places=6)

    def test_outside_distance(self):
        # Point at radius 3 along +z: 2 outside the unit shell.
        self.assertAlmostEqual(self.c.distance(_pose_at(0.0, 0.0, 3.0))[0], 2.0, places=6)

    def test_inside_distance(self):
        # Center is radius 1 from the shell.
        self.assertAlmostEqual(self.c.distance(_pose_at(0.0, 0.0, 0.0))[0], 1.0, places=6)

    def test_distance_is_radial(self):
        # Distance depends only on the radial offset, not the direction.
        d1, _ = self.c.distance(_pose_at(2.0, 0.0, 0.0))
        d2, _ = self.c.distance(_pose_at(0.0, -2.0, 0.0))
        self.assertAlmostEqual(d1, d2, places=6)

    def test_contains(self):
        self.assertTrue(self.c.contains(_pose_at(0.0, 1.0, 0.0), tolerance=1e-6))
        self.assertFalse(self.c.contains(_pose_at(0.0, 1.5, 0.0), tolerance=1e-6))


class TestSphereProjection(unittest.TestCase):
    """Exercises Sphere.project via distance() -> witness -> to_transform()."""

    def setUp(self):
        self.c = SphereConstraint.from_center_radius([0.5, 0.0, 0.6], 0.2)

    def test_witness_is_on_shell(self):
        _, witness = self.c.distance(_pose_at(0.5, 0.0, 0.9))
        projected = self.c.to_transform(witness)
        d, _ = self.c.distance(projected)
        self.assertAlmostEqual(d, 0.0, places=5)

    def test_projection_lands_on_nearest_point(self):
        # (0.5,0,0.9) is straight above the center: nearest shell point is (0.5,0,0.8).
        _, witness = self.c.distance(_pose_at(0.5, 0.0, 0.9))
        xyz, _ = motor_split(self.c.to_transform(witness))
        np.testing.assert_allclose(xyz, [0.5, 0.0, 0.8], atol=1e-5)

    def test_sample_is_on_shell(self):
        np.random.seed(0)
        for _ in range(50):
            d, _ = self.c.distance(self.c.sample())
            self.assertAlmostEqual(d, 0.0, places=5)


class TestOrientationPinned(unittest.TestCase):
    def setUp(self):
        self.biv = np.array([0.0, 0.0, np.pi / 2])  # yaw of pi/2 about z
        self.c = SphereConstraint(_unit_sphere(), orientation=self.biv)

    def test_position_and_orientation_satisfied(self):
        d, _ = self.c.distance(_pose_at(1.0, 0.0, 0.0, biv=self.biv))
        self.assertAlmostEqual(d, 0.0, places=5)

    def test_orientation_violation_counts(self):
        d, _ = self.c.distance(_pose_at(1.0, 0.0, 0.0, biv=(0.0, 0.0, 0.0)))
        self.assertGreater(d, 0.1)

    def test_projection_pins_orientation(self):
        _, witness = self.c.distance(_pose_at(0.0, 0.0, 3.0, biv=(0.0, 0.0, 0.0)))
        _, biv = motor_split(self.c.to_transform(witness))
        np.testing.assert_allclose(biv, self.biv, atol=1e-4)


class TestSerialization(unittest.TestCase):
    def test_round_trip(self):
        c = SphereConstraint.from_center_radius([0.5, 0.0, 0.6], 0.2)
        c2 = SphereConstraint.from_dict(c.to_dict())
        for pt in [(0.5, 0.0, 0.8), (0.5, 0.0, 1.0), (0.7, 0.3, 0.6)]:
            self.assertAlmostEqual(
                c.distance(_pose_at(*pt))[0],
                c2.distance(_pose_at(*pt))[0],
                places=5,
            )

    def test_round_trip_with_orientation(self):
        c = SphereConstraint(_unit_sphere(), orientation=[0.0, 0.0, np.pi / 2])
        c2 = SphereConstraint.from_dict(c.to_dict())
        d2, _ = c2.distance(_pose_at(1.0, 0.0, 0.0, biv=(0.0, 0.0, np.pi / 2)))
        self.assertAlmostEqual(d2, 0.0, places=5)

    def test_rejects_unknown_format(self):
        with self.assertRaises(ValueError):
            SphereConstraint.from_dict({"format": "nope", "sphere": [0, 0, 0, 0, 0]})


class TestFromCenterRadius(unittest.TestCase):
    def test_matches_explicit_sphere(self):
        c = SphereConstraint.from_center_radius([0.0, 0.0, 0.0], 1.0)
        self.assertAlmostEqual(c.distance(_pose_at(1.0, 0.0, 0.0))[0], 0.0, places=5)
        self.assertAlmostEqual(c.distance(_pose_at(0.0, 0.0, 3.0))[0], 2.0, places=5)

    def test_offset_sphere(self):
        c = SphereConstraint.from_center_radius([1.0, 2.0, 3.0], 0.5)
        self.assertAlmostEqual(c.distance(_pose_at(1.5, 2.0, 3.0))[0], 0.0, places=5)
        self.assertAlmostEqual(c.distance(_pose_at(1.0, 2.0, 3.0))[0], 0.5, places=5)


if __name__ == "__main__":
    unittest.main()
