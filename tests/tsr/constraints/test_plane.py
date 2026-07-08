#!/usr/bin/env python
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Tests for PlaneConstraint and the Constraint base class."""

import unittest

import numpy as np

from gafropy import Motor, Plane, Point
from tsr import Constraint, PlaneConstraint, TSR
from tests.tsr._motor_helpers import as_motor, motor_split


def _pose_at(x, y, z, biv=(0.0, 0.0, 0.0)) -> Motor:
    """World-frame EE Motor at position (x,y,z) with rotor-bivector log ``biv``."""
    from tests.tsr._motor_helpers import motor_from_split

    return motor_from_split(np.array([x, y, z], float), np.array(biv, float))


def _z1_plane() -> Plane:
    """The working z=1 plane (built via the 3-point constructor)."""
    return Plane(Point(0, 0, 1.0), Point(1, 0, 1.0), Point(0, 1, 1.0))


class TestConstraintInterface(unittest.TestCase):
    def test_tsr_is_constraint(self):
        self.assertIsInstance(TSR(), Constraint)

    def test_plane_is_constraint(self):
        self.assertIsInstance(PlaneConstraint(_z1_plane()), Constraint)


class TestPlaneDistance(unittest.TestCase):
    def setUp(self):
        self.c = PlaneConstraint(_z1_plane())

    def test_on_plane_is_zero(self):
        dist, _ = self.c.distance(_pose_at(0.5, -0.3, 1.0))
        self.assertAlmostEqual(dist, 0.0, places=6)

    def test_off_plane_distance(self):
        dist, _ = self.c.distance(_pose_at(0.0, 0.0, 3.0))
        self.assertAlmostEqual(dist, 2.0, places=6)

    def test_position_invariant_in_plane(self):
        # Distance depends only on perpendicular offset, not in-plane position.
        d1, _ = self.c.distance(_pose_at(0.0, 0.0, 2.0))
        d2, _ = self.c.distance(_pose_at(9.0, -4.0, 2.0))
        self.assertAlmostEqual(d1, d2, places=6)

    def test_contains(self):
        self.assertTrue(self.c.contains(_pose_at(1.0, 1.0, 1.0), tolerance=1e-6))
        self.assertFalse(self.c.contains(_pose_at(1.0, 1.0, 1.5), tolerance=1e-6))


class TestPlaneProjection(unittest.TestCase):
    """Exercises Plane.project via distance() -> witness -> to_transform()."""

    def setUp(self):
        self.c = PlaneConstraint(_z1_plane())

    def test_witness_is_on_plane(self):
        _, witness = self.c.distance(_pose_at(0.4, -0.2, 3.0))
        projected = self.c.to_transform(witness)
        d, _ = self.c.distance(projected)
        self.assertAlmostEqual(d, 0.0, places=5)

    def test_projection_preserves_in_plane_coords(self):
        _, witness = self.c.distance(_pose_at(0.4, -0.2, 3.0))
        xyz, _ = motor_split(self.c.to_transform(witness))
        self.assertAlmostEqual(xyz[0], 0.4, places=5)
        self.assertAlmostEqual(xyz[1], -0.2, places=5)
        self.assertAlmostEqual(xyz[2], 1.0, places=5)

    def test_sample_is_on_plane(self):
        for _ in range(20):
            pose = self.c.sample()
            d, _ = self.c.distance(pose)
            self.assertAlmostEqual(d, 0.0, places=5)


class TestOrientationPinned(unittest.TestCase):
    def setUp(self):
        self.biv = np.array([0.0, 0.0, np.pi / 2])  # yaw of pi/2 about z
        self.c = PlaneConstraint(_z1_plane(), orientation=self.biv)

    def test_position_and_orientation_satisfied(self):
        d, _ = self.c.distance(_pose_at(0.5, 0.5, 1.0, biv=self.biv))
        self.assertAlmostEqual(d, 0.0, places=5)

    def test_orientation_violation_counts(self):
        d, _ = self.c.distance(_pose_at(0.5, 0.5, 1.0, biv=(0.0, 0.0, 0.0)))
        self.assertGreater(d, 0.1)

    def test_projection_pins_orientation(self):
        _, witness = self.c.distance(_pose_at(0.5, 0.5, 3.0, biv=(0.0, 0.0, 0.0)))
        _, biv = motor_split(self.c.to_transform(witness))
        np.testing.assert_allclose(biv, self.biv, atol=1e-4)


class TestSerialization(unittest.TestCase):
    def test_round_trip(self):
        c = PlaneConstraint.from_point_normal([0, 0, 1.0], [0, 0, 1.0])
        d = c.to_dict()
        c2 = PlaneConstraint.from_dict(d)
        # Same geometry: identical distances on a probe set.
        for pt in [(0, 0, 1.0), (0, 0, 2.0), (3, 4, 1.0)]:
            self.assertAlmostEqual(
                c.distance(_pose_at(*pt))[0],
                c2.distance(_pose_at(*pt))[0],
                places=5,
            )

    def test_round_trip_with_orientation(self):
        c = PlaneConstraint(_z1_plane(), orientation=[0.0, 0.0, np.pi / 2])
        c2 = PlaneConstraint.from_dict(c.to_dict())
        d2, _ = c2.distance(_pose_at(0.5, 0.5, 1.0, biv=(0.0, 0.0, np.pi / 2)))
        self.assertAlmostEqual(d2, 0.0, places=5)

    def test_rejects_unknown_format(self):
        with self.assertRaises(ValueError):
            PlaneConstraint.from_dict({"format": "nope", "plane": [1, 0, 0, 0]})


class TestFromPointNormal(unittest.TestCase):
    def test_matches_explicit_plane(self):
        c = PlaneConstraint.from_point_normal([0, 0, 1.0], [0, 0, 1.0])
        self.assertAlmostEqual(c.distance(_pose_at(0, 0, 1.0))[0], 0.0, places=5)
        self.assertAlmostEqual(c.distance(_pose_at(0, 0, 3.0))[0], 2.0, places=5)

    def test_tilted_plane(self):
        # Plane through origin with normal along x: the x=0 plane.
        c = PlaneConstraint.from_point_normal([0, 0, 0.0], [1.0, 0.0, 0.0])
        self.assertAlmostEqual(c.distance(_pose_at(0, 5, 9.0))[0], 0.0, places=5)
        self.assertAlmostEqual(c.distance(_pose_at(2, 0, 0.0))[0], 2.0, places=5)


if __name__ == "__main__":
    unittest.main()
