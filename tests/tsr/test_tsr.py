# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

from unittest import TestCase

import numpy
from gafropy import Motor
from numpy import pi

from tests.tsr._motor_helpers import to_matrix
from tsr.tsr import TSR


class TsrTest(TestCase):
    def test_sample_bw(self):
        # Test zero-intervals (translation box + rotor-bivector box).
        Bw = [
            [0.0, 0.0],  # tx
            [1.0, 1.0],  # ty
            [-1.0, -1.0],  # tz
            [0.0, 0.0],  # b12
            [0.5, 0.5],  # b13
            [-0.5, -0.5],  # b23
        ]
        tsr = TSR(Bw=Bw)
        s = tsr.sample_bw()

        Bw = numpy.array(Bw)
        # For zero-intervals, the sampled value equals the bound exactly (no
        # Euler wrapping in the split parametrization).
        self.assertTrue(numpy.allclose(s, Bw[:, 0], atol=1e-10))

        # Test simple non-zero intervals
        Bw = [
            [-0.1, 0.1],  # tx
            [-0.1, 0.1],  # ty
            [-0.1, 0.1],  # tz
            [-pi / 4, pi / 4],  # b12
            [-pi / 4, pi / 4],  # b13
            [-pi / 4, pi / 4],  # b23
        ]
        tsr = TSR(Bw=Bw)
        s = tsr.sample_bw()

        Bw = numpy.array(Bw)
        self.assertTrue(numpy.all(s >= Bw[:, 0]))
        self.assertTrue(numpy.all(s <= Bw[:, 1]))

    def test_tsr_creation(self):
        """Test basic TSR creation."""
        T0_w = numpy.eye(4)
        Tw_e = numpy.eye(4)
        Bw = numpy.zeros((6, 2))
        # Rotation-first (Motor.log) order: rows 0:3 = rotor bivector (b12,b13,b23),
        # rows 3:6 = translation (tx,ty,tz). Row 5 is tz; row 2 is b23 (yaw).
        Bw[5, :] = [0.0, 0.02]  # Allow vertical (z) movement
        Bw[2, :] = [-pi, pi]  # Allow any yaw rotation (b23)

        tsr = TSR(T0_w=T0_w, Tw_e=Tw_e, Bw=Bw)

        self.assertIsInstance(tsr.T0_w, Motor)
        self.assertIsInstance(tsr.Tw_e, Motor)
        self.assertIsInstance(tsr.Bw, numpy.ndarray)
        self.assertEqual(to_matrix(tsr.T0_w).shape, (4, 4))
        self.assertEqual(to_matrix(tsr.Tw_e).shape, (4, 4))
        self.assertEqual(tsr.Bw.shape, (6, 2))

    def test_tsr_sampling(self):
        """Test TSR sampling functionality."""
        T0_w = numpy.eye(4)
        Tw_e = numpy.eye(4)
        Bw = numpy.zeros((6, 2))
        # Rotation-first (Motor.log) order: rows 0:3 = rotor bivector (b12,b13,b23),
        # rows 3:6 = translation (tx,ty,tz). Row 5 is tz; row 2 is b23 (yaw).
        Bw[5, :] = [0.0, 0.02]  # Allow vertical (z) movement
        Bw[2, :] = [-pi, pi]  # Allow any yaw rotation (b23)

        tsr = TSR(T0_w=T0_w, Tw_e=Tw_e, Bw=Bw)

        # Test sampling
        pose = tsr.sample()
        self.assertIsInstance(pose, Motor)
        self.assertEqual(to_matrix(pose).shape, (4, 4))

        # Test xyzrpy sampling
        xyzrpy = tsr.sample_bw()
        self.assertIsInstance(xyzrpy, numpy.ndarray)
        self.assertEqual(xyzrpy.shape, (6,))

    def test_tsr_validation(self):
        """Test TSR validation."""
        T0_w = numpy.eye(4)
        Tw_e = numpy.eye(4)
        Bw = numpy.zeros((6, 2))
        # Rotation-first (Motor.log) order: rows 0:3 = rotor bivector (b12,b13,b23),
        # rows 3:6 = translation (tx,ty,tz). Row 5 is tz; row 2 is b23 (yaw).
        Bw[5, :] = [0.0, 0.02]  # Allow vertical (z) movement
        Bw[2, :] = [-pi, pi]  # Allow any yaw rotation (b23)

        tsr = TSR(T0_w=T0_w, Tw_e=Tw_e, Bw=Bw)

        # Test valid bw (z = row 5 within [0, 0.02])
        valid_xyzrpy = numpy.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.01])
        self.assertTrue(all(tsr.is_valid(valid_xyzrpy)))

        # Test invalid bw (z = row 5 outside [0, 0.02])
        invalid_xyzrpy = numpy.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.1])  # z too large
        self.assertFalse(all(tsr.is_valid(invalid_xyzrpy)))

    def test_tsr_contains(self):
        """Test TSR containment checking."""
        T0_w = numpy.eye(4)
        Tw_e = numpy.eye(4)
        Bw = numpy.zeros((6, 2))
        # Rotation-first (Motor.log) order: rows 0:3 = rotor bivector (b12,b13,b23),
        # rows 3:6 = translation (tx,ty,tz). Row 5 is tz; row 2 is b23 (yaw).
        Bw[5, :] = [0.0, 0.02]  # Allow vertical (z) movement
        Bw[2, :] = [-pi, pi]  # Allow any yaw rotation (b23)

        tsr = TSR(T0_w=T0_w, Tw_e=Tw_e, Bw=Bw)

        # Test contained transform
        contained_transform = numpy.eye(4)
        contained_transform[2, 3] = 0.01  # Within z bounds
        self.assertTrue(tsr.contains(contained_transform))

        # Test non-contained transform
        non_contained_transform = numpy.eye(4)
        non_contained_transform[2, 3] = 0.1  # Outside z bounds
        self.assertFalse(tsr.contains(non_contained_transform))

    def test_tsr_distance(self):
        """Test TSR distance calculation."""
        T0_w = numpy.eye(4)
        Tw_e = numpy.eye(4)
        Bw = numpy.zeros((6, 2))
        # Rotation-first (Motor.log) order: rows 0:3 = rotor bivector (b12,b13,b23),
        # rows 3:6 = translation (tx,ty,tz). Row 5 is tz; row 2 is b23 (yaw).
        Bw[5, :] = [0.0, 0.02]  # Allow vertical (z) movement
        Bw[2, :] = [-pi, pi]  # Allow any yaw rotation (b23)

        tsr = TSR(T0_w=T0_w, Tw_e=Tw_e, Bw=Bw)

        # Test distance to contained transform
        contained_transform = numpy.eye(4)
        contained_transform[2, 3] = 0.01
        distance, bwopt = tsr.distance(contained_transform)
        self.assertEqual(distance, 0.0)

        # Test distance to non-contained transform
        non_contained_transform = numpy.eye(4)
        non_contained_transform[2, 3] = 0.1
        distance, bwopt = tsr.distance(non_contained_transform)
        self.assertGreater(distance, 0.0)

    def test_contains_with_non_identity_frames(self):
        """Test TSR containment with non-identity T0_w and Tw_e.

        This test ensures contains() correctly transforms to the TSR frame
        before checking bounds. With identity frames, bugs in frame handling
        are invisible.
        """
        # Non-identity T0_w: TSR origin offset from world origin
        T0_w = numpy.eye(4)
        T0_w[0, 3] = 1.0  # TSR origin at x=1
        T0_w[1, 3] = 2.0  # TSR origin at y=2

        # Non-identity Tw_e: end-effector offset from TSR frame
        Tw_e = numpy.eye(4)
        Tw_e[2, 3] = 0.5  # End-effector 0.5m above TSR frame

        Bw = numpy.array(
            [
                [-0.1, 0.1],  # X bounds
                [-0.1, 0.1],  # Y bounds
                [-0.1, 0.1],  # Z bounds
                [-pi / 4, pi / 4],  # roll bounds
                [-pi / 4, pi / 4],  # pitch bounds
                [-pi / 4, pi / 4],  # yaw bounds
            ]
        )

        tsr = TSR(T0_w=T0_w, Tw_e=Tw_e, Bw=Bw)

        # Generate a contained transform using to_transform (round-trip test)
        valid_bw = numpy.array([0.05, 0.05, 0.05, 0.1, 0.1, 0.1])
        contained_transform = tsr.to_transform(valid_bw)

        self.assertTrue(tsr.contains(contained_transform))

        # A transform at the world origin should NOT be contained
        # (it's far from the TSR which is centered at x=1, y=2)
        world_origin = numpy.eye(4)
        self.assertFalse(tsr.contains(world_origin))

    def test_distance_with_non_identity_frames(self):
        """Test TSR distance with non-identity T0_w and Tw_e.

        Verifies that distance() correctly handles frame transformations
        and returns 0 for contained transforms.
        """
        # Rotated T0_w: TSR frame rotated 90 degrees about Z
        T0_w = numpy.array([[0, -1, 0, 0.5], [1, 0, 0, 0.5], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)

        # Tw_e with rotation and translation
        Tw_e = numpy.array([[0, 0, 1, 0.1], [1, 0, 0, 0], [0, 1, 0, 0.05], [0, 0, 0, 1]], dtype=float)

        Bw = numpy.array(
            [
                [-0.05, 0.05],
                [-0.05, 0.05],
                [-0.05, 0.05],
                [-pi / 6, pi / 6],
                [-pi / 6, pi / 6],
                [-pi / 6, pi / 6],
            ]
        )

        tsr = TSR(T0_w=T0_w, Tw_e=Tw_e, Bw=Bw)

        # Generate contained transform via to_transform
        valid_bw = numpy.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        contained_transform = tsr.to_transform(valid_bw)

        distance, bwopt = tsr.distance(contained_transform)
        self.assertEqual(distance, 0.0)
        self.assertTrue(tsr.contains(contained_transform))

        # Test at bounds edge
        edge_bw = numpy.array([0.05, 0.05, 0.05, 0.0, 0.0, 0.0])
        edge_transform = tsr.to_transform(edge_bw)

        distance, bwopt = tsr.distance(edge_transform)
        # CGA composition leaves a machine-epsilon residual at the exact bound
        # edge (the legacy matrix path happened to land on exactly 0.0).
        self.assertAlmostEqual(distance, 0.0, places=9)
        self.assertTrue(tsr.contains(edge_transform))

    def test_contains_distance_consistency(self):
        """Test that contains() and distance() are consistent.

        For any transform:
        - contains(t) == True  implies distance(t) == 0
        - contains(t) == False implies distance(t) > 0
        """
        # Use non-trivial frames to catch frame-handling bugs
        T0_w = numpy.array([[1, 0, 0, 0.3], [0, 1, 0, -0.2], [0, 0, 1, 0.1], [0, 0, 0, 1]], dtype=float)

        Tw_e = numpy.array([[0, -1, 0, 0], [1, 0, 0, 0.1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)

        Bw = numpy.array(
            [
                [-0.1, 0.1],
                [-0.1, 0.1],
                [-0.1, 0.1],
                [-pi / 4, pi / 4],
                [-pi / 4, pi / 4],
                [-pi / 4, pi / 4],
            ]
        )

        tsr = TSR(T0_w=T0_w, Tw_e=Tw_e, Bw=Bw)

        # Test multiple random samples - all should be contained with distance 0
        for _ in range(10):
            sample_bw = tsr.sample_bw()
            sample_transform = tsr.to_transform(sample_bw)

            is_contained = tsr.contains(sample_transform)
            distance, _ = tsr.distance(sample_transform)

            self.assertTrue(is_contained, f"Sampled transform should be contained, bw={sample_bw}")
            self.assertEqual(
                distance,
                0.0,
                f"Distance should be 0 for contained transform, bw={sample_bw}",
            )

        # Test transforms outside the TSR
        outside_transforms = [
            numpy.eye(4),  # World origin - likely outside
            numpy.diag([1, 1, 1, 1]).astype(float),  # Another identity
        ]
        # Add a clearly outside transform
        far_away = numpy.eye(4)
        far_away[0, 3] = 100.0  # Very far in x
        outside_transforms.append(far_away)

        for t in outside_transforms:
            is_contained = tsr.contains(t)
            distance, _ = tsr.distance(t)

            # If not contained, distance must be > 0
            if not is_contained:
                self.assertGreater(
                    distance,
                    0.0,
                    "Non-contained transform should have positive distance",
                )

    def test_roundtrip_to_transform_to_xyzrpy(self):
        """Test that to_transform and to_xyzrpy are inverses.

        This validates the frame transformations are consistent.
        """
        T0_w = numpy.array([[0, 0, 1, 1.0], [0, 1, 0, 0], [-1, 0, 0, 0.5], [0, 0, 0, 1]], dtype=float)

        Tw_e = numpy.eye(4)
        Tw_e[0, 3] = 0.2

        Bw = numpy.array(
            [
                [-0.1, 0.1],
                [-0.1, 0.1],
                [-0.1, 0.1],
                [-pi / 4, pi / 4],
                [-pi / 4, pi / 4],
                [-pi / 4, pi / 4],
            ]
        )

        tsr = TSR(T0_w=T0_w, Tw_e=Tw_e, Bw=Bw)

        # Test round-trip: xyzrpy -> transform -> xyzrpy
        original_bw = numpy.array([0.05, -0.03, 0.02, 0.1, -0.1, 0.2])
        transform = tsr.to_transform(original_bw)
        recovered_bw = tsr.to_bw(transform)

        numpy.testing.assert_array_almost_equal(
            original_bw,
            recovered_bw,
            decimal=9,
            err_msg="Round-trip xyzrpy -> transform -> xyzrpy failed",
        )

    def test_full_turn_bivector_bound(self):
        """A rotor-bivector bound of [-pi, pi] on one axis is a full turn.

        Replaces the legacy Euler 'outer interval' test: in the split
        parametrization there are no wraparound outer intervals — a free
        rotation about an axis is simply ``b_axis in [-pi, pi]``.
        """
        # Free rotation about the z axis (b23), translation pinned to origin.
        Bw = numpy.array(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [-pi, pi],
            ]
        )
        tsr = TSR(Bw=Bw)

        # Any pure z-rotation is contained, with distance 0.
        for angle in numpy.linspace(-pi + 0.05, pi - 0.05, 12):
            trans = tsr.to_transform(numpy.array([0, 0, 0, 0, 0, angle]))
            self.assertTrue(tsr.contains(trans), f"z-rotation by {angle} should be contained")
            dist, _ = tsr.distance(trans)
            self.assertAlmostEqual(dist, 0.0, places=9)

        # A rotation about x (b12) is NOT in this TSR.
        from tests.tsr._motor_helpers import motor_from_split

        off_axis = motor_from_split([0, 0, 0], [0.5, 0, 0])
        self.assertFalse(tsr.contains(off_axis), "x-rotation should not be contained")
        dist, _ = tsr.distance(off_axis)
        self.assertGreater(dist, 0.0)

    def test_rotor_log_exp_roundtrip(self):
        """rotor_log / rotor_exp (and motor split) round-trip on the bivector.

        Replaces the legacy Euler rot<->rpy round-trip tests; the split
        parametrization has no gimbal singularity to special-case.
        """
        from tests.tsr._motor_helpers import motor_from_split, motor_split

        rng = numpy.random.default_rng(0)
        from gafropy import Motor

        for _ in range(200):
            M = Motor.Random()
            t, b = motor_split(M)
            M2 = motor_from_split(t, b)
            numpy.testing.assert_allclose(
                M.to_transformation_matrix(),
                M2.to_transformation_matrix(),
                atol=1e-9,
                err_msg="motor split round-trip failed",
            )
