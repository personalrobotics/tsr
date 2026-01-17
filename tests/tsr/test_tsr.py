import numpy
from numpy import pi
from tsr.core.tsr import TSR
from unittest import TestCase


class TsrTest(TestCase):
    def test_sample_xyzrpy(self):
        # Test zero-intervals.
        Bw = [[0.,   0.],    # X
              [1.,   1.],    # Y
              [-1., -1.],    # Z
              [0.,   0.],    # roll
              [pi,   pi],    # pitch
              [-pi, -pi]]  # yaw
        tsr = TSR(Bw=Bw)
        s = tsr.sample_xyzrpy()

        Bw = numpy.array(Bw)
        # For zero-intervals, the sampled value should be exactly equal to the bound
        # Note: angles get wrapped, so pi becomes -pi
        expected = Bw[:, 0].copy()
        expected[4] = -pi  # pitch gets wrapped from pi to -pi
        self.assertTrue(numpy.allclose(s, expected, atol=1e-10))

        # Test simple non-zero intervals
        Bw = [[-0.1, 0.1],   # X
              [-0.1, 0.1],   # Y
              [-0.1, 0.1],   # Z
              [-pi/4, pi/4], # roll
              [-pi/4, pi/4], # pitch
              [-pi/4, pi/4]] # yaw
        tsr = TSR(Bw=Bw)
        s = tsr.sample_xyzrpy()

        Bw = numpy.array(Bw)
        self.assertTrue(numpy.all(s >= Bw[:, 0]))
        self.assertTrue(numpy.all(s <= Bw[:, 1]))

    def test_tsr_creation(self):
        """Test basic TSR creation."""
        T0_w = numpy.eye(4)
        Tw_e = numpy.eye(4)
        Bw = numpy.zeros((6, 2))
        Bw[2, :] = [0.0, 0.02]  # Allow vertical movement
        Bw[5, :] = [-pi, pi]    # Allow any yaw rotation
        
        tsr = TSR(T0_w=T0_w, Tw_e=Tw_e, Bw=Bw)
        
        self.assertIsInstance(tsr.T0_w, numpy.ndarray)
        self.assertIsInstance(tsr.Tw_e, numpy.ndarray)
        self.assertIsInstance(tsr.Bw, numpy.ndarray)
        self.assertEqual(tsr.T0_w.shape, (4, 4))
        self.assertEqual(tsr.Tw_e.shape, (4, 4))
        self.assertEqual(tsr.Bw.shape, (6, 2))

    def test_tsr_sampling(self):
        """Test TSR sampling functionality."""
        T0_w = numpy.eye(4)
        Tw_e = numpy.eye(4)
        Bw = numpy.zeros((6, 2))
        Bw[2, :] = [0.0, 0.02]  # Allow vertical movement
        Bw[5, :] = [-pi, pi]    # Allow any yaw rotation
        
        tsr = TSR(T0_w=T0_w, Tw_e=Tw_e, Bw=Bw)
        
        # Test sampling
        pose = tsr.sample()
        self.assertIsInstance(pose, numpy.ndarray)
        self.assertEqual(pose.shape, (4, 4))
        
        # Test xyzrpy sampling
        xyzrpy = tsr.sample_xyzrpy()
        self.assertIsInstance(xyzrpy, numpy.ndarray)
        self.assertEqual(xyzrpy.shape, (6,))

    def test_tsr_validation(self):
        """Test TSR validation."""
        T0_w = numpy.eye(4)
        Tw_e = numpy.eye(4)
        Bw = numpy.zeros((6, 2))
        Bw[2, :] = [0.0, 0.02]  # Allow vertical movement
        Bw[5, :] = [-pi, pi]    # Allow any yaw rotation
        
        tsr = TSR(T0_w=T0_w, Tw_e=Tw_e, Bw=Bw)
        
        # Test valid xyzrpy
        valid_xyzrpy = numpy.array([0.0, 0.0, 0.01, 0.0, 0.0, 0.0])
        self.assertTrue(all(tsr.is_valid(valid_xyzrpy)))
        
        # Test invalid xyzrpy (outside bounds)
        invalid_xyzrpy = numpy.array([0.0, 0.0, 0.1, 0.0, 0.0, 0.0])  # z too large
        self.assertFalse(all(tsr.is_valid(invalid_xyzrpy)))

    def test_tsr_contains(self):
        """Test TSR containment checking."""
        T0_w = numpy.eye(4)
        Tw_e = numpy.eye(4)
        Bw = numpy.zeros((6, 2))
        Bw[2, :] = [0.0, 0.02]  # Allow vertical movement
        Bw[5, :] = [-pi, pi]    # Allow any yaw rotation
        
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
        Bw[2, :] = [0.0, 0.02]  # Allow vertical movement
        Bw[5, :] = [-pi, pi]    # Allow any yaw rotation

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

        Bw = numpy.array([
            [-0.1, 0.1],   # X bounds
            [-0.1, 0.1],   # Y bounds
            [-0.1, 0.1],   # Z bounds
            [-pi/4, pi/4], # roll bounds
            [-pi/4, pi/4], # pitch bounds
            [-pi/4, pi/4]  # yaw bounds
        ])

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
        T0_w = numpy.array([
            [0, -1, 0, 0.5],
            [1,  0, 0, 0.5],
            [0,  0, 1, 0],
            [0,  0, 0, 1]
        ], dtype=float)

        # Tw_e with rotation and translation
        Tw_e = numpy.array([
            [0, 0, 1, 0.1],
            [1, 0, 0, 0],
            [0, 1, 0, 0.05],
            [0, 0, 0, 1]
        ], dtype=float)

        Bw = numpy.array([
            [-0.05, 0.05],
            [-0.05, 0.05],
            [-0.05, 0.05],
            [-pi/6, pi/6],
            [-pi/6, pi/6],
            [-pi/6, pi/6]
        ])

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
        self.assertEqual(distance, 0.0)
        self.assertTrue(tsr.contains(edge_transform))

    def test_contains_distance_consistency(self):
        """Test that contains() and distance() are consistent.

        For any transform:
        - contains(t) == True  implies distance(t) == 0
        - contains(t) == False implies distance(t) > 0
        """
        # Use non-trivial frames to catch frame-handling bugs
        T0_w = numpy.array([
            [1, 0, 0, 0.3],
            [0, 1, 0, -0.2],
            [0, 0, 1, 0.1],
            [0, 0, 0, 1]
        ], dtype=float)

        Tw_e = numpy.array([
            [0, -1, 0, 0],
            [1,  0, 0, 0.1],
            [0,  0, 1, 0],
            [0,  0, 0, 1]
        ], dtype=float)

        Bw = numpy.array([
            [-0.1, 0.1],
            [-0.1, 0.1],
            [-0.1, 0.1],
            [-pi/4, pi/4],
            [-pi/4, pi/4],
            [-pi/4, pi/4]
        ])

        tsr = TSR(T0_w=T0_w, Tw_e=Tw_e, Bw=Bw)

        # Test multiple random samples - all should be contained with distance 0
        for _ in range(10):
            sample_bw = tsr.sample_xyzrpy()
            sample_transform = tsr.to_transform(sample_bw)

            is_contained = tsr.contains(sample_transform)
            distance, _ = tsr.distance(sample_transform)

            self.assertTrue(is_contained,
                f"Sampled transform should be contained, bw={sample_bw}")
            self.assertEqual(distance, 0.0,
                f"Distance should be 0 for contained transform, bw={sample_bw}")

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
                self.assertGreater(distance, 0.0,
                    "Non-contained transform should have positive distance")

    def test_roundtrip_to_transform_to_xyzrpy(self):
        """Test that to_transform and to_xyzrpy are inverses.

        This validates the frame transformations are consistent.
        """
        T0_w = numpy.array([
            [0, 0, 1, 1.0],
            [0, 1, 0, 0],
            [-1, 0, 0, 0.5],
            [0, 0, 0, 1]
        ], dtype=float)

        Tw_e = numpy.eye(4)
        Tw_e[0, 3] = 0.2

        Bw = numpy.array([
            [-0.1, 0.1],
            [-0.1, 0.1],
            [-0.1, 0.1],
            [-pi/4, pi/4],
            [-pi/4, pi/4],
            [-pi/4, pi/4]
        ])

        tsr = TSR(T0_w=T0_w, Tw_e=Tw_e, Bw=Bw)

        # Test round-trip: xyzrpy -> transform -> xyzrpy
        original_bw = numpy.array([0.05, -0.03, 0.02, 0.1, -0.1, 0.2])
        transform = tsr.to_transform(original_bw)
        recovered_bw = tsr.to_xyzrpy(transform)

        numpy.testing.assert_array_almost_equal(
            original_bw, recovered_bw, decimal=10,
            err_msg="Round-trip xyzrpy -> transform -> xyzrpy failed"
        )

    def test_outer_interval_bounds(self):
        """Test TSR with outer interval RPY bounds (wrapping around ±pi).

        An outer interval like [3*pi/4, -3*pi/4] for yaw means yaw values
        in the 'back hemisphere' (|yaw| > 3*pi/4). This tests that such
        intervals are handled correctly.
        """
        # Outer interval for yaw: values near ±pi (back hemisphere)
        Bw = numpy.array([
            [-0.1, 0.1],
            [-0.1, 0.1],
            [-0.1, 0.1],
            [-pi/4, pi/4],
            [-pi/4, pi/4],
            [3*pi/4, -3*pi/4]  # Outer interval: |yaw| > 3*pi/4
        ])

        tsr = TSR(Bw=Bw)

        # Verify _Bw_cont has correct interval size (pi/2, not negative)
        yaw_interval = tsr._Bw_cont[5, 1] - tsr._Bw_cont[5, 0]
        self.assertGreater(yaw_interval, 0,
            "Outer interval should produce positive continuous interval")
        self.assertAlmostEqual(yaw_interval, pi/2, places=10,
            msg="Outer interval [3*pi/4, -3*pi/4] should have size pi/2")

        # Test sampling produces values in the outer interval
        for _ in range(10):
            sample = tsr.sample_xyzrpy()
            yaw = sample[5]
            self.assertTrue(abs(yaw) > 3*pi/4 - 0.01,
                f"Sampled yaw {yaw} should be in outer interval (|yaw| > 3*pi/4)")

        # Test contains: transform with yaw near pi should be contained
        valid_bw = numpy.array([0, 0, 0, 0, 0, 0.9*pi])
        trans_in = tsr.to_transform(valid_bw)
        self.assertTrue(tsr.contains(trans_in),
            "Transform with yaw=0.9*pi should be in outer interval")

        # Test contains: transform with yaw=0 should NOT be contained
        # (yaw=0 is not in the outer interval [3*pi/4, -3*pi/4])
        trans_yaw0 = numpy.eye(4)
        self.assertFalse(tsr.contains(trans_yaw0),
            "Transform with yaw=0 should NOT be in outer interval")

        # Test distance consistency for outer interval
        distance, _ = tsr.distance(trans_in)
        self.assertEqual(distance, 0.0,
            "Distance should be 0 for contained transform in outer interval")
