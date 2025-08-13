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
