#!/usr/bin/env python
"""
Equivalence tests between old and new TSR implementations.

These tests ensure that the refactored TSR implementation produces
exactly the same results as the original implementation.
"""

import numpy as np
import unittest
from numpy import pi
import random

# Import both old and new implementations
from tsr.tsr import TSR as LegacyTSR
from tsr.core.tsr import TSR as CoreTSR


class TestTSEquivalence(unittest.TestCase):
    """Test that new TSR implementation is equivalent to legacy implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Set random seed for reproducible tests
        np.random.seed(42)
        random.seed(42)
        
        # Common test parameters
        self.T0_w = np.array([
            [1, 0, 0, 0.1],
            [0, 1, 0, 0.2],
            [0, 0, 1, 0.3],
            [0, 0, 0, 1]
        ])
        
        self.Tw_e = np.array([
            [0, 0, 1, 0.05],
            [1, 0, 0, 0],
            [0, 1, 0, 0.1],
            [0, 0, 0, 1]
        ])
        
        self.Bw = np.array([
            [-0.01, 0.01],  # x bounds
            [-0.01, 0.01],  # y bounds
            [-0.01, 0.01],  # z bounds
            [-pi/4, pi/4],   # roll bounds
            [-pi/4, pi/4],   # pitch bounds
            [-pi/2, pi/2]    # yaw bounds
        ])
    
    def test_tsr_creation_equivalence(self):
        """Test that TSR creation produces identical objects."""
        legacy_tsr = LegacyTSR(T0_w=self.T0_w, Tw_e=self.Tw_e, Bw=self.Bw)
        core_tsr = CoreTSR(T0_w=self.T0_w, Tw_e=self.Tw_e, Bw=self.Bw)
        
        # Check that all attributes are identical
        np.testing.assert_array_almost_equal(legacy_tsr.T0_w, core_tsr.T0_w)
        np.testing.assert_array_almost_equal(legacy_tsr.Tw_e, core_tsr.Tw_e)
        np.testing.assert_array_almost_equal(legacy_tsr.Bw, core_tsr.Bw)
        np.testing.assert_array_almost_equal(legacy_tsr._Bw_cont, core_tsr._Bw_cont)
    
    def test_sampling_equivalence(self):
        """Test that sampling produces identical results with same seed."""
        legacy_tsr = LegacyTSR(T0_w=self.T0_w, Tw_e=self.Tw_e, Bw=self.Bw)
        core_tsr = CoreTSR(T0_w=self.T0_w, Tw_e=self.Tw_e, Bw=self.Bw)
        
        # Test multiple samples
        for i in range(10):
            np.random.seed(42 + i)
            legacy_sample = legacy_tsr.sample_xyzrpy()
            
            np.random.seed(42 + i)
            core_sample = core_tsr.sample_xyzrpy()
            
            np.testing.assert_array_almost_equal(legacy_sample, core_sample)
    
    def test_transform_equivalence(self):
        """Test that transform calculations are identical."""
        legacy_tsr = LegacyTSR(T0_w=self.T0_w, Tw_e=self.Tw_e, Bw=self.Bw)
        core_tsr = CoreTSR(T0_w=self.T0_w, Tw_e=self.Tw_e, Bw=self.Bw)
        
        # Test with various xyzrpy inputs (all valid within TSR bounds)
        test_inputs = [
            np.zeros(6),  # Valid: within all bounds
            np.array([0.005, 0.005, 0.005, pi/8, pi/8, pi/4]),  # Valid: within bounds
            np.array([-0.005, -0.005, -0.005, -pi/8, -pi/8, -pi/4])  # Valid: within bounds
        ]
        
        for xyzrpy in test_inputs:
            legacy_transform = legacy_tsr.to_transform(xyzrpy)
            core_transform = core_tsr.to_transform(xyzrpy)
            
            np.testing.assert_array_almost_equal(legacy_transform, core_transform)
    
    def test_distance_equivalence(self):
        """Test that distance calculations are identical."""
        legacy_tsr = LegacyTSR(T0_w=self.T0_w, Tw_e=self.Tw_e, Bw=self.Bw)
        core_tsr = CoreTSR(T0_w=self.T0_w, Tw_e=self.Tw_e, Bw=self.Bw)
        
        # Test with various transforms
        test_transforms = [
            np.eye(4),
            self.T0_w,
            self.Tw_e,
            np.array([
                [1, 0, 0, 0.5],
                [0, 1, 0, 0.5],
                [0, 0, 1, 0.5],
                [0, 0, 0, 1]
            ])
        ]
        
        for transform in test_transforms:
            legacy_result = legacy_tsr.distance(transform)
            core_result = core_tsr.distance(transform)
            
            # Both methods return (distance, bwopt)
            legacy_distance = legacy_result[0] if isinstance(legacy_result, tuple) else legacy_result
            core_distance = core_result[0] if isinstance(core_result, tuple) else core_result
            
            # Test distance equivalence (should be identical)
            self.assertAlmostEqual(legacy_distance, core_distance, places=10)
    
    def test_containment_equivalence(self):
        """Test that containment tests give identical results."""
        legacy_tsr = LegacyTSR(T0_w=self.T0_w, Tw_e=self.Tw_e, Bw=self.Bw)
        core_tsr = CoreTSR(T0_w=self.T0_w, Tw_e=self.Tw_e, Bw=self.Bw)
        
        # Test with transforms that should be contained (identity transform)
        contained_transform = np.eye(4)  # Identity transform is within bounds
        self.assertTrue(legacy_tsr.contains(contained_transform))
        self.assertTrue(core_tsr.contains(contained_transform))
        
        # Test with transforms that should not be contained
        outside_transform = np.array([
            [1, 0, 0, 10.0],  # Far outside bounds
            [0, 1, 0, 10.0],
            [0, 0, 1, 10.0],
            [0, 0, 0, 1]
        ])
        self.assertFalse(legacy_tsr.contains(outside_transform))
        self.assertFalse(core_tsr.contains(outside_transform))
        
        # Test with a small transform that should be contained
        small_transform = np.array([
            [1, 0, 0, 0.005],  # Small translation within bounds
            [0, 1, 0, 0.005],
            [0, 0, 1, 0.005],
            [0, 0, 0, 1]
        ])
        self.assertTrue(legacy_tsr.contains(small_transform))
        self.assertTrue(core_tsr.contains(small_transform))
    
    def test_edge_cases_equivalence(self):
        """Test edge cases work identically."""
        # Test with zero bounds
        zero_bounds = np.zeros((6, 2))
        legacy_tsr = LegacyTSR(T0_w=self.T0_w, Tw_e=self.Tw_e, Bw=zero_bounds)
        core_tsr = CoreTSR(T0_w=self.T0_w, Tw_e=self.Tw_e, Bw=zero_bounds)
        
        # Set random seed for reproducible sampling
        np.random.seed(42)
        legacy_sample = legacy_tsr.sample_xyzrpy()
        
        np.random.seed(42)  # Reset seed for core
        core_sample = core_tsr.sample_xyzrpy()
        
        np.testing.assert_array_almost_equal(legacy_sample, core_sample)
        
        # Test with wrapped angle bounds
        wrapped_bounds = self.Bw.copy()
        wrapped_bounds[3:6, 0] = [pi, pi/2, -pi]  # Roll, pitch, yaw
        wrapped_bounds[3:6, 1] = [3*pi, 3*pi/2, pi]
        
        legacy_tsr = LegacyTSR(T0_w=self.T0_w, Tw_e=self.Tw_e, Bw=wrapped_bounds)
        core_tsr = CoreTSR(T0_w=self.T0_w, Tw_e=self.Tw_e, Bw=wrapped_bounds)
        
        # Set random seed for reproducible sampling
        np.random.seed(43)  # Different seed for second test
        legacy_sample = legacy_tsr.sample_xyzrpy()
        
        np.random.seed(43)  # Reset seed for core
        core_sample = core_tsr.sample_xyzrpy()
        
        np.testing.assert_array_almost_equal(legacy_sample, core_sample)
    
    def test_validation_equivalence(self):
        """Test that validation errors are identical."""
        # Test invalid bounds (min > max)
        invalid_bounds = self.Bw.copy()
        invalid_bounds[0, 0] = 1.0  # min > max for x
        
        with self.assertRaises(ValueError):
            LegacyTSR(T0_w=self.T0_w, Tw_e=self.Tw_e, Bw=invalid_bounds)
        
        with self.assertRaises(ValueError):
            CoreTSR(T0_w=self.T0_w, Tw_e=self.Tw_e, Bw=invalid_bounds)
        
        # Test invalid xyzrpy input
        legacy_tsr = LegacyTSR(T0_w=self.T0_w, Tw_e=self.Tw_e, Bw=self.Bw)
        core_tsr = CoreTSR(T0_w=self.T0_w, Tw_e=self.Tw_e, Bw=self.Bw)
        
        with self.assertRaises(ValueError):
            legacy_tsr.to_transform(np.array([1, 2, 3, 4, 5]))  # Wrong length
        
        with self.assertRaises(ValueError):
            core_tsr.to_transform(np.array([1, 2, 3, 4, 5]))  # Wrong length


if __name__ == '__main__':
    unittest.main() 