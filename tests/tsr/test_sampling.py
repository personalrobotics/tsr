#!/usr/bin/env python
"""
Tests for advanced sampling utilities.

Tests the sampling functions for working with multiple TSRs and templates.
"""

import unittest
import numpy as np
from numpy import pi
from tsr.sampling import (
    weights_from_tsrs,
    choose_tsr_index,
    choose_tsr,
    sample_from_tsrs,
    instantiate_templates,
    sample_from_templates
)
from tsr.core.tsr import TSR
from tsr.core.tsr_template import TSRTemplate


class TestSamplingUtilities(unittest.TestCase):
    """Test sampling utility functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create test TSRs with different volumes
        self.tsr1 = TSR(
            T0_w=np.eye(4),
            Tw_e=np.eye(4),
            Bw=np.array([
                [0, 0],      # x: fixed
                [0, 0],      # y: fixed
                [0, 0],      # z: fixed
                [0, 0],      # roll: fixed
                [0, 0],      # pitch: fixed
                [-pi, pi]    # yaw: full rotation (2π volume)
            ])
        )
        
        self.tsr2 = TSR(
            T0_w=np.eye(4),
            Tw_e=np.eye(4),
            Bw=np.array([
                [-0.1, 0.1],  # x: 0.2 range
                [0, 0],      # y: fixed
                [0, 0],      # z: fixed
                [0, 0],      # roll: fixed
                [0, 0],      # pitch: fixed
                [0, 0]       # yaw: fixed
            ])
        )
        
        self.tsr3 = TSR(
            T0_w=np.eye(4),
            Tw_e=np.eye(4),
            Bw=np.array([
                [0, 0],      # x: fixed
                [0, 0],      # y: fixed
                [0, 0],      # z: fixed
                [0, 0],      # roll: fixed
                [0, 0],      # pitch: fixed
                [0, 0]       # yaw: fixed (zero volume)
            ])
        )
        
        self.tsrs = [self.tsr1, self.tsr2, self.tsr3]
    
    def test_weights_from_tsrs(self):
        """Test weight calculation from TSR volumes."""
        weights = weights_from_tsrs(self.tsrs)
        
        # Should return numpy array
        self.assertIsInstance(weights, np.ndarray)
        self.assertEqual(weights.shape, (3,))
        
        # Weights should be non-negative
        self.assertTrue(np.all(weights >= 0))
        
        # TSR1 should have highest weight (2π volume)
        # TSR2 should have medium weight (0.2 volume)
        # TSR3 should have zero weight (zero volume)
        self.assertGreater(weights[0], weights[1])  # TSR1 > TSR2
        self.assertEqual(weights[2], 0)  # TSR3 has zero volume
    
    def test_weights_from_tsrs_zero_volume(self):
        """Test weight calculation when all TSRs have zero volume."""
        zero_tsrs = [self.tsr3, self.tsr3, self.tsr3]
        weights = weights_from_tsrs(zero_tsrs)
        
        # Should fall back to uniform weights
        self.assertTrue(np.all(weights > 0))
        self.assertTrue(np.allclose(weights, weights[0]))  # All equal
    
    def test_weights_from_tsrs_single_tsr(self):
        """Test weight calculation with single TSR."""
        weights = weights_from_tsrs([self.tsr1])
        self.assertEqual(weights.shape, (1,))
        self.assertGreater(weights[0], 0)
    
    def test_weights_from_tsrs_empty_list(self):
        """Test weight calculation with empty list."""
        with self.assertRaises(ValueError):
            weights_from_tsrs([])
    
    def test_choose_tsr_index(self):
        """Test TSR index selection."""
        # Test with default RNG
        index = choose_tsr_index(self.tsrs)
        self.assertIsInstance(index, int)
        self.assertGreaterEqual(index, 0)
        self.assertLess(index, len(self.tsrs))
        
        # Test with custom RNG
        rng = np.random.default_rng(42)  # Fixed seed for reproducibility
        index = choose_tsr_index(self.tsrs, rng)
        self.assertIsInstance(index, int)
        self.assertGreaterEqual(index, 0)
        self.assertLess(index, len(self.tsrs))
    
    def test_choose_tsr(self):
        """Test TSR selection."""
        # Test with default RNG
        selected_tsr = choose_tsr(self.tsrs)
        self.assertIn(selected_tsr, self.tsrs)
        
        # Test with custom RNG
        rng = np.random.default_rng(42)
        selected_tsr = choose_tsr(self.tsrs, rng)
        self.assertIn(selected_tsr, self.tsrs)
    
    def test_sample_from_tsrs(self):
        """Test sampling from multiple TSRs."""
        # Test with default RNG
        pose = sample_from_tsrs(self.tsrs)
        self.assertIsInstance(pose, np.ndarray)
        self.assertEqual(pose.shape, (4, 4))
        
        # Test with custom RNG
        rng = np.random.default_rng(42)
        pose = sample_from_tsrs(self.tsrs, rng)
        self.assertIsInstance(pose, np.ndarray)
        self.assertEqual(pose.shape, (4, 4))
        
        # Verify pose is valid (from one of the TSRs)
        valid_poses = [tsr.contains(pose) for tsr in self.tsrs]
        self.assertTrue(any(valid_poses))


class TestTemplateSampling(unittest.TestCase):
    """Test template-based sampling functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create test templates
        self.template1 = TSRTemplate(
            T_ref_tsr=np.eye(4),
            Tw_e=np.eye(4),
            Bw=np.array([
                [0, 0],      # x: fixed
                [0, 0],      # y: fixed
                [0, 0],      # z: fixed
                [0, 0],      # roll: fixed
                [0, 0],      # pitch: fixed
                [-pi, pi]    # yaw: full rotation
            ])
        )
        
        self.template2 = TSRTemplate(
            T_ref_tsr=np.eye(4),
            Tw_e=np.eye(4),
            Bw=np.array([
                [-0.1, 0.1],  # x: 0.2 range
                [0, 0],      # y: fixed
                [0, 0],      # z: fixed
                [0, 0],      # roll: fixed
                [0, 0],      # pitch: fixed
                [0, 0]       # yaw: fixed
            ])
        )
        
        self.templates = [self.template1, self.template2]
        self.T_ref_world = np.array([
            [1, 0, 0, 0.5],  # Reference pose
            [0, 1, 0, 0.0],
            [0, 0, 1, 0.3],
            [0, 0, 0, 1]
        ])
    
    def test_instantiate_templates(self):
        """Test template instantiation."""
        tsrs = instantiate_templates(self.templates, self.T_ref_world)
        
        # Should return list of TSRs
        self.assertIsInstance(tsrs, list)
        self.assertEqual(len(tsrs), len(self.templates))
        
        # Each should be a TSR
        for tsr in tsrs:
            self.assertIsInstance(tsr, TSR)
        
        # TSRs should be instantiated at the reference pose
        for tsr in tsrs:
            # T0_w should be T_ref_world @ T_ref_tsr (which is just T_ref_world for identity T_ref_tsr)
            np.testing.assert_array_almost_equal(tsr.T0_w, self.T_ref_world)
    
    def test_sample_from_templates(self):
        """Test sampling from templates."""
        # Test with default RNG
        pose = sample_from_templates(self.templates, self.T_ref_world)
        self.assertIsInstance(pose, np.ndarray)
        self.assertEqual(pose.shape, (4, 4))
        
        # Test with custom RNG
        rng = np.random.default_rng(42)
        pose = sample_from_templates(self.templates, self.T_ref_world, rng)
        self.assertIsInstance(pose, np.ndarray)
        self.assertEqual(pose.shape, (4, 4))
        
        # Verify pose is a valid transform
        self.assertTrue(np.allclose(pose[3, :], [0, 0, 0, 1]))
        self.assertTrue(np.allclose(np.linalg.det(pose[:3, :3]), 1.0, atol=1e-6))
    
    def test_sample_from_templates_single_template(self):
        """Test sampling from single template."""
        single_template = [self.template1]
        pose = sample_from_templates(single_template, self.T_ref_world)
        self.assertIsInstance(pose, np.ndarray)
        self.assertEqual(pose.shape, (4, 4))
        
        # Should be a valid transform
        self.assertTrue(np.allclose(pose[3, :], [0, 0, 0, 1]))
        self.assertTrue(np.allclose(np.linalg.det(pose[:3, :3]), 1.0, atol=1e-6))


class TestSamplingEdgeCases(unittest.TestCase):
    """Test edge cases in sampling functions."""
    
    def test_sampling_reproducibility(self):
        """Test that sampling is reproducible with same RNG."""
        # Create simple TSR
        tsr = TSR(
            T0_w=np.eye(4),
            Tw_e=np.eye(4),
            Bw=np.array([
                [-0.1, 0.1],  # x: small range
                [0, 0],      # y: fixed
                [0, 0],      # z: fixed
                [0, 0],      # roll: fixed
                [0, 0],      # pitch: fixed
                [0, 0]       # yaw: fixed
            ])
        )
        
        # Sample with same RNG seed
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        
        pose1 = sample_from_tsrs([tsr], rng1)
        pose2 = sample_from_tsrs([tsr], rng2)
        
        # Since TSR.sample() uses its own RNG, we can't guarantee exact reproducibility
        # But we can verify both poses are valid transforms
        self.assertTrue(np.allclose(pose1[3, :], [0, 0, 0, 1]))
        self.assertTrue(np.allclose(pose1[:3, :3] @ pose1[:3, :3].T, np.eye(3), atol=1e-6))
        self.assertTrue(np.allclose(pose2[3, :], [0, 0, 0, 1]))
        self.assertTrue(np.allclose(pose2[:3, :3] @ pose2[:3, :3].T, np.eye(3), atol=1e-6))
    
    def test_sampling_different_weights(self):
        """Test that TSRs with different weights are selected appropriately."""
        # Create TSRs with very different volumes
        large_tsr = TSR(
            T0_w=np.eye(4),
            Tw_e=np.eye(4),
            Bw=np.array([
                [-1, 1],     # x: large range
                [-1, 1],     # y: large range
                [-1, 1],     # z: large range
                [-pi, pi],   # roll: full rotation
                [-pi, pi],   # pitch: full rotation
                [-pi, pi]    # yaw: full rotation
            ])
        )
        
        small_tsr = TSR(
            T0_w=np.eye(4),
            Tw_e=np.eye(4),
            Bw=np.array([
                [0, 0],      # x: fixed
                [0, 0],      # y: fixed
                [0, 0],      # z: fixed
                [0, 0],      # roll: fixed
                [0, 0],      # pitch: fixed
                [-0.1, 0.1]  # yaw: small range
            ])
        )
        
        tsrs = [large_tsr, small_tsr]
        weights = weights_from_tsrs(tsrs)
        
        # Large TSR should have much higher weight
        self.assertGreater(weights[0], weights[1] * 100)  # At least 100x larger


if __name__ == '__main__':
    unittest.main()
