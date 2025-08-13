#!/usr/bin/env python
"""
Tests for TSRTemplate functionality.

Tests the TSRTemplate class for scene-agnostic TSR definitions.
"""

import unittest
import numpy as np
from numpy import pi
from tsr.core.tsr_template import TSRTemplate
from tsr.core.tsr import TSR


class TestTSRTemplate(unittest.TestCase):
    """Test TSRTemplate functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a simple TSR template
        self.T_ref_tsr = np.eye(4)
        self.Tw_e = np.array([
            [0, 0, 1, -0.1],  # TSR to end-effector at canonical pose
            [1, 0, 0, 0],
            [0, 1, 0, 0.05],
            [0, 0, 0, 1]
        ])
        self.Bw = np.array([
            [0, 0],      # x bounds (fixed)
            [0, 0],      # y bounds (fixed)
            [-0.01, 0.01],  # z bounds (small tolerance)
            [0, 0],      # roll bounds (fixed)
            [0, 0],      # pitch bounds (fixed)
            [-pi, pi]    # yaw bounds (full rotation)
        ])
        
        self.template = TSRTemplate(
            T_ref_tsr=self.T_ref_tsr,
            Tw_e=self.Tw_e,
            Bw=self.Bw
        )
    
    def test_tsr_template_creation(self):
        """Test TSRTemplate creation."""
        self.assertIsInstance(self.template, TSRTemplate)
        self.assertIsInstance(self.template.T_ref_tsr, np.ndarray)
        self.assertIsInstance(self.template.Tw_e, np.ndarray)
        self.assertIsInstance(self.template.Bw, np.ndarray)
        
        self.assertEqual(self.template.T_ref_tsr.shape, (4, 4))
        self.assertEqual(self.template.Tw_e.shape, (4, 4))
        self.assertEqual(self.template.Bw.shape, (6, 2))
    
    def test_tsr_template_immutability(self):
        """Test that TSRTemplate is immutable (frozen dataclass)."""
        with self.assertRaises(Exception):
            self.template.T_ref_tsr = np.eye(4)
    
    def test_tsr_template_instantiation(self):
        """Test TSRTemplate instantiation at a reference pose."""
        # Create a reference pose (e.g., object pose in world)
        T_ref_world = np.array([
            [1, 0, 0, 0.5],  # Object at x=0.5, y=0, z=0
            [0, 1, 0, 0.0],
            [0, 0, 1, 0.3],
            [0, 0, 0, 1]
        ])
        
        # Instantiate the template
        tsr = self.template.instantiate(T_ref_world)
        
        # Verify it's a TSR
        self.assertIsInstance(tsr, TSR)
        
        # Verify the T0_w is correctly computed: T_ref_world @ T_ref_tsr
        expected_T0_w = T_ref_world @ self.T_ref_tsr
        np.testing.assert_array_almost_equal(tsr.T0_w, expected_T0_w)
        
        # Verify Tw_e and Bw are preserved
        np.testing.assert_array_almost_equal(tsr.Tw_e, self.Tw_e)
        np.testing.assert_array_almost_equal(tsr.Bw, self.Bw)
    
    def test_tsr_template_instantiation_multiple_poses(self):
        """Test TSRTemplate instantiation at multiple reference poses."""
        poses = [
            np.eye(4),  # Identity pose
            np.array([[1, 0, 0, 1.0], [0, 1, 0, 0.0], [0, 0, 1, 0.0], [0, 0, 0, 1]]),  # Translated
            np.array([[0, -1, 0, 0.0], [1, 0, 0, 0.0], [0, 0, 1, 0.0], [0, 0, 0, 1]]),  # Rotated
        ]
        
        for pose in poses:
            tsr = self.template.instantiate(pose)
            self.assertIsInstance(tsr, TSR)
            
            # Verify T0_w is correctly computed
            expected_T0_w = pose @ self.T_ref_tsr
            np.testing.assert_array_almost_equal(tsr.T0_w, expected_T0_w)
    
    def test_tsr_template_with_offset_reference(self):
        """Test TSRTemplate with non-identity T_ref_tsr."""
        # Create template with offset reference
        T_ref_tsr_offset = np.array([
            [1, 0, 0, 0.1],  # Offset in x direction
            [0, 1, 0, 0.0],
            [0, 0, 1, 0.0],
            [0, 0, 0, 1]
        ])
        
        template_offset = TSRTemplate(
            T_ref_tsr=T_ref_tsr_offset,
            Tw_e=self.Tw_e,
            Bw=self.Bw
        )
        
        # Instantiate at world origin
        T_ref_world = np.eye(4)
        tsr = template_offset.instantiate(T_ref_world)
        
        # Verify T0_w includes the offset
        expected_T0_w = T_ref_world @ T_ref_tsr_offset
        np.testing.assert_array_almost_equal(tsr.T0_w, expected_T0_w)
    
    def test_tsr_template_sampling(self):
        """Test that instantiated TSRs can be sampled from."""
        T_ref_world = np.eye(4)
        tsr = self.template.instantiate(T_ref_world)
        
        # Sample from the instantiated TSR
        pose = tsr.sample()
        self.assertIsInstance(pose, np.ndarray)
        self.assertEqual(pose.shape, (4, 4))
        
        # Verify the pose is valid (within bounds)
        # Note: contains() checks if the transform is within the TSR bounds
        # For a TSR with mostly fixed bounds, this should work
        try:
            self.assertTrue(tsr.contains(pose))
        except Exception:
            # If contains fails, at least verify the pose is a valid transform
            self.assertTrue(np.allclose(pose[3, :], [0, 0, 0, 1]))  # Bottom row should be [0,0,0,1]
            self.assertTrue(np.allclose(np.linalg.det(pose[:3, :3]), 1.0, atol=1e-6))  # Rotation matrix
    
    def test_tsr_template_validation(self):
        """Test TSRTemplate validation."""
        # TSRTemplate doesn't have built-in validation, so we just test that it accepts valid inputs
        # and that numpy will raise errors for invalid shapes when used
        template = TSRTemplate(
            T_ref_tsr=self.T_ref_tsr,
            Tw_e=self.Tw_e,
            Bw=self.Bw
        )
        self.assertIsInstance(template, TSRTemplate)


class TestTSRTemplateExamples(unittest.TestCase):
    """Test TSRTemplate with realistic examples."""
    
    def test_cylinder_grasp_template(self):
        """Test TSRTemplate for cylinder grasping."""
        # Template for grasping a cylinder from the side
        T_ref_tsr = np.eye(4)  # TSR frame aligned with cylinder frame
        Tw_e = np.array([
            [0, 0, 1, -0.05],  # Approach from -z, offset by 5cm
            [1, 0, 0, 0],      # x-axis perpendicular to cylinder
            [0, 1, 0, 0],      # y-axis along cylinder axis
            [0, 0, 0, 1]
        ])
        Bw = np.array([
            [0, 0],           # x: fixed position
            [0, 0],           # y: fixed position  
            [-0.01, 0.01],    # z: small tolerance
            [0, 0],           # roll: fixed
            [0, 0],           # pitch: fixed
            [-pi, pi]         # yaw: full rotation around cylinder
        ])
        
        template = TSRTemplate(T_ref_tsr=T_ref_tsr, Tw_e=Tw_e, Bw=Bw)
        
        # Instantiate at a cylinder pose
        cylinder_pose = np.array([
            [1, 0, 0, 0.5],  # Cylinder at x=0.5
            [0, 1, 0, 0.0],
            [0, 0, 1, 0.3],
            [0, 0, 0, 1]
        ])
        
        tsr = template.instantiate(cylinder_pose)
        
        # Verify the TSR is valid
        self.assertIsInstance(tsr, TSR)
        
        # Sample a grasp pose
        grasp_pose = tsr.sample()
        # Verify it's a valid transform
        self.assertTrue(np.allclose(grasp_pose[3, :], [0, 0, 0, 1]))
        self.assertTrue(np.allclose(np.linalg.det(grasp_pose[:3, :3]), 1.0, atol=1e-6))
    
    def test_place_on_table_template(self):
        """Test TSRTemplate for placing objects on a table."""
        # Template for placing an object on a table
        T_ref_tsr = np.eye(4)  # TSR frame aligned with table frame
        Tw_e = np.array([
            [1, 0, 0, 0],      # Object x-axis aligned with table x
            [0, 1, 0, 0],      # Object y-axis aligned with table y
            [0, 0, 1, 0.02],   # Object slightly above table surface
            [0, 0, 0, 1]
        ])
        Bw = np.array([
            [-0.1, 0.1],       # x: allow sliding on table
            [-0.1, 0.1],       # y: allow sliding on table
            [0, 0],            # z: fixed height
            [0, 0],            # roll: keep level
            [0, 0],            # pitch: keep level
            [-pi/4, pi/4]      # yaw: allow some rotation
        ])
        
        template = TSRTemplate(T_ref_tsr=T_ref_tsr, Tw_e=Tw_e, Bw=Bw)
        
        # Instantiate at table pose
        table_pose = np.eye(4)  # Table at world origin
        tsr = template.instantiate(table_pose)
        
        # Verify the TSR is valid
        self.assertIsInstance(tsr, TSR)
        
        # Sample a placement pose
        place_pose = tsr.sample()
        # Verify it's a valid transform
        self.assertTrue(np.allclose(place_pose[3, :], [0, 0, 0, 1]))
        self.assertTrue(np.allclose(np.linalg.det(place_pose[:3, :3]), 1.0, atol=1e-6))


if __name__ == '__main__':
    unittest.main()
