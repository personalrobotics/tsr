#!/usr/bin/env python
"""
Tests for MuJoCo wrapper functionality.
"""

import unittest
import numpy as np
from unittest.mock import Mock, patch
from numpy import pi

from tsr.wrappers.mujoco import MuJoCoRobotAdapter
from tsr.wrappers.mujoco.tsr import (
    cylinder_grasp,
    box_grasp,
    place_object,
    transport_upright
)
from tsr.core.tsr import TSR
from tsr.core.tsr_chain import TSRChain


class TestMuJoCoRobotAdapter(unittest.TestCase):
    """Test the MuJoCo robot adapter."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a mock MuJoCo robot
        self.mock_robot = Mock()
        self.robot_adapter = MuJoCoRobotAdapter(self.mock_robot, manip_idx=0)
    
    def test_initialization(self):
        """Test robot adapter initialization."""
        self.assertEqual(self.robot_adapter.get_primary_manipulator_index(), 0)
        self.assertEqual(self.robot_adapter.get_active_manipulator_index(), 0)
        self.assertEqual(self.robot_adapter.get_manipulator_count(), 1)
        self.assertEqual(self.robot_adapter.get_robot_name(), "mujoco_robot")
    
    def test_manipulator_management(self):
        """Test manipulator management."""
        # Test getting manipulator name
        self.assertEqual(self.robot_adapter.get_manipulator_name(0), "manipulator_0")
        
        # Test getting manipulator index
        self.assertEqual(self.robot_adapter.get_manipulator_index("manipulator_0"), 0)
        
        # Test setting active manipulator
        self.robot_adapter.set_active_manipulator(0)
        self.assertEqual(self.robot_adapter.get_active_manipulator_index(), 0)
        
        # Test invalid manipulator index
        with self.assertRaises(ValueError):
            self.robot_adapter.get_manipulator_name(1)
        
        with self.assertRaises(ValueError):
            self.robot_adapter.set_active_manipulator(1)
    
    def test_transform_methods(self):
        """Test transform-related methods."""
        # These methods return placeholder values for now
        transform = self.robot_adapter.get_manipulator_transform(0)
        self.assertEqual(transform.shape, (4, 4))
        np.testing.assert_array_equal(transform, np.eye(4))
        
        obj_transform = self.robot_adapter.get_object_transform("test_object")
        self.assertEqual(obj_transform.shape, (4, 4))
        np.testing.assert_array_equal(obj_transform, np.eye(4))
    
    def test_object_methods(self):
        """Test object-related methods."""
        # Test getting object name
        mock_obj = Mock()
        obj_name = self.robot_adapter.get_object_name(mock_obj)
        self.assertEqual(obj_name, "unknown_object")
        
        # Test grabbing check
        is_grabbing = self.robot_adapter.is_manipulator_grabbing(0, "test_object")
        self.assertFalse(is_grabbing)
    
    def test_multi_arm_support(self):
        """Test multi-arm robot support."""
        # Add a second manipulator
        self.robot_adapter.add_manipulator(1, "manipulator_1")
        
        self.assertEqual(self.robot_adapter.get_manipulator_count(), 2)
        self.assertEqual(self.robot_adapter.get_manipulator_name(1), "manipulator_1")
        self.assertEqual(self.robot_adapter.get_manipulator_index("manipulator_1"), 1)
        
        # Test switching between manipulators
        self.robot_adapter.set_active_manipulator(1)
        self.assertEqual(self.robot_adapter.get_active_manipulator_index(), 1)


class TestMuJoCoTSRFunctions(unittest.TestCase):
    """Test MuJoCo-specific TSR functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a mock robot adapter
        self.robot_adapter = MuJoCoRobotAdapter(Mock(), manip_idx=0)
        
        # Mock the transform methods to return realistic values
        self.robot_adapter.get_object_transform = Mock(return_value=np.eye(4))
        self.robot_adapter.get_manipulator_transform = Mock(return_value=np.eye(4))
        self.robot_adapter.get_object_name = Mock(return_value="test_object")
        self.robot_adapter.is_manipulator_grabbing = Mock(return_value=True)
        
        # Create a mock object
        self.mock_obj = Mock()
    
    def test_cylinder_grasp(self):
        """Test cylinder grasp function."""
        chains = cylinder_grasp(
            self.robot_adapter,
            self.mock_obj,
            obj_radius=0.05,
            obj_height=0.1
        )
        
        self.assertIsInstance(chains, list)
        self.assertEqual(len(chains), 2)  # Two orientations
        
        for chain in chains:
            self.assertIsInstance(chain, TSRChain)
            self.assertEqual(len(chain.TSRs), 1)
            self.assertIsInstance(chain.TSRs[0], TSR)
    
    def test_cylinder_grasp_with_manip_idx(self):
        """Test cylinder grasp with specific manipulator index."""
        chains = cylinder_grasp(
            self.robot_adapter,
            self.mock_obj,
            obj_radius=0.05,
            obj_height=0.1,
            manip_idx=1
        )
        
        self.assertIsInstance(chains, list)
        self.assertEqual(len(chains), 2)
    
    def test_box_grasp(self):
        """Test box grasp function."""
        chains = box_grasp(
            self.robot_adapter,
            self.mock_obj,
            length=0.1,
            width=0.05,
            height=0.03
        )
        
        self.assertIsInstance(chains, list)
        # Box grasp should return 12 chains (6 faces × 2 orientations)
        self.assertEqual(len(chains), 12)
        
        for chain in chains:
            self.assertIsInstance(chain, TSRChain)
            self.assertEqual(len(chain.TSRs), 1)
            self.assertIsInstance(chain.TSRs[0], TSR)
    
    def test_place_object(self):
        """Test place object function."""
        # Create a mock pose TSR chain
        pose_tsr = TSR(T0_w=np.eye(4), Tw_e=np.eye(4), Bw=np.zeros((6, 2)))
        pose_chain = TSRChain(sample_start=False, sample_goal=True, constrain=False, TSR=pose_tsr)
        
        chains = place_object(
            self.robot_adapter,
            self.mock_obj,
            pose_chain
        )
        
        self.assertIsInstance(chains, list)
        self.assertEqual(len(chains), 1)
        
        chain = chains[0]
        self.assertIsInstance(chain, TSRChain)
        self.assertEqual(len(chain.TSRs), 2)  # Pose TSR + grasp TSR
    
    def test_transport_upright(self):
        """Test transport upright function."""
        chains = transport_upright(
            self.robot_adapter,
            self.mock_obj,
            roll_epsilon=0.1,
            pitch_epsilon=0.1,
            yaw_epsilon=0.1
        )
        
        self.assertIsInstance(chains, list)
        self.assertEqual(len(chains), 1)
        
        chain = chains[0]
        self.assertIsInstance(chain, TSRChain)
        self.assertEqual(len(chain.TSRs), 1)
        self.assertTrue(chain.constrain)  # Should be trajectory constraint
        self.assertFalse(chain.sample_start)
        self.assertFalse(chain.sample_goal)
    
    def test_invalid_parameters(self):
        """Test error handling for invalid parameters."""
        # Test invalid cylinder parameters
        with self.assertRaises(ValueError):
            cylinder_grasp(self.robot_adapter, self.mock_obj, obj_radius=-0.1, obj_height=0.1)
        
        with self.assertRaises(ValueError):
            cylinder_grasp(self.robot_adapter, self.mock_obj, obj_radius=0.1, obj_height=-0.1)
        
        # Test invalid box parameters
        with self.assertRaises(ValueError):
            box_grasp(self.robot_adapter, self.mock_obj, length=-0.1, width=0.05, height=0.03)
        
        # Test invalid transport parameters
        with self.assertRaises(ValueError):
            transport_upright(self.robot_adapter, self.mock_obj, roll_epsilon=-0.1)


if __name__ == '__main__':
    unittest.main() 