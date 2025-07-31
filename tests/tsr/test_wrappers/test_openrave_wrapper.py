#!/usr/bin/env python
"""
Tests for the OpenRAVE wrapper implementation.

These tests ensure that the OpenRAVE wrapper correctly implements
the abstract robot interface and maintains compatibility with existing code.
"""

import numpy as np
import unittest
from unittest.mock import Mock, patch
from numpy import pi

# Import test fixtures
from fixtures.mock_robot import (
    MockRobot, MockKinBody, MockManipulator, 
    create_test_robot, create_test_object, setup_grasp_scenario
)

# Import the wrapper (will be created during refactoring)
# from tsr.wrappers.openrave.robot import OpenRAVERobotAdapter
# from tsr.wrappers.openrave.tsr import place_object, transport_upright


class TestOpenRAVEWrapper(unittest.TestCase):
    """Test the OpenRAVE wrapper functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.robot = create_test_robot()
        self.obj = create_test_object()
        self.manip_idx = 0
        
        # Set up a basic grasp scenario
        setup_grasp_scenario(self.robot, self.obj, self.manip_idx)
    
    def test_robot_adapter_creation(self):
        """Test that the robot adapter can be created."""
        # This test will be implemented when we create the wrapper
        # adapter = OpenRAVERobotAdapter(self.robot)
        # self.assertIsNotNone(adapter)
        pass
    
    def test_manipulator_transform_access(self):
        """Test that we can access manipulator transforms."""
        manip = self.robot.GetManipulator(self.manip_idx)
        transform = manip.GetEndEffectorTransform()
        
        # Should be a 4x4 matrix
        self.assertEqual(transform.shape, (4, 4))
        
        # Should be a valid transform matrix
        self.assertTrue(np.allclose(transform[3, :], [0, 0, 0, 1]))
    
    def test_object_transform_access(self):
        """Test that we can access object transforms."""
        transform = self.obj.GetTransform()
        
        # Should be a 4x4 matrix
        self.assertEqual(transform.shape, (4, 4))
        
        # Should be a valid transform matrix
        self.assertTrue(np.allclose(transform[3, :], [0, 0, 0, 1]))
    
    def test_grasp_scenario_setup(self):
        """Test that grasp scenario setup works correctly."""
        robot, obj = setup_grasp_scenario(self.robot, self.obj, self.manip_idx)
        
        # Check that object is positioned correctly
        obj_transform = obj.GetTransform()
        self.assertAlmostEqual(obj_transform[0, 3], 0.5)  # x position
        self.assertAlmostEqual(obj_transform[1, 3], 0.0)  # y position
        self.assertAlmostEqual(obj_transform[2, 3], 0.3)  # z position
        
        # Check that end-effector is positioned relative to object
        manip = robot.GetManipulator(self.manip_idx)
        ee_transform = manip.GetEndEffectorTransform()
        
        # End-effector should be above the object (z > object z)
        self.assertGreater(ee_transform[2, 3], obj_transform[2, 3])
    
    def test_manipulator_grabbing_state(self):
        """Test manipulator grabbing state management."""
        manip = self.robot.GetManipulator(self.manip_idx)
        
        # Initially not grabbing
        self.assertFalse(manip.IsGrabbing(self.obj))
        
        # Set to grabbing
        manip.SetGrabbing(self.obj, True)
        self.assertTrue(manip.IsGrabbing(self.obj))
        
        # Set to not grabbing
        manip.SetGrabbing(self.obj, False)
        self.assertFalse(manip.IsGrabbing(self.obj))
    
    def test_robot_manipulator_management(self):
        """Test robot manipulator management."""
        # Check initial state
        self.assertEqual(self.robot.GetActiveManipulatorIndex(), 0)
        
        # Change active manipulator
        self.robot.SetActiveManipulator(1)
        self.assertEqual(self.robot.GetActiveManipulatorIndex(), 1)
        
        # Test invalid manipulator index
        with self.assertRaises(ValueError):
            self.robot.SetActiveManipulator(10)
    
    def test_object_type_detection(self):
        """Test object type detection from XML filename."""
        # Test with valid filename
        obj = MockKinBody("test_object")
        self.assertEqual(obj.GetName(), "test_object")
        self.assertEqual(obj.GetXMLFilename(), "test_object.xml")
        
        # Test with different name
        obj2 = MockKinBody("different_object")
        self.assertEqual(obj2.GetName(), "different_object")
        self.assertEqual(obj2.GetXMLFilename(), "different_object.xml")


class TestOpenRAVETSRFunctions(unittest.TestCase):
    """Test OpenRAVE-specific TSR functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.robot = create_test_robot()
        self.obj = create_test_object()
        self.manip_idx = 0
        setup_grasp_scenario(self.robot, self.obj, self.manip_idx)
    
    def test_place_object_function(self):
        """Test the place_object function."""
        # This test will be implemented when we move the function to the wrapper
        # The function should:
        # 1. Check that manipulator is grabbing the object
        # 2. Calculate ee_in_obj transform
        # 3. Create appropriate TSR chains
        pass
    
    def test_transport_upright_function(self):
        """Test the transport_upright function."""
        # This test will be implemented when we move the function to the wrapper
        # The function should:
        # 1. Validate epsilon parameters
        # 2. Calculate ee_in_obj transform
        # 3. Create transport TSR with appropriate bounds
        pass
    
    def test_cylinder_grasp_function(self):
        """Test the cylinder_grasp function."""
        # This test will be implemented when we move the function to the wrapper
        pass
    
    def test_box_grasp_function(self):
        """Test the box_grasp function."""
        # This test will be implemented when we move the function to the wrapper
        pass


class TestOpenRAVECompatibility(unittest.TestCase):
    """Test compatibility with existing OpenRAVE code patterns."""
    
    def test_legacy_tsr_creation(self):
        """Test that legacy TSR creation still works."""
        # Import the legacy TSR
        from tsr.tsr import TSR
        
        T0_w = np.eye(4)
        Tw_e = np.eye(4)
        Bw = np.zeros((6, 2))
        
        # Should work with manipindex parameter
        tsr = TSR(T0_w=T0_w, Tw_e=Tw_e, Bw=Bw, manipindex=0)
        self.assertEqual(tsr.manipindex, 0)
        
        # Should work with bodyandlink parameter
        tsr2 = TSR(T0_w=T0_w, Tw_e=Tw_e, Bw=Bw, bodyandlink="test")
        self.assertEqual(tsr2.bodyandlink, "test")
    
    def test_legacy_tsr_chain_creation(self):
        """Test that legacy TSRChain creation still works."""
        # Import the legacy TSRChain
        from tsr.tsr import TSRChain, TSR
        
        tsr = TSR()
        chain = TSRChain(sample_start=False, sample_goal=True, constrain=False, TSR=tsr)
        
        self.assertFalse(chain.sample_start)
        self.assertTrue(chain.sample_goal)
        self.assertFalse(chain.constrain)
        self.assertEqual(len(chain.TSRs), 1)


if __name__ == '__main__':
    unittest.main() 