#!/usr/bin/env python
"""
Mock robot interface for testing without simulator dependencies.

This provides a mock implementation that mimics OpenRAVE robot behavior
for testing purposes.
"""

import numpy as np
from typing import List, Optional


class MockManipulator:
    """Mock manipulator for testing."""
    
    def __init__(self, name: str = "mock_manipulator"):
        self.name = name
        self._transform = np.eye(4)
        self._is_grabbing = False
        self._grabbed_object = None
    
    def GetEndEffectorTransform(self) -> np.ndarray:
        """Get the end-effector transform."""
        return self._transform.copy()
    
    def SetEndEffectorTransform(self, transform: np.ndarray):
        """Set the end-effector transform."""
        self._transform = transform.copy()
    
    def GetName(self) -> str:
        """Get the manipulator name."""
        return self.name
    
    def IsGrabbing(self, obj) -> bool:
        """Check if this manipulator is grabbing the given object."""
        return self._is_grabbing and self._grabbed_object == obj
    
    def SetGrabbing(self, obj, is_grabbing: bool):
        """Set whether this manipulator is grabbing an object."""
        self._is_grabbing = is_grabbing
        self._grabbed_object = obj if is_grabbing else None


class MockKinBody:
    """Mock KinBody for testing."""
    
    def __init__(self, name: str = "mock_object"):
        self.name = name
        self._transform = np.eye(4)
        self._xml_filename = f"{name}.xml"
    
    def GetTransform(self) -> np.ndarray:
        """Get the object transform."""
        return self._transform.copy()
    
    def SetTransform(self, transform: np.ndarray):
        """Set the object transform."""
        self._transform = transform.copy()
    
    def GetName(self) -> str:
        """Get the object name."""
        return self.name
    
    def GetXMLFilename(self) -> str:
        """Get the XML filename."""
        return self._xml_filename


class MockRobot:
    """Mock robot for testing without simulator dependencies."""
    
    def __init__(self, name: str = "mock_robot"):
        self.name = name
        self.manipulators = [MockManipulator("right_arm"), MockManipulator("left_arm")]
        self._active_manip_idx = 0
        self._xml_filename = f"{name}.xml"
    
    def GetManipulators(self) -> List[MockManipulator]:
        """Get all manipulators."""
        return self.manipulators
    
    def GetActiveManipulatorIndex(self) -> int:
        """Get the active manipulator index."""
        return self._active_manip_idx
    
    def SetActiveManipulator(self, manip_idx: int):
        """Set the active manipulator."""
        if 0 <= manip_idx < len(self.manipulators):
            self._active_manip_idx = manip_idx
        else:
            raise ValueError(f"Invalid manipulator index: {manip_idx}")
    
    def GetName(self) -> str:
        """Get the robot name."""
        return self.name
    
    def GetXMLFilename(self) -> str:
        """Get the XML filename."""
        return self._xml_filename
    
    def GetManipulator(self, manip_idx: int) -> MockManipulator:
        """Get a specific manipulator by index."""
        if 0 <= manip_idx < len(self.manipulators):
            return self.manipulators[manip_idx]
        else:
            raise ValueError(f"Invalid manipulator index: {manip_idx}")


class MockEnvironment:
    """Mock environment for testing."""
    
    def __init__(self):
        self.robots = {}
        self.objects = {}
    
    def AddRobot(self, robot: MockRobot):
        """Add a robot to the environment."""
        self.robots[robot.GetName()] = robot
    
    def AddKinBody(self, obj: MockKinBody):
        """Add a KinBody to the environment."""
        self.objects[obj.GetName()] = obj
    
    def GetRobot(self, name: str) -> Optional[MockRobot]:
        """Get a robot by name."""
        return self.robots.get(name)
    
    def GetKinBody(self, name: str) -> Optional[MockKinBody]:
        """Get a KinBody by name."""
        return self.objects.get(name)
    
    def GetRobots(self) -> List[MockRobot]:
        """Get all robots."""
        return list(self.robots.values())
    
    def GetKinBodies(self) -> List[MockKinBody]:
        """Get all KinBodies."""
        return list(self.objects.values())


# Factory functions for easy test setup
def create_test_robot(name: str = "test_robot") -> MockRobot:
    """Create a test robot with default manipulators."""
    return MockRobot(name)


def create_test_object(name: str = "test_object") -> MockKinBody:
    """Create a test object."""
    return MockKinBody(name)


def create_test_environment() -> MockEnvironment:
    """Create a test environment with a robot and object."""
    env = MockEnvironment()
    
    robot = create_test_robot()
    obj = create_test_object()
    
    env.AddRobot(robot)
    env.AddKinBody(obj)
    
    return env


def setup_grasp_scenario(robot: MockRobot, obj: MockKinBody, manip_idx: int = 0):
    """Set up a grasp scenario for testing."""
    # Set object position
    obj_transform = np.array([
        [1, 0, 0, 0.5],
        [0, 1, 0, 0.0],
        [0, 0, 1, 0.3],
        [0, 0, 0, 1]
    ])
    obj.SetTransform(obj_transform)
    
    # Set end-effector position relative to object
    ee_transform = np.array([
        [0, 0, 1, 0.4],  # Approach from above
        [1, 0, 0, 0.0],
        [0, 1, 0, 0.0],
        [0, 0, 0, 1]
    ])
    
    manip = robot.GetManipulator(manip_idx)
    manip.SetEndEffectorTransform(ee_transform)
    
    # Set robot as active manipulator
    robot.SetActiveManipulator(manip_idx)
    
    return robot, obj 