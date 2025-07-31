# SPDX-License-Identifier: BSD-2-Clause
# Authors: Siddhartha Srinivasa and contributors to TSR

"""
OpenRAVE robot adapter for TSR library.

This module provides an adapter that implements the abstract robot interface
for OpenRAVE robots.
"""

import numpy as np
from typing import List, Optional
import os

from ..base import RobotInterface, ObjectInterface


class OpenRAVERobotAdapter(RobotInterface):
    """
    OpenRAVE robot adapter that implements the abstract robot interface.
    """
    
    def __init__(self, robot):
        """
        Initialize the OpenRAVE robot adapter.
        
        Args:
            robot: OpenRAVE robot object
        """
        self._robot = robot
        self._manipulators = robot.GetManipulators()
        self._manipulator_names = [manip.GetName() for manip in self._manipulators]
        self._manipulator_indices = {name: i for i, name in enumerate(self._manipulator_names)}
    
    def get_manipulator_transform(self, manip_idx: int) -> np.ndarray:
        """Get the end-effector transform for a manipulator."""
        if manip_idx < 0 or manip_idx >= len(self._manipulators):
            raise ValueError(f"Invalid manipulator index: {manip_idx}")
        
        manip = self._manipulators[manip_idx]
        return manip.GetEndEffectorTransform()
    
    def get_object_transform(self, obj_name: str) -> np.ndarray:
        """Get the transform of an object."""
        # This method requires access to the environment
        # For now, we'll raise an error - this should be handled by the environment adapter
        raise NotImplementedError("Object transforms should be accessed through the environment adapter")
    
    def get_manipulator_index(self, manip_name: str) -> int:
        """Get manipulator index by name."""
        if manip_name not in self._manipulator_indices:
            raise ValueError(f"Manipulator '{manip_name}' not found. Available: {self._manipulator_names}")
        
        return self._manipulator_indices[manip_name]
    
    def get_manipulator_name(self, manip_idx: int) -> str:
        """Get manipulator name by index."""
        if manip_idx < 0 or manip_idx >= len(self._manipulators):
            raise ValueError(f"Invalid manipulator index: {manip_idx}")
        
        return self._manipulator_names[manip_idx]
    
    def get_active_manipulator_index(self) -> int:
        """Get the currently active manipulator index."""
        return self._robot.GetActiveManipulatorIndex()
    
    def set_active_manipulator(self, manip_idx: int):
        """Set the active manipulator."""
        if manip_idx < 0 or manip_idx >= len(self._manipulators):
            raise ValueError(f"Invalid manipulator index: {manip_idx}")
        
        self._robot.SetActiveManipulator(manip_idx)
    
    def get_manipulator_count(self) -> int:
        """Get the number of manipulators."""
        return len(self._manipulators)
    
    def is_manipulator_grabbing(self, manip_idx: int, obj_name: str) -> bool:
        """Check if a manipulator is grabbing an object."""
        if manip_idx < 0 or manip_idx >= len(self._manipulators):
            raise ValueError(f"Invalid manipulator index: {manip_idx}")
        
        manip = self._manipulators[manip_idx]
        
        # Get all grabbed objects
        grabbed_objects = self._robot.GetGrabbed()
        
        # Check if any grabbed object has the specified name
        for grabbed_obj in grabbed_objects:
            if grabbed_obj.GetName() == obj_name:
                return True
        
        return False
    
    def get_object_name(self, obj) -> str:
        """Get the name of an object."""
        return obj.GetName()
    
    def get_robot_name(self) -> str:
        """Get the name of the robot."""
        return self._robot.GetName()
    
    def get_robot_type(self) -> str:
        """Get the type of the robot (from XML filename)."""
        path = self._robot.GetXMLFilename()
        filename = os.path.basename(path)
        name, _, _ = filename.partition('.')  # remove extension
        return name if name else "unknown"


class OpenRAVEObjectAdapter(ObjectInterface):
    """
    OpenRAVE object adapter that implements the abstract object interface.
    """
    
    def __init__(self, obj):
        """
        Initialize the OpenRAVE object adapter.
        
        Args:
            obj: OpenRAVE KinBody object
        """
        self._obj = obj
    
    def get_transform(self) -> np.ndarray:
        """Get the object's transform."""
        return self._obj.GetTransform()
    
    def get_name(self) -> str:
        """Get the object's name."""
        return self._obj.GetName()
    
    def get_type(self) -> str:
        """Get the object's type (from XML filename)."""
        path = self._obj.GetXMLFilename()
        filename = os.path.basename(path)
        name, _, _ = filename.partition('.')  # remove extension
        return name if name else "unknown"


class OpenRAVEEnvironmentAdapter:
    """
    OpenRAVE environment adapter.
    
    This provides access to robots and objects in an OpenRAVE environment.
    """
    
    def __init__(self, env):
        """
        Initialize the OpenRAVE environment adapter.
        
        Args:
            env: OpenRAVE environment object
        """
        self._env = env
        self._robots = {}
        self._objects = {}
        
        # Cache robots and objects
        self._cache_robots()
        self._cache_objects()
    
    def _cache_robots(self):
        """Cache all robots in the environment."""
        robots = self._env.GetRobots()
        for robot in robots:
            adapter = OpenRAVERobotAdapter(robot)
            self._robots[robot.GetName()] = adapter
    
    def _cache_objects(self):
        """Cache all objects in the environment."""
        bodies = self._env.GetBodies()
        for body in bodies:
            # Skip robots (they're handled separately)
            if body.IsRobot():
                continue
            
            adapter = OpenRAVEObjectAdapter(body)
            self._objects[body.GetName()] = adapter
    
    def get_robot(self, name: str) -> Optional[OpenRAVERobotAdapter]:
        """Get a robot by name."""
        return self._robots.get(name)
    
    def get_object(self, name: str) -> Optional[OpenRAVEObjectAdapter]:
        """Get an object by name."""
        return self._objects.get(name)
    
    def get_all_robots(self) -> List[OpenRAVERobotAdapter]:
        """Get all robots in the environment."""
        return list(self._robots.values())
    
    def get_all_objects(self) -> List[OpenRAVEObjectAdapter]:
        """Get all objects in the environment."""
        return list(self._objects.values())
    
    def refresh(self):
        """Refresh the cached robots and objects."""
        self._robots.clear()
        self._objects.clear()
        self._cache_robots()
        self._cache_objects() 