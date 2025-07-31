# SPDX-License-Identifier: BSD-2-Clause
# Authors: Siddhartha Srinivasa and contributors to TSR

"""
Abstract robot interface for TSR wrappers.

This module defines the abstract base classes that all robot adapters
must implement, regardless of the underlying simulator.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Union
import numpy as np


class RobotInterface(ABC):
    """
    Abstract interface for robot adapters.
    
    This defines the contract that all robot adapters must implement,
    regardless of the underlying simulator (OpenRAVE, MuJoCo, etc.).
    """
    
    @abstractmethod
    def get_manipulator_transform(self, manip_idx: int) -> np.ndarray:
        """
        Get the end-effector transform for a manipulator.
        
        Args:
            manip_idx: Index of the manipulator
            
        Returns:
            4x4 transformation matrix from world to end-effector frame
        """
        pass
    
    @abstractmethod
    def get_object_transform(self, obj_name: str) -> np.ndarray:
        """
        Get the transform of an object.
        
        Args:
            obj_name: Name of the object
            
        Returns:
            4x4 transformation matrix from world to object frame
        """
        pass
    
    @abstractmethod
    def get_manipulator_index(self, manip_name: str) -> int:
        """
        Get manipulator index by name.
        
        Args:
            manip_name: Name of the manipulator
            
        Returns:
            Index of the manipulator
        """
        pass
    
    @abstractmethod
    def get_manipulator_name(self, manip_idx: int) -> str:
        """
        Get manipulator name by index.
        
        Args:
            manip_idx: Index of the manipulator
            
        Returns:
            Name of the manipulator
        """
        pass
    
    @abstractmethod
    def get_active_manipulator_index(self) -> int:
        """
        Get the currently active manipulator index.
        
        Returns:
            Index of the active manipulator
        """
        pass
    
    @abstractmethod
    def set_active_manipulator(self, manip_idx: int):
        """
        Set the active manipulator.
        
        Args:
            manip_idx: Index of the manipulator to activate
        """
        pass
    
    @abstractmethod
    def get_manipulator_count(self) -> int:
        """
        Get the number of manipulators.
        
        Returns:
            Number of manipulators
        """
        pass
    
    @abstractmethod
    def is_manipulator_grabbing(self, manip_idx: int, obj_name: str) -> bool:
        """
        Check if a manipulator is grabbing an object.
        
        Args:
            manip_idx: Index of the manipulator
            obj_name: Name of the object
            
        Returns:
            True if the manipulator is grabbing the object
        """
        pass
    
    @abstractmethod
    def get_object_name(self, obj) -> str:
        """
        Get the name of an object.
        
        Args:
            obj: Object reference (simulator-specific)
            
        Returns:
            Name of the object
        """
        pass
    
    @abstractmethod
    def get_robot_name(self) -> str:
        """
        Get the name of the robot.
        
        Returns:
            Name of the robot
        """
        pass


class ObjectInterface(ABC):
    """
    Abstract interface for object adapters.
    
    This defines the contract that all object adapters must implement.
    """
    
    @abstractmethod
    def get_transform(self) -> np.ndarray:
        """
        Get the object's transform.
        
        Returns:
            4x4 transformation matrix from world to object frame
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """
        Get the object's name.
        
        Returns:
            Name of the object
        """
        pass
    
    @abstractmethod
    def get_type(self) -> str:
        """
        Get the object's type/class.
        
        Returns:
            Type of the object
        """
        pass


class EnvironmentInterface(ABC):
    """
    Abstract interface for environment adapters.
    
    This defines the contract that all environment adapters must implement.
    """
    
    @abstractmethod
    def get_robot(self, name: str) -> Optional[RobotInterface]:
        """
        Get a robot by name.
        
        Args:
            name: Name of the robot
            
        Returns:
            Robot interface or None if not found
        """
        pass
    
    @abstractmethod
    def get_object(self, name: str) -> Optional[ObjectInterface]:
        """
        Get an object by name.
        
        Args:
            name: Name of the object
            
        Returns:
            Object interface or None if not found
        """
        pass
    
    @abstractmethod
    def get_all_robots(self) -> List[RobotInterface]:
        """
        Get all robots in the environment.
        
        Returns:
            List of robot interfaces
        """
        pass
    
    @abstractmethod
    def get_all_objects(self) -> List[ObjectInterface]:
        """
        Get all objects in the environment.
        
        Returns:
            List of object interfaces
        """
        pass


class TSRWrapperFactory:
    """
    Factory for creating TSR wrappers for different simulators.
    """
    
    _wrappers = {}
    
    @classmethod
    def register_wrapper(cls, simulator_type: str, wrapper_class):
        """
        Register a wrapper class for a simulator type.
        
        Args:
            simulator_type: Name of the simulator (e.g., 'openrave', 'mujoco')
            wrapper_class: Class that implements the wrapper interface
        """
        cls._wrappers[simulator_type] = wrapper_class
    
    @classmethod
    def create_wrapper(cls, simulator_type: str, robot, manip_idx: int, **kwargs):
        """
        Create a TSR wrapper for the specified simulator.
        
        Args:
            simulator_type: Name of the simulator
            robot: Robot object (simulator-specific)
            manip_idx: Index of the manipulator
            **kwargs: Additional arguments for the wrapper
            
        Returns:
            TSR wrapper instance
            
        Raises:
            ValueError: If simulator type is not supported
        """
        if simulator_type not in cls._wrappers:
            raise ValueError(f"Unsupported simulator type: {simulator_type}. "
                           f"Available: {list(cls._wrappers.keys())}")
        
        wrapper_class = cls._wrappers[simulator_type]
        return wrapper_class(robot, manip_idx, **kwargs)
    
    @classmethod
    def get_supported_simulators(cls) -> List[str]:
        """
        Get list of supported simulator types.
        
        Returns:
            List of supported simulator names
        """
        return list(cls._wrappers.keys()) 