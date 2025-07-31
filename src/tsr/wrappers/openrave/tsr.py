# SPDX-License-Identifier: BSD-2-Clause
# Authors: Siddhartha Srinivasa and contributors to TSR

"""
OpenRAVE-specific TSR functions.

This module contains TSR functions that are specific to OpenRAVE robots
and objects.
"""

import numpy as np
from typing import List, Optional
from numpy import pi

from tsr.core import TSR, TSRChain
from .robot import OpenRAVERobotAdapter, OpenRAVEObjectAdapter


def place_object(robot_adapter: OpenRAVERobotAdapter, obj_adapter: OpenRAVEObjectAdapter, 
                 pose_tsr_chain: TSRChain, manip_idx: int, **kwargs) -> List[TSRChain]:
    """
    Generates end-effector poses for placing an object.
    This function assumes the object is grasped when called
    
    Args:
        robot_adapter: The robot adapter grasping the object
        obj_adapter: The grasped object adapter
        pose_tsr_chain: The TSR chain for sampling placement poses for the object
        manip_idx: The index of the manipulator to perform the grasp
        **kwargs: Additional arguments
        
    Returns:
        List of TSR chains for placement
        
    Raises:
        Exception: If manipulator is not grabbing the object
    """
    # Check if manipulator is grabbing the object
    if not robot_adapter.is_manipulator_grabbing(manip_idx, obj_adapter.get_name()):
        raise Exception(f'Manipulator {manip_idx} is not grabbing {obj_adapter.get_name()}')

    # Calculate end-effector transform in object frame
    obj_transform = obj_adapter.get_transform()
    ee_transform = robot_adapter.get_manipulator_transform(manip_idx)
    ee_in_obj = np.dot(np.linalg.inv(obj_transform), ee_transform)
    
    # Create bounds for grasp TSR (zero bounds = fixed grasp)
    Bw = np.zeros((6, 2))
    
    # Verify pose_tsr_chain is for the correct manipulator
    for tsr in pose_tsr_chain.TSRs:
        if hasattr(tsr, 'manipindex') and tsr.manipindex != manip_idx:
            raise Exception('pose_tsr_chain defined for a different manipulator.')

    # Create grasp TSR
    grasp_tsr = TSR(Tw_e=ee_in_obj, Bw=Bw)
    
    # Combine pose and grasp TSRs
    all_tsrs = list(pose_tsr_chain.TSRs) + [grasp_tsr]
    place_chain = TSRChain(sample_start=False, sample_goal=True, constrain=False, TSRs=all_tsrs)

    return [place_chain]


def transport_upright(robot_adapter: OpenRAVERobotAdapter, obj_adapter: OpenRAVEObjectAdapter,
                     manip_idx: int, roll_epsilon: float = 0.2, 
                     pitch_epsilon: float = 0.2, yaw_epsilon: float = 0.2,
                     **kwargs) -> List[TSRChain]:
    """
    Generates a trajectory-wide constraint for transporting the object with little roll, pitch or yaw.
    Assumes the object has already been grasped and is in the proper configuration for transport.

    Args:
        robot_adapter: The robot adapter grasping the object
        obj_adapter: The grasped object adapter
        manip_idx: The index of the manipulator to perform the grasp
        roll_epsilon: The amount to let the object roll during transport (object frame)
        pitch_epsilon: The amount to let the object pitch during transport (object frame)
        yaw_epsilon: The amount to let the object yaw during transport (object frame)
        **kwargs: Additional arguments
        
    Returns:
        List of TSR chains for transport
        
    Raises:
        Exception: If epsilon parameters are negative
    """
    # Validate epsilon parameters
    if roll_epsilon < 0.0:
        raise Exception('roll_epsilon must be >= 0')
    if pitch_epsilon < 0.0:
        raise Exception('pitch_epsilon must be >= 0')
    if yaw_epsilon < 0.0:
        raise Exception('yaw_epsilon must be >= 0')

    # Calculate end-effector transform in object frame
    obj_transform = obj_adapter.get_transform()
    ee_transform = robot_adapter.get_manipulator_transform(manip_idx)
    ee_in_obj = np.dot(np.linalg.inv(obj_transform), ee_transform)
    
    # Create bounds that cover full reachability of manipulator
    Bw = np.array([
        [-100., 100.],  # x bounds
        [-100., 100.],  # y bounds
        [-100., 100.],  # z bounds
        [-roll_epsilon, roll_epsilon],    # roll bounds
        [-pitch_epsilon, pitch_epsilon],  # pitch bounds
        [-yaw_epsilon, yaw_epsilon]       # yaw bounds
    ])
    
    # Create transport TSR
    transport_tsr = TSR(
        T0_w=obj_transform,
        Tw_e=ee_in_obj,
        Bw=Bw
    )

    # Create transport chain
    transport_chain = TSRChain(
        sample_start=False, 
        sample_goal=False, 
        constrain=True, 
        TSR=transport_tsr
    )
    
    return [transport_chain]


def cylinder_grasp(robot_adapter: OpenRAVERobotAdapter, obj_adapter: OpenRAVEObjectAdapter,
                  obj_radius: float, obj_height: float, lateral_offset: float = 0.0, 
                  vertical_tolerance: float = 0.02, yaw_range: Optional[List[float]] = None,
                  manip_idx: Optional[int] = None, **kwargs) -> List[TSRChain]:
    """
    Generate TSRs for grasping a cylindrical object.
    
    Args:
        robot_adapter: The robot adapter
        obj_adapter: The cylindrical object adapter
        obj_radius: Radius of the cylinder
        obj_height: Height of the cylinder
        lateral_offset: Lateral offset from cylinder center
        vertical_tolerance: Vertical tolerance for grasp
        yaw_range: Range of yaw angles [min, max] (if None, allow full rotation)
        manip_idx: Index of manipulator to use (if None, use active manipulator)
        **kwargs: Additional arguments
        
    Returns:
        List of TSR chains for cylinder grasping
    """
    if manip_idx is None:
        manip_idx = robot_adapter.get_active_manipulator_index()
    
    if yaw_range is None:
        yaw_range = [-pi, pi]
    
    # Get object transform
    obj_transform = obj_adapter.get_transform()
    
    # Create grasp TSR
    # Approach from above with lateral offset
    Tw_e = np.array([
        [1., 0., 0., lateral_offset],
        [0., 1., 0., 0.],
        [0., 0., 1., obj_height/2.0],  # Grasp at middle of cylinder
        [0., 0., 0., 1.]
    ])
    
    # Create bounds
    Bw = np.array([
        [-0.01, 0.01],  # x bounds (tight)
        [-0.01, 0.01],  # y bounds (tight)
        [-vertical_tolerance, vertical_tolerance],  # z bounds
        [-0.1, 0.1],    # roll bounds
        [-0.1, 0.1],    # pitch bounds
        [yaw_range[0], yaw_range[1]]  # yaw bounds
    ])
    
    grasp_tsr = TSR(T0_w=obj_transform, Tw_e=Tw_e, Bw=Bw)
    grasp_chain = TSRChain(sample_start=False, sample_goal=True, constrain=False, TSR=grasp_tsr)
    
    return [grasp_chain]


def box_grasp(robot_adapter: OpenRAVERobotAdapter, obj_adapter: OpenRAVEObjectAdapter,
              length: float, width: float, height: float, manip_idx: int,
              lateral_offset: float = 0.0, lateral_tolerance: float = 0.02,
              **kwargs) -> List[TSRChain]:
    """
    Generate TSRs for grasping a box-shaped object.
    
    Args:
        robot_adapter: The robot adapter
        obj_adapter: The box object adapter
        length: Length of the box
        width: Width of the box
        height: Height of the box
        manip_idx: Index of manipulator to use
        lateral_offset: Lateral offset from box center
        lateral_tolerance: Lateral tolerance for grasp
        **kwargs: Additional arguments
        
    Returns:
        List of TSR chains for box grasping
    """
    # Get object transform
    obj_transform = obj_adapter.get_transform()
    
    # Create grasp TSR
    # Approach from above with lateral offset
    Tw_e = np.array([
        [1., 0., 0., lateral_offset],
        [0., 1., 0., 0.],
        [0., 0., 1., height/2.0],  # Grasp at middle of box
        [0., 0., 0., 1.]
    ])
    
    # Create bounds
    Bw = np.array([
        [-lateral_tolerance, lateral_tolerance],  # x bounds
        [-lateral_tolerance, lateral_tolerance],  # y bounds
        [-0.01, 0.01],  # z bounds (tight)
        [-0.1, 0.1],    # roll bounds
        [-0.1, 0.1],    # pitch bounds
        [-pi, pi]       # yaw bounds (allow full rotation)
    ])
    
    grasp_tsr = TSR(T0_w=obj_transform, Tw_e=Tw_e, Bw=Bw)
    grasp_chain = TSRChain(sample_start=False, sample_goal=True, constrain=False, TSR=grasp_tsr)
    
    return [grasp_chain] 