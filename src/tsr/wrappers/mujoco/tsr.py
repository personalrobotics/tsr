# SPDX-License-Identifier: BSD-2-Clause
# Authors: Siddhartha Srinivasa and contributors to TSR

"""
MuJoCo-specific TSR functions.

This module provides MuJoCo-adapted versions of the generic TSR functions,
handling MuJoCo-specific data structures and multi-arm scenarios.
"""

import numpy as np
from typing import List, Optional
from ...core.tsr import TSR
from ...core.tsr_chain import TSRChain
from .robot import MuJoCoRobotAdapter


def cylinder_grasp(robot: MuJoCoRobotAdapter, obj, obj_radius: float, obj_height: float,
                   lateral_offset: float = 0.0, 
                   vertical_tolerance: float = 0.02,
                   yaw_range: Optional[List[float]] = None,
                   manip_idx: Optional[int] = None, **kwargs) -> List[TSRChain]:
    """
    Generate TSR chains for grasping a cylinder with MuJoCo robot.
    
    This is a MuJoCo-adapted version of the generic cylinder_grasp function.
    
    Args:
        robot: MuJoCo robot adapter
        obj: MuJoCo object to grasp
        obj_radius: Radius of the cylinder
        obj_height: Height of the cylinder
        lateral_offset: Lateral offset from edge of object to end-effector
        vertical_tolerance: Maximum vertical distance from center for grasp
        yaw_range: Allowable range of yaw around object (default: [-pi, pi])
        manip_idx: Index of manipulator to use (defaults to primary manipulator)
        **kwargs: Additional arguments
        
    Returns:
        List of TSR chains for grasping
    """
    if obj_radius <= 0.0:
        raise ValueError('obj_radius must be > 0')

    if obj_height <= 0.0:
        raise ValueError('obj_height must be > 0')
        
    if vertical_tolerance < 0.0:
        raise ValueError('vertical_tolerance must be >= 0')

    if yaw_range is not None and len(yaw_range) != 2:
        raise ValueError('yaw_range parameter must be 2 element list specifying min and max values')

    if yaw_range is not None and yaw_range[0] > yaw_range[1]:
        raise ValueError('The first element of the yaw_range parameter must be greater '
                        'than or equal to the second (current values [%f, %f])' 
                        % (yaw_range[0], yaw_range[1]))

    # Use specified manipulator or primary manipulator
    if manip_idx is None:
        manip_idx = robot.get_primary_manipulator_index()
    
    # Get object transform from MuJoCo
    obj_name = robot.get_object_name(obj)
    T0_w = robot.get_object_transform(obj_name)
    total_offset = lateral_offset + obj_radius

    # First hand orientation
    Tw_e_1 = np.array([[ 0., 0., 1., -total_offset], 
                        [1., 0., 0., 0.], 
                        [0., 1., 0., obj_height*0.5], 
                        [0., 0., 0., 1.]])

    Bw = np.zeros((6,2))
    Bw[2,:] = [-vertical_tolerance, vertical_tolerance]  # Allow a little vertical movement
    if yaw_range is None:
        Bw[5,:] = [-np.pi, np.pi]  # Allow any orientation
    else:
        Bw[5,:] = yaw_range
    
    # Create TSR with manipindex for multi-arm disambiguation
    grasp_tsr1 = TSR(T0_w=T0_w, Tw_e=Tw_e_1, Bw=Bw)
    grasp_chain1 = TSRChain(sample_start=False, sample_goal=True, 
                            constrain=False, TSR=grasp_tsr1)

    # Flipped hand orientation
    Tw_e_2 = np.array([[ 0., 0., 1., -total_offset], 
                          [-1., 0., 0., 0.], 
                          [0.,-1., 0., obj_height*0.5], 
                          [0., 0., 0., 1.]])

    grasp_tsr2 = TSR(T0_w=T0_w, Tw_e=Tw_e_2, Bw=Bw)
    grasp_chain2 = TSRChain(sample_start=False, sample_goal=True, 
                            constrain=False, TSR=grasp_tsr2)

    return [grasp_chain1, grasp_chain2]


def box_grasp(robot: MuJoCoRobotAdapter, box, length: float, width: float, height: float,
              lateral_offset: float = 0.0,
              lateral_tolerance: float = 0.02,
              manip_idx: Optional[int] = None, **kwargs) -> List[TSRChain]:
    """
    Generate TSR chains for grasping a box with MuJoCo robot.
    
    This is a MuJoCo-adapted version of the generic box_grasp function.
    
    Args:
        robot: MuJoCo robot adapter
        box: MuJoCo box object to grasp
        length: Length of the box - along its x-axis
        width: Width of the box - along its y-axis
        height: Height of the box - along its z-axis
        lateral_offset: Offset from edge of box to end-effector
        lateral_tolerance: Maximum distance along edge from center for good grasp
        manip_idx: Index of manipulator to use (defaults to primary manipulator)
        **kwargs: Additional arguments
        
    Returns:
        List of TSR chains for grasping
    """
    if length <= 0.0:
        raise ValueError('length must be > 0')

    if width <= 0.0:
        raise ValueError('width must be > 0')
        
    if height <= 0.0:
        raise ValueError('height must be > 0')

    if lateral_tolerance < 0.0:
        raise ValueError('lateral_tolerance must be >= 0.0')

    # Use specified manipulator or primary manipulator
    if manip_idx is None:
        manip_idx = robot.get_primary_manipulator_index()

    # Get object transform from MuJoCo
    box_name = robot.get_object_name(box)
    T0_w = robot.get_object_transform(box_name)

    chain_list = []
    
    # Top face
    Tw_e_top1 = np.array([[0., 1.,  0., 0.],
                             [1., 0.,  0., 0.],
                             [0., 0., -1., lateral_offset + height],
                             [0., 0.,  0., 1.]])
    Bw_top1 = np.zeros((6,2))
    Bw_top1[1,:] = [-lateral_tolerance, lateral_tolerance]
    top_tsr1 = TSR(T0_w=T0_w, Tw_e=Tw_e_top1, Bw=Bw_top1)
    grasp_chain_top = TSRChain(sample_start=False, sample_goal=True,
                            constrain=False, TSR=top_tsr1)
    chain_list += [grasp_chain_top]

    # Bottom face
    Tw_e_bottom1 = np.array([[ 0., 1.,  0., 0.],
                                [-1., 0.,  0., 0.],
                                [ 0., 0.,  1., -lateral_offset],
                                [ 0., 0.,  0., 1.]])
    Bw_bottom1 = np.zeros((6,2))
    Bw_bottom1[1,:] = [-lateral_tolerance, lateral_tolerance]
    bottom_tsr1 = TSR(T0_w=T0_w, Tw_e=Tw_e_bottom1, Bw=Bw_bottom1)
    grasp_chain_bottom = TSRChain(sample_start=False, sample_goal=True,
                                  constrain=False, TSR=bottom_tsr1)
    chain_list += [grasp_chain_bottom]

    # Front - yz face
    Tw_e_front1 = np.array([[ 0., 0., -1., 0.5*length + lateral_offset],
                               [ 1., 0.,  0., 0.],
                               [ 0.,-1.,  0., 0.5*height],
                               [ 0., 0.,  0., 1.]])
    Bw_front1 = np.zeros((6,2))
    Bw_front1[1,:] = [-lateral_tolerance, lateral_tolerance]
    front_tsr1 = TSR(T0_w=T0_w, Tw_e=Tw_e_front1, Bw=Bw_front1)
    grasp_chain_front = TSRChain(sample_start=False, sample_goal=True,
                                 constrain=False, TSR=front_tsr1)
    chain_list += [grasp_chain_front]

    # Back - yz face
    Tw_e_back1 = np.array([[ 0., 0.,  1., -0.5*length - lateral_offset],
                              [-1., 0.,  0., 0.],
                              [ 0.,-1.,  0., 0.5*height],
                              [ 0., 0.,  0., 1.]])
    Bw_back1 = np.zeros((6,2))
    Bw_back1[1,:] = [-lateral_tolerance, lateral_tolerance]
    back_tsr1 = TSR(T0_w=T0_w, Tw_e=Tw_e_back1, Bw=Bw_back1)
    grasp_chain_back = TSRChain(sample_start=False, sample_goal=True,
                                constrain=False, TSR=back_tsr1)
    chain_list += [grasp_chain_back]

    # Side - xz face
    Tw_e_side1 = np.array([[-1., 0.,  0., 0.],
                              [ 0., 0., -1., 0.5*width + lateral_offset],
                              [ 0.,-1.,  0., 0.5*height],
                              [ 0., 0.,  0., 1.]])
    Bw_side1 = np.zeros((6,2))
    Bw_side1[0,:] = [-lateral_tolerance, lateral_tolerance]
    side_tsr1 = TSR(T0_w=T0_w, Tw_e=Tw_e_side1, Bw=Bw_side1)
    grasp_chain_side1 = TSRChain(sample_start=False, sample_goal=True,
                                constrain=False, TSR=side_tsr1)
    chain_list += [grasp_chain_side1]

    # Other Side - xz face
    Tw_e_side2 = np.array([[ 1., 0.,  0., 0.],
                              [ 0., 0.,  1.,-0.5*width - lateral_offset],
                              [ 0.,-1.,  0., 0.5*height],
                              [ 0., 0.,  0., 1.]])
    Bw_side2 = np.zeros((6,2))
    Bw_side2[0,:] = [-lateral_tolerance, lateral_tolerance]
    side_tsr2 = TSR(T0_w=T0_w, Tw_e=Tw_e_side2, Bw=Bw_side2)
    grasp_chain_side2 = TSRChain(sample_start=False, sample_goal=True,
                                 constrain=False, TSR=side_tsr2)
    chain_list += [grasp_chain_side2]

    # Each chain in the list can also be rotated by 180 degrees around z
    rotated_chain_list = []
    for c in chain_list:
        rval = np.pi
        R = np.array([[np.cos(rval), -np.sin(rval), 0., 0.],
                         [np.sin(rval),  np.cos(rval), 0., 0.],
                         [             0.,               0., 1., 0.],
                         [             0.,               0., 0., 1.]])
        tsr = c.TSRs[0]
        Tw_e = tsr.Tw_e
        Tw_e_new = np.dot(Tw_e, R)
        tsr_new = TSR(T0_w=tsr.T0_w, Tw_e=Tw_e_new, Bw=tsr.Bw)
        tsr_chain_new = TSRChain(sample_start=False, sample_goal=True, constrain=False,
                                     TSR=tsr_new)
        rotated_chain_list += [tsr_chain_new]

    return chain_list + rotated_chain_list


def place_object(robot: MuJoCoRobotAdapter, obj, pose_tsr_chain: TSRChain,
                 manip_idx: Optional[int] = None, **kwargs) -> List[TSRChain]:
    """
    Generate TSR chains for placing an object with MuJoCo robot.
    
    This is a MuJoCo-adapted version of the generic place_object function.
    
    Args:
        robot: MuJoCo robot adapter
        obj: MuJoCo object to place
        pose_tsr_chain: TSR chain for sampling placement poses
        manip_idx: Index of manipulator to use (defaults to primary manipulator)
        **kwargs: Additional arguments
        
    Returns:
        List of TSR chains for placing
    """
    # Use specified manipulator or primary manipulator
    if manip_idx is None:
        manip_idx = robot.get_primary_manipulator_index()

    # Check if manipulator is grabbing the object
    obj_name = robot.get_object_name(obj)
    if not robot.is_manipulator_grabbing(manip_idx, obj_name):
        raise ValueError(f'manipulator {manip_idx} is not grabbing {obj_name}')

    # Calculate end-effector in object transform
    obj_transform = robot.get_object_transform(obj_name)
    ee_transform = robot.get_manipulator_transform(manip_idx)
    ee_in_obj = np.dot(np.linalg.inv(obj_transform), ee_transform)
    
    Bw = np.zeros((6,2)) 
   
    # Create grasp TSR
    grasp_tsr = TSR(Tw_e=ee_in_obj, Bw=Bw)
    all_tsrs = list(pose_tsr_chain.TSRs) + [grasp_tsr]
    place_chain = TSRChain(sample_start=False, sample_goal=True, constrain=False,
                           TSRs=all_tsrs)

    return [place_chain]


def transport_upright(robot: MuJoCoRobotAdapter, obj,
                      roll_epsilon: float = 0.2, 
                      pitch_epsilon: float = 0.2, 
                      yaw_epsilon: float = 0.2,
                      manip_idx: Optional[int] = None, **kwargs) -> List[TSRChain]:
    """
    Generate trajectory-wide constraint for upright transport with MuJoCo robot.
    
    This is a MuJoCo-adapted version of the generic transport_upright function.
    
    Args:
        robot: MuJoCo robot adapter
        obj: MuJoCo object to transport
        roll_epsilon: Amount to let object roll during transport
        pitch_epsilon: Amount to let object pitch during transport
        yaw_epsilon: Amount to let object yaw during transport
        manip_idx: Index of manipulator to use (defaults to primary manipulator)
        **kwargs: Additional arguments
        
    Returns:
        List of TSR chains for transport
    """
    if roll_epsilon < 0.0:
        raise ValueError('roll_epsilon must be >= 0')
        
    if pitch_epsilon < 0.0:
        raise ValueError('pitch_epsilon must be >= 0')

    if yaw_epsilon < 0.0:
        raise ValueError('yaw_epsilon must be >= 0')

    # Use specified manipulator or primary manipulator
    if manip_idx is None:
        manip_idx = robot.get_primary_manipulator_index()

    # Calculate end-effector in object transform
    obj_transform = robot.get_object_transform(robot.get_object_name(obj))
    ee_transform = robot.get_manipulator_transform(manip_idx)
    ee_in_obj = np.dot(np.linalg.inv(obj_transform), ee_transform)
    
    Bw = np.array([[-100., 100.], # bounds that cover full reachability of manip
                      [-100., 100.],
                      [-100., 100.],
                      [-roll_epsilon, roll_epsilon],
                      [-pitch_epsilon, pitch_epsilon],
                      [-yaw_epsilon, yaw_epsilon]])
    
    transport_tsr = TSR(T0_w=obj_transform,
                        Tw_e=ee_in_obj,
                        Bw=Bw)

    transport_chain = TSRChain(sample_start=False, sample_goal=False, 
                               constrain=True, TSR=transport_tsr)
    
    return [transport_chain] 