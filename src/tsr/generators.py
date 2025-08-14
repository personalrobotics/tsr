"""Generic TSR template generators for primitive objects.

This module provides functions to generate TSR templates for common primitive
objects (cylinders, boxes, spheres) and common tasks (grasping, placing, transport).
All functions are simulator-agnostic and return TSRTemplate objects with semantic context.
"""

import numpy as np
from typing import Optional, Tuple, List
from .core.tsr_template import TSRTemplate
from .schema import EntityClass, TaskCategory, TaskType


def generate_cylinder_grasp_template(
    subject_entity: EntityClass,
    reference_entity: EntityClass,
    variant: str,
    cylinder_radius: float,
    cylinder_height: float,
    approach_distance: float = 0.05,
    vertical_tolerance: float = 0.02,
    yaw_range: Optional[Tuple[float, float]] = None,
    name: str = "",
    description: str = ""
) -> TSRTemplate:
    """Generate a TSR template for grasping a cylindrical object.
    
    This function generates TSR templates for grasping cylindrical objects
    like mugs, bottles, or cans. It supports different grasp variants:
    - "side": Side grasp with approach from the side
    - "top": Top grasp with approach from above
    - "bottom": Bottom grasp with approach from below
    
    Args:
        subject_entity: The entity performing the grasp (e.g., gripper)
        reference_entity: The entity being grasped (e.g., mug, bottle)
        variant: Grasp variant ("side", "top", "bottom")
        cylinder_radius: Radius of the cylinder in meters
        cylinder_height: Height of the cylinder in meters
        approach_distance: Distance from cylinder surface to end-effector
        vertical_tolerance: Allowable vertical movement during grasp
        yaw_range: Allowable yaw rotation range (min, max) in radians
        name: Optional name for the template
        description: Optional description of the template
        
    Returns:
        TSRTemplate for the specified grasp variant
        
    Raises:
        ValueError: If parameters are invalid
    """
    if cylinder_radius <= 0.0:
        raise ValueError('cylinder_radius must be > 0')
    if cylinder_height <= 0.0:
        raise ValueError('cylinder_height must be > 0')
    if approach_distance < 0.0:
        raise ValueError('approach_distance must be >= 0')
    if vertical_tolerance < 0.0:
        raise ValueError('vertical_tolerance must be >= 0')
    
    # Default yaw range if not specified
    if yaw_range is None:
        yaw_range = (-np.pi, np.pi)
    
    # Generate name if not provided
    if not name:
        name = f"{reference_entity.value.title()} {variant.title()} Grasp"
    
    # Generate description if not provided
    if not description:
        description = f"{variant.title()} grasp for {reference_entity.value} with {approach_distance*1000:.0f}mm approach distance"
    
    # Set up transform matrices based on variant
    if variant == "side":
        # Side grasp: approach from -z, x perpendicular to cylinder
        T_ref_tsr = np.eye(4)
        Tw_e = np.array([
            [0, 0, 1, -(cylinder_radius + approach_distance)],  # Approach from -z
            [1, 0, 0, 0],                                        # x perpendicular to cylinder
            [0, 1, 0, cylinder_height * 0.5],                   # y along cylinder axis
            [0, 0, 0, 1]
        ])
        
        # Bounds: fixed x,y position, small z tolerance, full yaw rotation
        Bw = np.array([
            [0, 0],                    # x: fixed position
            [0, 0],                    # y: fixed position
            [-vertical_tolerance, vertical_tolerance],  # z: small tolerance
            [0, 0],                    # roll: fixed
            [0, 0],                    # pitch: fixed
            [yaw_range[0], yaw_range[1]]  # yaw: configurable range
        ])
        
    elif variant == "top":
        # Top grasp: approach from -z, centered on top
        T_ref_tsr = np.eye(4)
        Tw_e = np.array([
            [0, 0, 1, -approach_distance],  # Approach from -z
            [1, 0, 0, 0],                   # x perpendicular
            [0, 1, 0, cylinder_height],     # y at top of cylinder
            [0, 0, 0, 1]
        ])
        
        # Bounds: small x,y tolerance, fixed z, full yaw rotation
        Bw = np.array([
            [-vertical_tolerance, vertical_tolerance],  # x: small tolerance
            [-vertical_tolerance, vertical_tolerance],  # y: small tolerance
            [0, 0],                    # z: fixed position
            [0, 0],                    # roll: fixed
            [0, 0],                    # pitch: fixed
            [yaw_range[0], yaw_range[1]]  # yaw: configurable range
        ])
        
    elif variant == "bottom":
        # Bottom grasp: approach from +z, centered on bottom
        T_ref_tsr = np.eye(4)
        Tw_e = np.array([
            [0, 0, -1, approach_distance],  # Approach from +z
            [1, 0, 0, 0],                   # x perpendicular
            [0, 1, 0, 0],                   # y at bottom of cylinder
            [0, 0, 0, 1]
        ])
        
        # Bounds: small x,y tolerance, fixed z, full yaw rotation
        Bw = np.array([
            [-vertical_tolerance, vertical_tolerance],  # x: small tolerance
            [-vertical_tolerance, vertical_tolerance],  # y: small tolerance
            [0, 0],                    # z: fixed position
            [0, 0],                    # roll: fixed
            [0, 0],                    # pitch: fixed
            [yaw_range[0], yaw_range[1]]  # yaw: configurable range
        ])
        
    else:
        raise ValueError(f'Unknown variant "{variant}". Must be "side", "top", or "bottom"')
    
    return TSRTemplate(
        T_ref_tsr=T_ref_tsr,
        Tw_e=Tw_e,
        Bw=Bw,
        subject_entity=subject_entity,
        reference_entity=reference_entity,
        task_category=TaskCategory.GRASP,
        variant=variant,
        name=name,
        description=description
    )


def generate_box_grasp_template(
    subject_entity: EntityClass,
    reference_entity: EntityClass,
    variant: str,
    box_length: float,
    box_width: float,
    box_height: float,
    approach_distance: float = 0.05,
    lateral_tolerance: float = 0.02,
    yaw_range: Optional[Tuple[float, float]] = None,
    name: str = "",
    description: str = ""
) -> TSRTemplate:
    """Generate a TSR template for grasping a box-shaped object.
    
    This function generates TSR templates for grasping box-shaped objects
    like books, packages, or rectangular containers. It supports different
    grasp variants based on which face to grasp:
    - "side_x": Grasp from side along x-axis
    - "side_y": Grasp from side along y-axis
    - "top": Grasp from top face
    - "bottom": Grasp from bottom face
    
    Args:
        subject_entity: The entity performing the grasp (e.g., gripper)
        reference_entity: The entity being grasped (e.g., box, book)
        variant: Grasp variant ("side_x", "side_y", "top", "bottom")
        box_length: Length of the box in meters (x dimension)
        box_width: Width of the box in meters (y dimension)
        box_height: Height of the box in meters (z dimension)
        approach_distance: Distance from box surface to end-effector
        lateral_tolerance: Allowable lateral movement during grasp
        yaw_range: Allowable yaw rotation range (min, max) in radians
        name: Optional name for the template
        description: Optional description of the template
        
    Returns:
        TSRTemplate for the specified grasp variant
        
    Raises:
        ValueError: If parameters are invalid
    """
    if box_length <= 0.0 or box_width <= 0.0 or box_height <= 0.0:
        raise ValueError('box dimensions must be > 0')
    if approach_distance < 0.0:
        raise ValueError('approach_distance must be >= 0')
    if lateral_tolerance < 0.0:
        raise ValueError('lateral_tolerance must be >= 0')
    
    # Default yaw range if not specified
    if yaw_range is None:
        yaw_range = (-np.pi, np.pi)
    
    # Generate name if not provided
    if not name:
        name = f"{reference_entity.value.title()} {variant.title()} Grasp"
    
    # Generate description if not provided
    if not description:
        description = f"{variant.title()} grasp for {reference_entity.value} with {approach_distance*1000:.0f}mm approach distance"
    
    # Set up transform matrices based on variant
    if variant == "side_x":
        # Side grasp along x-axis: approach from -x
        T_ref_tsr = np.eye(4)
        Tw_e = np.array([
            [-1, 0, 0, -(box_length/2 + approach_distance)],  # Approach from -x
            [0, 1, 0, 0],                                      # y along box width
            [0, 0, 1, box_height/2],                           # z at center height
            [0, 0, 0, 1]
        ])
        
        # Bounds: fixed x position, small y,z tolerance, full yaw rotation
        Bw = np.array([
            [0, 0],                    # x: fixed position
            [-lateral_tolerance, lateral_tolerance],  # y: small tolerance
            [-lateral_tolerance, lateral_tolerance],  # z: small tolerance
            [0, 0],                    # roll: fixed
            [0, 0],                    # pitch: fixed
            [yaw_range[0], yaw_range[1]]  # yaw: configurable range
        ])
        
    elif variant == "side_y":
        # Side grasp along y-axis: approach from -y
        T_ref_tsr = np.eye(4)
        Tw_e = np.array([
            [1, 0, 0, 0],                                      # x along box length
            [0, -1, 0, -(box_width/2 + approach_distance)],    # Approach from -y
            [0, 0, 1, box_height/2],                           # z at center height
            [0, 0, 0, 1]
        ])
        
        # Bounds: small x,z tolerance, fixed y position, full yaw rotation
        Bw = np.array([
            [-lateral_tolerance, lateral_tolerance],  # x: small tolerance
            [0, 0],                    # y: fixed position
            [-lateral_tolerance, lateral_tolerance],  # z: small tolerance
            [0, 0],                    # roll: fixed
            [0, 0],                    # pitch: fixed
            [yaw_range[0], yaw_range[1]]  # yaw: configurable range
        ])
        
    elif variant == "top":
        # Top grasp: approach from -z
        T_ref_tsr = np.eye(4)
        Tw_e = np.array([
            [1, 0, 0, 0],                                      # x along box length
            [0, 1, 0, 0],                                      # y along box width
            [0, 0, 1, box_height + approach_distance],         # Approach from -z
            [0, 0, 0, 1]
        ])
        
        # Bounds: small x,y tolerance, fixed z position, full yaw rotation
        Bw = np.array([
            [-lateral_tolerance, lateral_tolerance],  # x: small tolerance
            [-lateral_tolerance, lateral_tolerance],  # y: small tolerance
            [0, 0],                    # z: fixed position
            [0, 0],                    # roll: fixed
            [0, 0],                    # pitch: fixed
            [yaw_range[0], yaw_range[1]]  # yaw: configurable range
        ])
        
    elif variant == "bottom":
        # Bottom grasp: approach from +z
        T_ref_tsr = np.eye(4)
        Tw_e = np.array([
            [1, 0, 0, 0],                                      # x along box length
            [0, 1, 0, 0],                                      # y along box width
            [0, 0, -1, -approach_distance],                    # Approach from +z
            [0, 0, 0, 1]
        ])
        
        # Bounds: small x,y tolerance, fixed z position, full yaw rotation
        Bw = np.array([
            [-lateral_tolerance, lateral_tolerance],  # x: small tolerance
            [-lateral_tolerance, lateral_tolerance],  # y: small tolerance
            [0, 0],                    # z: fixed position
            [0, 0],                    # roll: fixed
            [0, 0],                    # pitch: fixed
            [yaw_range[0], yaw_range[1]]  # yaw: configurable range
        ])
        
    else:
        raise ValueError(f'Unknown variant "{variant}". Must be "side_x", "side_y", "top", or "bottom"')
    
    return TSRTemplate(
        T_ref_tsr=T_ref_tsr,
        Tw_e=Tw_e,
        Bw=Bw,
        subject_entity=subject_entity,
        reference_entity=reference_entity,
        task_category=TaskCategory.GRASP,
        variant=variant,
        name=name,
        description=description
    )


def generate_place_template(
    subject_entity: EntityClass,
    reference_entity: EntityClass,
    variant: str,
    surface_height: float = 0.0,
    placement_tolerance: float = 0.1,
    orientation_tolerance: float = 0.2,
    name: str = "",
    description: str = ""
) -> TSRTemplate:
    """Generate a TSR template for placing an object on a surface.
    
    This function generates TSR templates for placing objects on surfaces
    like tables, shelves, or other flat surfaces. It supports different
    placement variants:
    - "on": Place object on top of surface
    - "in": Place object inside a container
    - "against": Place object against a wall
    
    Args:
        subject_entity: The entity being placed (e.g., mug, box)
        reference_entity: The surface/container being placed on (e.g., table, shelf)
        variant: Placement variant ("on", "in", "against")
        surface_height: Height of the surface above world origin
        placement_tolerance: Allowable lateral movement on surface
        orientation_tolerance: Allowable orientation variation in radians
        name: Optional name for the template
        description: Optional description of the template
        
    Returns:
        TSRTemplate for the specified placement variant
        
    Raises:
        ValueError: If parameters are invalid
    """
    if placement_tolerance < 0.0:
        raise ValueError('placement_tolerance must be >= 0')
    if orientation_tolerance < 0.0:
        raise ValueError('orientation_tolerance must be >= 0')
    
    # Generate name if not provided
    if not name:
        name = f"{subject_entity.value.title()} {variant.title()} Placement"
    
    # Generate description if not provided
    if not description:
        description = f"Place {subject_entity.value} {variant} {reference_entity.value}"
    
    # Set up transform matrices based on variant
    if variant == "on":
        # Place on top of surface
        T_ref_tsr = np.eye(4)
        T_ref_tsr[2, 3] = surface_height  # Surface at specified height
        
        Tw_e = np.array([
            [1, 0, 0, 0],                                      # x along surface
            [0, 1, 0, 0],                                      # y along surface
            [0, 0, 1, 0.02],                                   # 2cm above surface
            [0, 0, 0, 1]
        ])
        
        # Bounds: allow sliding on surface, small orientation tolerance
        Bw = np.array([
            [-placement_tolerance, placement_tolerance],  # x: sliding tolerance
            [-placement_tolerance, placement_tolerance],  # y: sliding tolerance
            [0, 0],                    # z: fixed height
            [-orientation_tolerance, orientation_tolerance],  # roll: small tolerance
            [-orientation_tolerance, orientation_tolerance],  # pitch: small tolerance
            [-orientation_tolerance, orientation_tolerance]   # yaw: small tolerance
        ])
        
    elif variant == "in":
        # Place inside container (simplified as "on" for now)
        T_ref_tsr = np.eye(4)
        T_ref_tsr[2, 3] = surface_height
        
        Tw_e = np.array([
            [1, 0, 0, 0],                                      # x along container
            [0, 1, 0, 0],                                      # y along container
            [0, 0, 1, 0.01],                                   # 1cm above bottom
            [0, 0, 0, 1]
        ])
        
        # Bounds: smaller tolerance for container placement
        Bw = np.array([
            [-placement_tolerance/2, placement_tolerance/2],  # x: smaller tolerance
            [-placement_tolerance/2, placement_tolerance/2],  # y: smaller tolerance
            [0, 0],                    # z: fixed height
            [-orientation_tolerance/2, orientation_tolerance/2],  # roll: smaller tolerance
            [-orientation_tolerance/2, orientation_tolerance/2],  # pitch: smaller tolerance
            [-orientation_tolerance/2, orientation_tolerance/2]   # yaw: smaller tolerance
        ])
        
    elif variant == "against":
        # Place against wall (simplified as side placement)
        T_ref_tsr = np.eye(4)
        T_ref_tsr[2, 3] = surface_height
        
        Tw_e = np.array([
            [0, 0, 1, 0.02],                                   # Approach from wall
            [1, 0, 0, 0],                                      # x along wall
            [0, 1, 0, 0],                                      # y along wall
            [0, 0, 0, 1]
        ])
        
        # Bounds: allow sliding along wall
        Bw = np.array([
            [0, 0],                    # x: fixed distance from wall
            [-placement_tolerance, placement_tolerance],  # y: sliding along wall
            [-placement_tolerance, placement_tolerance],  # z: vertical tolerance
            [-orientation_tolerance, orientation_tolerance],  # roll: small tolerance
            [-orientation_tolerance, orientation_tolerance],  # pitch: small tolerance
            [-orientation_tolerance, orientation_tolerance]   # yaw: small tolerance
        ])
        
    else:
        raise ValueError(f'Unknown variant "{variant}". Must be "on", "in", or "against"')
    
    return TSRTemplate(
        T_ref_tsr=T_ref_tsr,
        Tw_e=Tw_e,
        Bw=Bw,
        subject_entity=subject_entity,
        reference_entity=reference_entity,
        task_category=TaskCategory.PLACE,
        variant=variant,
        name=name,
        description=description
    )


def generate_transport_template(
    subject_entity: EntityClass,
    reference_entity: EntityClass,
    variant: str,
    roll_epsilon: float = 0.2,
    pitch_epsilon: float = 0.2,
    yaw_epsilon: float = 0.2,
    name: str = "",
    description: str = ""
) -> TSRTemplate:
    """Generate a TSR template for transporting an object.
    
    This function generates TSR templates for transporting objects while
    maintaining their orientation. It's useful for trajectory-wide constraints
    during object transport.
    
    Args:
        subject_entity: The entity being transported (e.g., mug, box)
        reference_entity: The reference frame (e.g., world, gripper)
        variant: Transport variant ("upright", "horizontal", "custom")
        roll_epsilon: Allowable roll variation in radians
        pitch_epsilon: Allowable pitch variation in radians
        yaw_epsilon: Allowable yaw variation in radians
        name: Optional name for the template
        description: Optional description of the template
        
    Returns:
        TSRTemplate for the specified transport variant
        
    Raises:
        ValueError: If parameters are invalid
    """
    if roll_epsilon < 0.0 or pitch_epsilon < 0.0 or yaw_epsilon < 0.0:
        raise ValueError('orientation tolerances must be >= 0')
    
    # Generate name if not provided
    if not name:
        name = f"{subject_entity.value.title()} {variant.title()} Transport"
    
    # Generate description if not provided
    if not description:
        description = f"Transport {subject_entity.value} in {variant} orientation"
    
    # Set up transform matrices
    T_ref_tsr = np.eye(4)
    Tw_e = np.eye(4)  # Identity transform for transport
    
    # Set up bounds based on variant
    if variant == "upright":
        # Keep object upright during transport
        Bw = np.array([
            [-100, 100],                # x: full reachability
            [-100, 100],                # y: full reachability
            [-100, 100],                # z: full reachability
            [-roll_epsilon, roll_epsilon],    # roll: small tolerance
            [-pitch_epsilon, pitch_epsilon],  # pitch: small tolerance
            [-yaw_epsilon, yaw_epsilon]       # yaw: small tolerance
        ])
        
    elif variant == "horizontal":
        # Keep object horizontal during transport
        Bw = np.array([
            [-100, 100],                # x: full reachability
            [-100, 100],                # y: full reachability
            [-100, 100],                # z: full reachability
            [-roll_epsilon, roll_epsilon],    # roll: small tolerance
            [-pitch_epsilon, pitch_epsilon],  # pitch: small tolerance
            [-yaw_epsilon, yaw_epsilon]       # yaw: small tolerance
        ])
        
    elif variant == "custom":
        # Custom orientation constraints
        Bw = np.array([
            [-100, 100],                # x: full reachability
            [-100, 100],                # y: full reachability
            [-100, 100],                # z: full reachability
            [-roll_epsilon, roll_epsilon],    # roll: custom tolerance
            [-pitch_epsilon, pitch_epsilon],  # pitch: custom tolerance
            [-yaw_epsilon, yaw_epsilon]       # yaw: custom tolerance
        ])
        
    else:
        raise ValueError(f'Unknown variant "{variant}". Must be "upright", "horizontal", or "custom"')
    
    return TSRTemplate(
        T_ref_tsr=T_ref_tsr,
        Tw_e=Tw_e,
        Bw=Bw,
        subject_entity=subject_entity,
        reference_entity=reference_entity,
        task_category=TaskCategory.PLACE,  # Using PLACE for transport constraints
        variant=variant,
        name=name,
        description=description
    )


# Convenience functions for common use cases
def generate_mug_grasp_template(
    subject_entity: EntityClass = EntityClass.GENERIC_GRIPPER,
    reference_entity: EntityClass = EntityClass.MUG,
    variant: str = "side",
    mug_radius: float = 0.04,
    mug_height: float = 0.12,
    **kwargs
) -> TSRTemplate:
    """Generate a TSR template for grasping a mug.
    
    Convenience function with default parameters for a typical mug.
    
    Args:
        subject_entity: The entity performing the grasp
        reference_entity: The mug being grasped
        variant: Grasp variant ("side", "top", "bottom")
        mug_radius: Radius of the mug in meters
        mug_height: Height of the mug in meters
        **kwargs: Additional arguments passed to generate_cylinder_grasp_template
        
    Returns:
        TSRTemplate for mug grasping
    """
    return generate_cylinder_grasp_template(
        subject_entity=subject_entity,
        reference_entity=reference_entity,
        variant=variant,
        cylinder_radius=mug_radius,
        cylinder_height=mug_height,
        **kwargs
    )


def generate_box_place_template(
    subject_entity: EntityClass = EntityClass.BOX,
    reference_entity: EntityClass = EntityClass.TABLE,
    variant: str = "on",
    **kwargs
) -> TSRTemplate:
    """Generate a TSR template for placing a box on a surface.
    
    Convenience function with default parameters for box placement.
    
    Args:
        subject_entity: The box being placed
        reference_entity: The surface being placed on
        variant: Placement variant ("on", "in", "against")
        **kwargs: Additional arguments passed to generate_place_template
        
    Returns:
        TSRTemplate for box placement
    """
    return generate_place_template(
        subject_entity=subject_entity,
        reference_entity=reference_entity,
        variant=variant,
        **kwargs
    )
