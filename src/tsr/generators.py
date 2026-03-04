"""Generic TSR template generators for primitive objects.

This module provides functions to generate TSR templates for common primitive
objects (cylinders, boxes, spheres) and common tasks (grasping, placing, transport).
All functions are simulator-agnostic and return TSRTemplate objects.

Gripper frame convention:
    z = approach direction (toward object surface)
    y = finger opening direction
    x = palm normal (right-hand rule: x = y × z)
"""

import numpy as np
from typing import Optional, Tuple
from .template import TSRTemplate


def generate_cylinder_grasp_template(
    subject: str,
    reference: str,
    variant: str,
    cylinder_radius: float,
    cylinder_height: float,
    approach_distance: float = 0.05,
    vertical_tolerance: float = 0.02,
    yaw_range: Optional[Tuple[float, float]] = None,
    name: str = "",
    description: str = "",
    preshape: Optional[np.ndarray] = None
) -> TSRTemplate:
    """Generate a TSR template for grasping a cylindrical object.

    Args:
        subject: The entity performing the grasp (e.g., "gripper")
        reference: The entity being grasped (e.g., "mug", "bottle")
        variant: Grasp variant ("side", "top", "bottom")
        cylinder_radius: Radius of the cylinder in meters
        cylinder_height: Height of the cylinder in meters
        approach_distance: Distance from cylinder surface to end-effector
        vertical_tolerance: Allowable vertical movement during grasp
        yaw_range: Allowable yaw rotation range (min, max) in radians
        name: Optional name for the template
        description: Optional description of the template
        preshape: Optional gripper configuration as DOF values

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

    if yaw_range is None:
        yaw_range = (-np.pi, np.pi)

    if not name:
        name = f"{reference.title()} {variant.title()} Grasp"

    if not description:
        description = f"{variant.title()} grasp for {reference} with {approach_distance*1000:.0f}mm approach distance"

    if variant == "side":
        T_ref_tsr = np.eye(4)
        Tw_e = np.array([
            [0, 0, 1, -(cylinder_radius + approach_distance)],
            [1, 0, 0, 0],
            [0, 1, 0, cylinder_height * 0.5],
            [0, 0, 0, 1]
        ])
        Bw = np.array([
            [0, 0],
            [0, 0],
            [-vertical_tolerance, vertical_tolerance],
            [0, 0],
            [0, 0],
            [yaw_range[0], yaw_range[1]]
        ])

    elif variant == "top":
        T_ref_tsr = np.eye(4)
        Tw_e = np.array([
            [0, 0, 1, -approach_distance],
            [1, 0, 0, 0],
            [0, 1, 0, cylinder_height],
            [0, 0, 0, 1]
        ])
        Bw = np.array([
            [-vertical_tolerance, vertical_tolerance],
            [-vertical_tolerance, vertical_tolerance],
            [0, 0],
            [0, 0],
            [0, 0],
            [yaw_range[0], yaw_range[1]]
        ])

    elif variant == "bottom":
        T_ref_tsr = np.eye(4)
        Tw_e = np.array([
            [0, 0, -1, approach_distance],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ])
        Bw = np.array([
            [-vertical_tolerance, vertical_tolerance],
            [-vertical_tolerance, vertical_tolerance],
            [0, 0],
            [0, 0],
            [0, 0],
            [yaw_range[0], yaw_range[1]]
        ])

    else:
        raise ValueError(f'Unknown variant "{variant}". Must be "side", "top", or "bottom"')

    return TSRTemplate(
        T_ref_tsr=T_ref_tsr,
        Tw_e=Tw_e,
        Bw=Bw,
        task="grasp",
        subject=subject,
        reference=reference,
        variant=variant,
        name=name,
        description=description,
        preshape=preshape
    )


def generate_box_grasp_template(
    subject: str,
    reference: str,
    variant: str,
    box_length: float,
    box_width: float,
    box_height: float,
    approach_distance: float = 0.05,
    lateral_tolerance: float = 0.02,
    yaw_range: Optional[Tuple[float, float]] = None,
    name: str = "",
    description: str = "",
    preshape: Optional[np.ndarray] = None
) -> TSRTemplate:
    """Generate a TSR template for grasping a box-shaped object.

    Args:
        subject: The entity performing the grasp (e.g., "gripper")
        reference: The entity being grasped (e.g., "box", "book")
        variant: Grasp variant ("side_x", "side_y", "top", "bottom")
        box_length: Length of the box in meters (x dimension)
        box_width: Width of the box in meters (y dimension)
        box_height: Height of the box in meters (z dimension)
        approach_distance: Distance from box surface to end-effector
        lateral_tolerance: Allowable lateral movement during grasp
        yaw_range: Allowable yaw rotation range (min, max) in radians
        name: Optional name for the template
        description: Optional description of the template
        preshape: Optional gripper configuration as DOF values

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

    if yaw_range is None:
        yaw_range = (-np.pi, np.pi)

    if not name:
        name = f"{reference.title()} {variant.title()} Grasp"

    if not description:
        description = f"{variant.title()} grasp for {reference} with {approach_distance*1000:.0f}mm approach distance"

    if variant == "side_x":
        T_ref_tsr = np.eye(4)
        Tw_e = np.array([
            [-1, 0, 0, -(box_length/2 + approach_distance)],
            [0, 1, 0, 0],
            [0, 0, 1, box_height/2],
            [0, 0, 0, 1]
        ])
        Bw = np.array([
            [0, 0],
            [-lateral_tolerance, lateral_tolerance],
            [-lateral_tolerance, lateral_tolerance],
            [0, 0],
            [0, 0],
            [yaw_range[0], yaw_range[1]]
        ])

    elif variant == "side_y":
        T_ref_tsr = np.eye(4)
        Tw_e = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, -(box_width/2 + approach_distance)],
            [0, 0, 1, box_height/2],
            [0, 0, 0, 1]
        ])
        Bw = np.array([
            [-lateral_tolerance, lateral_tolerance],
            [0, 0],
            [-lateral_tolerance, lateral_tolerance],
            [0, 0],
            [0, 0],
            [yaw_range[0], yaw_range[1]]
        ])

    elif variant == "top":
        T_ref_tsr = np.eye(4)
        Tw_e = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, box_height + approach_distance],
            [0, 0, 0, 1]
        ])
        Bw = np.array([
            [-lateral_tolerance, lateral_tolerance],
            [-lateral_tolerance, lateral_tolerance],
            [0, 0],
            [0, 0],
            [0, 0],
            [yaw_range[0], yaw_range[1]]
        ])

    elif variant == "bottom":
        T_ref_tsr = np.eye(4)
        Tw_e = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, -1, -approach_distance],
            [0, 0, 0, 1]
        ])
        Bw = np.array([
            [-lateral_tolerance, lateral_tolerance],
            [-lateral_tolerance, lateral_tolerance],
            [0, 0],
            [0, 0],
            [0, 0],
            [yaw_range[0], yaw_range[1]]
        ])

    else:
        raise ValueError(f'Unknown variant "{variant}". Must be "side_x", "side_y", "top", or "bottom"')

    return TSRTemplate(
        T_ref_tsr=T_ref_tsr,
        Tw_e=Tw_e,
        Bw=Bw,
        task="grasp",
        subject=subject,
        reference=reference,
        variant=variant,
        name=name,
        description=description,
        preshape=preshape
    )


def generate_place_template(
    subject: str,
    reference: str,
    variant: str,
    surface_height: float = 0.0,
    placement_tolerance: float = 0.1,
    orientation_tolerance: float = 0.2,
    name: str = "",
    description: str = ""
) -> TSRTemplate:
    """Generate a TSR template for placing an object on a surface.

    Args:
        subject: The entity being placed (e.g., "mug", "box")
        reference: The surface/container being placed on (e.g., "table", "shelf")
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

    if not name:
        name = f"{subject.title()} {variant.title()} Placement"

    if not description:
        description = f"Place {subject} {variant} {reference}"

    if variant == "on":
        T_ref_tsr = np.eye(4)
        T_ref_tsr[2, 3] = surface_height
        Tw_e = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0.02],
            [0, 0, 0, 1]
        ])
        Bw = np.array([
            [-placement_tolerance, placement_tolerance],
            [-placement_tolerance, placement_tolerance],
            [0, 0],
            [-orientation_tolerance, orientation_tolerance],
            [-orientation_tolerance, orientation_tolerance],
            [-orientation_tolerance, orientation_tolerance]
        ])

    elif variant == "in":
        T_ref_tsr = np.eye(4)
        T_ref_tsr[2, 3] = surface_height
        Tw_e = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0.01],
            [0, 0, 0, 1]
        ])
        Bw = np.array([
            [-placement_tolerance/2, placement_tolerance/2],
            [-placement_tolerance/2, placement_tolerance/2],
            [0, 0],
            [-orientation_tolerance/2, orientation_tolerance/2],
            [-orientation_tolerance/2, orientation_tolerance/2],
            [-orientation_tolerance/2, orientation_tolerance/2]
        ])

    elif variant == "against":
        T_ref_tsr = np.eye(4)
        T_ref_tsr[2, 3] = surface_height
        Tw_e = np.array([
            [0, 0, 1, 0.02],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ])
        Bw = np.array([
            [0, 0],
            [-placement_tolerance, placement_tolerance],
            [-placement_tolerance, placement_tolerance],
            [-orientation_tolerance, orientation_tolerance],
            [-orientation_tolerance, orientation_tolerance],
            [-orientation_tolerance, orientation_tolerance]
        ])

    else:
        raise ValueError(f'Unknown variant "{variant}". Must be "on", "in", or "against"')

    return TSRTemplate(
        T_ref_tsr=T_ref_tsr,
        Tw_e=Tw_e,
        Bw=Bw,
        task="place",
        subject=subject,
        reference=reference,
        variant=variant,
        name=name,
        description=description
    )


def generate_transport_template(
    subject: str,
    reference: str,
    variant: str,
    roll_epsilon: float = 0.2,
    pitch_epsilon: float = 0.2,
    yaw_epsilon: float = 0.2,
    name: str = "",
    description: str = ""
) -> TSRTemplate:
    """Generate a TSR template for transporting an object.

    Useful for trajectory-wide constraints during object transport.

    Args:
        subject: The entity being transported (e.g., "mug", "box")
        reference: The reference frame (e.g., "world", "gripper")
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

    if not name:
        name = f"{subject.title()} {variant.title()} Transport"

    if not description:
        description = f"Transport {subject} in {variant} orientation"

    if variant not in ("upright", "horizontal", "custom"):
        raise ValueError(f'Unknown variant "{variant}". Must be "upright", "horizontal", or "custom"')

    T_ref_tsr = np.eye(4)
    Tw_e = np.eye(4)
    Bw = np.array([
        [-100, 100],
        [-100, 100],
        [-100, 100],
        [-roll_epsilon, roll_epsilon],
        [-pitch_epsilon, pitch_epsilon],
        [-yaw_epsilon, yaw_epsilon]
    ])

    return TSRTemplate(
        T_ref_tsr=T_ref_tsr,
        Tw_e=Tw_e,
        Bw=Bw,
        task="constrain",
        subject=subject,
        reference=reference,
        variant=variant,
        name=name,
        description=description
    )


# Convenience functions for common use cases
def generate_mug_grasp_template(
    subject: str = "gripper",
    reference: str = "mug",
    variant: str = "side",
    mug_radius: float = 0.04,
    mug_height: float = 0.12,
    **kwargs
) -> TSRTemplate:
    """Generate a TSR template for grasping a mug.

    Convenience function with default parameters for a typical mug.
    """
    return generate_cylinder_grasp_template(
        subject=subject,
        reference=reference,
        variant=variant,
        cylinder_radius=mug_radius,
        cylinder_height=mug_height,
        **kwargs
    )


def generate_box_place_template(
    subject: str = "box",
    reference: str = "table",
    variant: str = "on",
    **kwargs
) -> TSRTemplate:
    """Generate a TSR template for placing a box on a surface.

    Convenience function with default parameters for box placement.
    """
    return generate_place_template(
        subject=subject,
        reference=reference,
        variant=variant,
        **kwargs
    )
