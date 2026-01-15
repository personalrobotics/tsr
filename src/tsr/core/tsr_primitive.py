# SPDX-License-Identifier: BSD-2-Clause
# Authors: Siddhartha Srinivasa and contributors to TSR

"""
TSR Primitive Parser

Converts human-friendly geometric primitives to raw TSR bounds.

The 9 supported primitives:
- Cartesian: point, line, plane, box
- Cylindrical: ring, disk, cylinder, shell
- Spherical: sphere

Units: meters for distance, degrees for angles.
"""

from __future__ import annotations

import numpy as np
from numpy import pi
from typing import Dict, Any, Tuple, Optional, Union
from dataclasses import dataclass


def deg2rad(deg: float) -> float:
    """Convert degrees to radians."""
    return deg * pi / 180.0


def ensure_range(val: Any) -> Tuple[float, float]:
    """Convert a value or [min, max] list to a (min, max) tuple."""
    if isinstance(val, (list, tuple)):
        return (float(val[0]), float(val[1]))
    return (float(val), float(val))


def ensure_range_deg(val: Any) -> Tuple[float, float]:
    """Convert a degree value or range to radians."""
    if isinstance(val, (list, tuple)):
        return (deg2rad(float(val[0])), deg2rad(float(val[1])))
    return (deg2rad(float(val)), deg2rad(float(val)))


# =============================================================================
# Position Primitives
# =============================================================================

def parse_point(params: Dict[str, Any]) -> np.ndarray:
    """
    Parse a point primitive.

    Args:
        params: dict with x, y, z values

    Returns:
        6x2 Bw array with fixed position, zero rotation
    """
    x = float(params.get('x', 0))
    y = float(params.get('y', 0))
    z = float(params.get('z', 0))

    Bw = np.zeros((6, 2))
    Bw[0, :] = [x, x]
    Bw[1, :] = [y, y]
    Bw[2, :] = [z, z]
    return Bw


def parse_line(params: Dict[str, Any]) -> np.ndarray:
    """
    Parse a line primitive.

    Args:
        params: dict with axis ('x', 'y', or 'z') and range

    Returns:
        6x2 Bw array
    """
    axis = params.get('axis', 'z')
    range_val = ensure_range(params.get('range', [0, 0]))

    Bw = np.zeros((6, 2))

    axis_map = {'x': 0, 'y': 1, 'z': 2}
    axis_idx = axis_map.get(axis, 2)

    Bw[axis_idx, :] = range_val

    # Set fixed values for other axes
    for i in range(3):
        if i != axis_idx:
            val = float(params.get(['x', 'y', 'z'][i], 0))
            Bw[i, :] = [val, val]

    return Bw


def parse_plane(params: Dict[str, Any]) -> np.ndarray:
    """
    Parse a plane primitive.

    Args:
        params: dict with two varying axes and one fixed

    Returns:
        6x2 Bw array
    """
    Bw = np.zeros((6, 2))

    for i, axis in enumerate(['x', 'y', 'z']):
        if axis in params:
            val = params[axis]
            r = ensure_range(val)
            Bw[i, :] = r

    return Bw


def parse_box(params: Dict[str, Any]) -> np.ndarray:
    """
    Parse a box primitive.

    Args:
        params: dict with x, y, z ranges

    Returns:
        6x2 Bw array
    """
    Bw = np.zeros((6, 2))

    Bw[0, :] = ensure_range(params.get('x', 0))
    Bw[1, :] = ensure_range(params.get('y', 0))
    Bw[2, :] = ensure_range(params.get('z', 0))

    return Bw


def parse_ring(params: Dict[str, Any]) -> np.ndarray:
    """
    Parse a ring primitive (circle/arc around axis).

    Args:
        params: dict with axis, radius, angle (degrees), and optional height

    Returns:
        6x2 Bw array
    """
    axis = params.get('axis', 'z')
    radius = float(params.get('radius', 0))
    angle = ensure_range_deg(params.get('angle', [0, 360]))
    height = float(params.get('height', 0))

    Bw = np.zeros((6, 2))

    if axis == 'z':
        # Ring around z-axis: x=radius, y=0, z=height, yaw varies
        Bw[0, :] = [radius, radius]
        Bw[1, :] = [0, 0]
        Bw[2, :] = [height, height]
        Bw[5, :] = angle  # yaw
    elif axis == 'x':
        # Ring around x-axis: x=height, y=radius, z=0, roll varies
        Bw[0, :] = [height, height]
        Bw[1, :] = [radius, radius]
        Bw[2, :] = [0, 0]
        Bw[3, :] = angle  # roll
    elif axis == 'y':
        # Ring around y-axis: x=radius, y=height, z=0, pitch varies
        Bw[0, :] = [radius, radius]
        Bw[1, :] = [height, height]
        Bw[2, :] = [0, 0]
        Bw[4, :] = angle  # pitch

    return Bw


def parse_disk(params: Dict[str, Any]) -> np.ndarray:
    """
    Parse a disk primitive (filled circle/annulus).

    Args:
        params: dict with axis, radius (range), angle (degrees), and optional height

    Returns:
        6x2 Bw array
    """
    axis = params.get('axis', 'z')
    radius = ensure_range(params.get('radius', [0, 0]))
    angle = ensure_range_deg(params.get('angle', [0, 360]))
    height = float(params.get('height', 0))

    Bw = np.zeros((6, 2))

    if axis == 'z':
        Bw[0, :] = radius  # x = radius range
        Bw[1, :] = [0, 0]
        Bw[2, :] = [height, height]
        Bw[5, :] = angle  # yaw
    elif axis == 'x':
        Bw[0, :] = [height, height]
        Bw[1, :] = radius
        Bw[2, :] = [0, 0]
        Bw[3, :] = angle  # roll
    elif axis == 'y':
        Bw[0, :] = radius
        Bw[1, :] = [height, height]
        Bw[2, :] = [0, 0]
        Bw[4, :] = angle  # pitch

    return Bw


def parse_cylinder(params: Dict[str, Any]) -> np.ndarray:
    """
    Parse a cylinder primitive (cylinder surface around axis).

    Args:
        params: dict with axis, radius, height (range), angle (degrees)

    Returns:
        6x2 Bw array
    """
    axis = params.get('axis', 'z')
    radius = float(params.get('radius', 0))
    height = ensure_range(params.get('height', [0, 0]))
    angle = ensure_range_deg(params.get('angle', [0, 360]))

    Bw = np.zeros((6, 2))

    if axis == 'z':
        Bw[0, :] = [radius, radius]
        Bw[1, :] = [0, 0]
        Bw[2, :] = height
        Bw[5, :] = angle  # yaw
    elif axis == 'x':
        Bw[0, :] = height
        Bw[1, :] = [radius, radius]
        Bw[2, :] = [0, 0]
        Bw[3, :] = angle  # roll
    elif axis == 'y':
        Bw[0, :] = [radius, radius]
        Bw[1, :] = height
        Bw[2, :] = [0, 0]
        Bw[4, :] = angle  # pitch

    return Bw


def parse_shell(params: Dict[str, Any]) -> np.ndarray:
    """
    Parse a shell primitive (thick-walled cylinder).

    Args:
        params: dict with axis, radius (range), height (range), angle (degrees)

    Returns:
        6x2 Bw array
    """
    axis = params.get('axis', 'z')
    radius = ensure_range(params.get('radius', [0, 0]))
    height = ensure_range(params.get('height', [0, 0]))
    angle = ensure_range_deg(params.get('angle', [0, 360]))

    Bw = np.zeros((6, 2))

    if axis == 'z':
        Bw[0, :] = radius
        Bw[1, :] = [0, 0]
        Bw[2, :] = height
        Bw[5, :] = angle  # yaw
    elif axis == 'x':
        Bw[0, :] = height
        Bw[1, :] = radius
        Bw[2, :] = [0, 0]
        Bw[3, :] = angle  # roll
    elif axis == 'y':
        Bw[0, :] = radius
        Bw[1, :] = height
        Bw[2, :] = [0, 0]
        Bw[4, :] = angle  # pitch

    return Bw


def parse_sphere(params: Dict[str, Any]) -> np.ndarray:
    """
    Parse a sphere primitive.

    Args:
        params: dict with radius, pitch (range), yaw (range) in degrees

    Returns:
        6x2 Bw array
    """
    radius = float(params.get('radius', 0))
    pitch = ensure_range_deg(params.get('pitch', [-90, 90]))
    yaw = ensure_range_deg(params.get('yaw', [0, 360]))

    Bw = np.zeros((6, 2))
    Bw[0, :] = [radius, radius]
    Bw[1, :] = [0, 0]
    Bw[2, :] = [0, 0]
    Bw[4, :] = pitch
    Bw[5, :] = yaw

    return Bw


def parse_raw(params: Dict[str, Any]) -> np.ndarray:
    """
    Parse raw DOF specification for non-standard primitives.

    Args:
        params: dict with x, y, z, roll, pitch, yaw (values or ranges)
               Angles in degrees.

    Returns:
        6x2 Bw array
    """
    Bw = np.zeros((6, 2))

    # Position DOFs (meters)
    Bw[0, :] = ensure_range(params.get('x', 0))
    Bw[1, :] = ensure_range(params.get('y', 0))
    Bw[2, :] = ensure_range(params.get('z', 0))

    # Rotation DOFs (degrees -> radians)
    Bw[3, :] = ensure_range_deg(params.get('roll', 0))
    Bw[4, :] = ensure_range_deg(params.get('pitch', 0))
    Bw[5, :] = ensure_range_deg(params.get('yaw', 0))

    return Bw


# Primitive parser dispatch
POSITION_PARSERS = {
    'point': parse_point,
    'line': parse_line,
    'plane': parse_plane,
    'box': parse_box,
    'ring': parse_ring,
    'disk': parse_disk,
    'cylinder': parse_cylinder,
    'shell': parse_shell,
    'sphere': parse_sphere,
    'raw': parse_raw,
}


def parse_position(position: Dict[str, Any]) -> np.ndarray:
    """
    Parse a position specification into Bw bounds.

    Args:
        position: dict with 'type' and primitive-specific parameters

    Returns:
        6x2 Bw array
    """
    ptype = position.get('type', 'raw')
    parser = POSITION_PARSERS.get(ptype)

    if parser is None:
        raise ValueError(f"Unknown position primitive type: {ptype}")

    return parser(position)


# =============================================================================
# Orientation and Approach
# =============================================================================

def rotation_matrix_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """Create rotation matrix from axis-angle representation."""
    axis = axis / np.linalg.norm(axis)
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    return R


def approach_to_rotation(approach: str, axis: str = 'z') -> np.ndarray:
    """
    Convert approach direction to rotation matrix.

    The rotation orients the end-effector so its z-axis points
    in the approach direction.

    Args:
        approach: One of 'radial', 'axial', '+x', '-x', '+y', '-y', '+z', '-z'
        axis: Reference axis for radial/axial (default 'z')

    Returns:
        3x3 rotation matrix
    """
    if approach == 'radial':
        # Gripper z-axis points toward the reference axis (inward)
        # For z-axis reference: approach from +x direction
        if axis == 'z':
            # Rotate so gripper -z points toward cylinder center
            # i.e., gripper z points outward (+x in TSR frame)
            return np.array([
                [0, 0, 1],
                [1, 0, 0],
                [0, 1, 0]
            ])
        elif axis == 'x':
            return np.array([
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 0]
            ])
        elif axis == 'y':
            return np.array([
                [1, 0, 0],
                [0, 0, 1],
                [0, 1, 0]
            ])
    elif approach == 'axial':
        # Gripper z-axis points along the reference axis
        if axis == 'z':
            return np.eye(3)
        elif axis == 'x':
            return np.array([
                [0, 0, 1],
                [0, 1, 0],
                [-1, 0, 0]
            ])
        elif axis == 'y':
            return np.array([
                [1, 0, 0],
                [0, 0, 1],
                [0, -1, 0]
            ])
    elif approach == '+z':
        return np.eye(3)
    elif approach == '-z':
        # Flip: gripper z points down (-z world)
        return np.array([
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, -1]
        ])
    elif approach == '+x':
        return np.array([
            [0, 0, 1],
            [0, 1, 0],
            [-1, 0, 0]
        ])
    elif approach == '-x':
        return np.array([
            [0, 0, -1],
            [0, 1, 0],
            [1, 0, 0]
        ])
    elif approach == '+y':
        return np.array([
            [1, 0, 0],
            [0, 0, 1],
            [0, -1, 0]
        ])
    elif approach == '-y':
        return np.array([
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0]
        ])
    else:
        raise ValueError(f"Unknown approach direction: {approach}")

    return np.eye(3)


def parse_orientation(orientation: Dict[str, Any], position: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse orientation specification.

    Args:
        orientation: dict with approach and optional freedom specs
        position: position dict (needed to get axis for radial/axial)

    Returns:
        Tuple of (3x3 rotation matrix for Tw_e, 6x2 orientation bounds to add to Bw)
    """
    approach = orientation.get('approach', '+z')
    axis = position.get('axis', 'z')

    R = approach_to_rotation(approach, axis)

    # Parse orientation freedoms
    Bw_orient = np.zeros((6, 2))

    # Roll freedom (rotation around gripper x-axis)
    if 'roll' in orientation:
        roll = orientation['roll']
        if roll == 'free':
            Bw_orient[3, :] = [-pi, pi]
        else:
            Bw_orient[3, :] = ensure_range_deg(roll)

    # Pitch freedom (rotation around gripper y-axis)
    if 'pitch' in orientation:
        pitch = orientation['pitch']
        if pitch == 'free':
            Bw_orient[4, :] = [-pi, pi]
        else:
            Bw_orient[4, :] = ensure_range_deg(pitch)

    # Yaw freedom (rotation around gripper z-axis)
    if 'yaw' in orientation:
        yaw = orientation['yaw']
        if yaw == 'free':
            Bw_orient[5, :] = [-pi, pi]
        else:
            Bw_orient[5, :] = ensure_range_deg(yaw)

    return R, Bw_orient


def build_Tw_e(rotation: np.ndarray, standoff: float, approach: str = '-z') -> np.ndarray:
    """
    Build the Tw_e transform from rotation and standoff.

    Args:
        rotation: 3x3 rotation matrix
        standoff: Distance along approach direction
        approach: Approach direction for offset

    Returns:
        4x4 transform matrix
    """
    Tw_e = np.eye(4)
    Tw_e[0:3, 0:3] = rotation

    # Offset along the gripper's z-axis (approach direction)
    # Negative because we want gripper to be standoff distance away
    Tw_e[0:3, 3] = rotation @ np.array([0, 0, -standoff])

    return Tw_e


# =============================================================================
# Main Template Parser
# =============================================================================

@dataclass
class ParsedTemplate:
    """Result of parsing a template specification."""
    name: str
    description: str
    task: str
    subject: str
    reference: str
    T_ref_tsr: np.ndarray
    Tw_e: np.ndarray
    Bw: np.ndarray
    gripper: Optional[Dict[str, Any]] = None
    reference_frame: Optional[str] = None  # e.g., "bottom", "handle", None = object origin


def parse_template(spec: Dict[str, Any]) -> ParsedTemplate:
    """
    Parse a human-friendly template specification into TSR components.

    Args:
        spec: Template specification dict

    Returns:
        ParsedTemplate with all TSR components
    """
    name = spec.get('name', '')
    description = spec.get('description', '')
    task = spec.get('task', '')
    subject = spec.get('subject', 'gripper')
    reference = spec.get('reference', '')
    reference_frame = spec.get('reference_frame', None)  # None = object origin

    # Parse position primitive
    position = spec.get('position', {'type': 'point'})
    Bw = parse_position(position)

    # Parse orientation
    orientation = spec.get('orientation', {})
    if orientation:
        R, Bw_orient = parse_orientation(orientation, position)
        # Merge orientation bounds (orientation freedoms add to what position set)
        for i in range(3, 6):
            if Bw_orient[i, 0] != 0 or Bw_orient[i, 1] != 0:
                Bw[i, :] = Bw_orient[i, :]
    else:
        R = np.eye(3)

    # Build Tw_e from orientation and standoff
    standoff = float(spec.get('standoff', 0))
    approach = orientation.get('approach', '-z') if orientation else '-z'
    Tw_e = build_Tw_e(R, standoff, approach)

    # T_ref_tsr is identity by default (TSR frame = reference frame)
    T_ref_tsr = np.eye(4)

    # Gripper info
    gripper = spec.get('gripper', None)

    return ParsedTemplate(
        name=name,
        description=description,
        task=task,
        subject=subject,
        reference=reference,
        T_ref_tsr=T_ref_tsr,
        Tw_e=Tw_e,
        Bw=Bw,
        gripper=gripper,
        reference_frame=reference_frame,
    )


def load_template_yaml(yaml_str: str) -> ParsedTemplate:
    """
    Load a template from YAML string.

    Args:
        yaml_str: YAML string containing template specification

    Returns:
        ParsedTemplate
    """
    import yaml
    spec = yaml.safe_load(yaml_str)
    return parse_template(spec)


def load_template_file(filepath: str) -> ParsedTemplate:
    """
    Load a template from a YAML file.

    Args:
        filepath: Path to YAML file

    Returns:
        ParsedTemplate
    """
    with open(filepath, 'r') as f:
        return load_template_yaml(f.read())
