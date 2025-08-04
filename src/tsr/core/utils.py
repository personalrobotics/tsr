# SPDX-License-Identifier: BSD-2-Clause
# Authors: Siddhartha Srinivasa and contributors to TSR

import numpy as np
from numpy import pi

EPSILON = 0.001


def wrap_to_interval(angles: np.ndarray, lower: np.ndarray = None) -> np.ndarray:
    """
    Wrap a vector of angles to a continuous interval starting at `lower`.

    Args:
        angles: (N,) array of angles (in radians)
        lower: (N,) array of lower bounds; defaults to -pi if None

    Returns:
        wrapped: (N,) array of wrapped angles
    """
    if lower is None:
        lower = -pi * np.ones_like(angles)
    return (angles - lower) % (2 * pi) + lower


def geodesic_error(t1: np.ndarray, t2: np.ndarray) -> np.ndarray:
    """
    Compute the error in global coordinates between two transforms.

    Args:
        t1: current transform (4x4)
        t2: goal transform (4x4)

    Returns:
        error: 4-vector [dx, dy, dz, solid angle]
    """
    trel = np.dot(np.linalg.inv(t1), t2)
    trans = np.dot(t1[0:3, 0:3], trel[0:3, 3])

    # Extract rotation error (simplified - just use the rotation matrix)
    # For a more accurate geodesic distance, we'd need to extract the rotation angle
    # For now, use a simple approximation
    angle_error = np.linalg.norm(trel[0:3, 0:3] - np.eye(3))

    return np.hstack((trans, angle_error))


def geodesic_distance(t1: np.ndarray, t2: np.ndarray, r: float = 1.0) -> float:
    """
    Compute the geodesic distance between two transforms.

    Args:
        t1: current transform (4x4)
        t2: goal transform (4x4)
        r: in units of meters/radians converts radians to meters

    Returns:
        distance: geodesic distance
    """
    error = geodesic_error(t1, t2)
    error[3] = r * error[3]
    return np.linalg.norm(error)
