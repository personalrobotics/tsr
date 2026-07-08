# SPDX-License-Identifier: BSD-2-Clause
# Authors: Siddhartha Srinivasa and contributors to TSR

import numpy as np
from numpy import pi

from gafropy import Motor

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


def geodesic_error(t1, t2) -> np.ndarray:
    """
    Compute the geodesic error between two transforms on SE(3).

    The error is computed as:
    - Translation error: the Euclidean distance between positions
    - Rotation error: the angle of the relative rotation, taken from the rotor
      of the relative ``Motor`` (m1^-1 * m2).

    Args:
        t1: first transform (Motor or 4x4 matrix)
        t2: second transform (Motor or 4x4 matrix)

    Returns:
        error: 4-vector [dx, dy, dz, rotation_angle]
               where dx, dy, dz are in meters and rotation_angle is in radians
    """
    m1, m2 = Motor(t1), Motor(t2)

    # Translation error (in world frame), from the translators directly.
    p1, p2 = m1.get_translator(), m2.get_translator()
    trans_error = np.array([p2.x() - p1.x(), p2.y() - p1.y(), p2.z() - p1.z()])

    # Rotation error: the *minimal* angle of the relative rotation. ``Rotor.angle``
    # is unwrapped (grows past pi), which makes the error discontinuous at pi and
    # breaks gradient-based callers; the rotor bivector-log norm is the minimal
    # angle in [0, pi] and is smooth, so we use that.
    rel = m1.inverse().multiply(m2)
    angle_error = float(np.linalg.norm(rel.get_rotor().log().to_array()))

    return np.hstack((trans_error, angle_error))


def geodesic_distance(t1, t2, r: float = 1.0) -> float:
    """
    Compute the geodesic distance between two transforms on SE(3).

    The distance combines translation (meters) and rotation (radians) errors.
    The parameter `r` allows weighting rotation relative to translation.

    As noted in Section 4.2 of Berenson et al. 2011:
    "we implicitly weigh rotation in radians and translation in meters equally
    when computing ||Δx||, but the two types of units can be weighed in an
    arbitrary manner"

    Args:
        t1: first transform (Motor or 4x4 matrix)
        t2: second transform (Motor or 4x4 matrix)
        r: weight for rotation in units of meters/radian (default 1.0)
           Higher values penalize rotation errors more.

    Returns:
        distance: weighted geodesic distance
    """
    error = geodesic_error(t1, t2)
    # Weight the rotation component
    error[3] = r * error[3]
    return np.linalg.norm(error)
