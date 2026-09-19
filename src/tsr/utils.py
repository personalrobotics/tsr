# SPDX-License-Identifier: BSD-2-Clause
# Authors: Siddhartha Srinivasa and contributors to TSR

"""Back-compat shim: the math helpers now live in :mod:`tsr.core.utils`.

Importing ``from tsr.utils import wrap_to_interval`` keeps working. New code
should prefer ``from tsr.core import ...``.
"""

from .core.utils import (
    EPSILON,
    geodesic_distance,
    geodesic_error,
    rotation_angle,
    wrap_to_interval,
)

__all__ = [
    "EPSILON",
    "wrap_to_interval",
    "rotation_angle",
    "geodesic_error",
    "geodesic_distance",
]
