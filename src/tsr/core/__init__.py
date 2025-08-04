# SPDX-License-Identifier: BSD-2-Clause
# Authors: Siddhartha Srinivasa and contributors to TSR

"""
Core TSR module — robot-agnostic Task Space Region implementation.

This module provides the fundamental TSR classes that are independent
of any specific robot or simulator.
"""

from .tsr import TSR
from .tsr_chain import TSRChain
from .utils import EPSILON, geodesic_distance, geodesic_error, wrap_to_interval

__all__ = [
    "TSR",
    "TSRChain",
    "wrap_to_interval",
    "EPSILON",
    "geodesic_distance",
    "geodesic_error",
]
