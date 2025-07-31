#!/usr/bin/env python

# Copyright (c) 2013, Carnegie Mellon University
# All rights reserved.
# Authors: Michael Koval <mkoval@cs.cmu.edu>
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
# - Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
# - Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# - Neither the name of Carnegie Mellon University nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""
TSR Library - Task Space Regions for Robotics

This library provides robot-agnostic Task Space Region (TSR) functionality
with simulator-specific wrappers for OpenRAVE and MuJoCo.

Core Classes:
    TSR: Robot-agnostic Task Space Region
    TSRChain: Chain of TSRs for complex constraints

Wrappers:
    OpenRAVE: OpenRAVE-specific adapters and functions
    MuJoCo: MuJoCo-specific adapters and functions (future)

Usage:
    # Core usage (robot-agnostic)
    from tsr.core import TSR, TSRChain
    
    # OpenRAVE usage
    from tsr.wrappers.openrave import OpenRAVERobotAdapter, place_object
    
    # Legacy usage (still supported)
    from tsr import TSR as LegacyTSR
"""

# Import core classes
from .core import TSR, TSRChain, wrap_to_interval, EPSILON

# Import wrapper interfaces
from .wrappers import (
    RobotInterface, 
    ObjectInterface, 
    EnvironmentInterface, 
    TSRWrapperFactory
)

# Import legacy classes for backward compatibility
try:
    import rodrigues, tsr, tsrlibrary
    from tsr import TSR as LegacyTSR, TSRChain as LegacyTSRChain
    _LEGACY_AVAILABLE = True
except ImportError:
    _LEGACY_AVAILABLE = False

# Import utility modules
try:
    from . import kin, rodrigues, util
    _UTILS_AVAILABLE = True
except ImportError:
    _UTILS_AVAILABLE = False

# Export all symbols
__all__ = [
    # Core classes
    'TSR',
    'TSRChain', 
    'wrap_to_interval',
    'EPSILON',
    
    # Wrapper interfaces
    'RobotInterface',
    'ObjectInterface',
    'EnvironmentInterface', 
    'TSRWrapperFactory'
]

# Add legacy classes if available
if _LEGACY_AVAILABLE:
    __all__.extend(['LegacyTSR', 'LegacyTSRChain'])

# Add utility modules if available
if _UTILS_AVAILABLE:
    __all__.extend(['kin', 'rodrigues', 'util', 'tsrlibrary'])

# Convenience functions for creating wrappers
def create_openrave_wrapper(robot, manip_idx: int):
    """Create an OpenRAVE wrapper for the given robot."""
    try:
        from .wrappers.openrave import OpenRAVERobotAdapter
        return OpenRAVERobotAdapter(robot)
    except ImportError:
        raise ImportError("OpenRAVE wrapper not available. Install OpenRAVE to use this function.")

def create_mujoco_wrapper(robot, manip_idx: int):
    """Create a MuJoCo wrapper for the given robot."""
    try:
        from .wrappers.mujoco import MuJoCoRobotAdapter
        return MuJoCoRobotAdapter(robot, manip_idx)
    except ImportError:
        raise ImportError("MuJoCo wrapper not available. Install MuJoCo to use this function.")

def create_tsr_library(robot, manip_idx: int, simulator_type: str = "openrave"):
    """
    Create a TSR library for the specified simulator.
    
    Args:
        robot: Robot object (simulator-specific)
        manip_idx: Index of the manipulator
        simulator_type: Type of simulator ('openrave' or 'mujoco')
        
    Returns:
        TSR library instance
    """
    if simulator_type == "openrave":
        return create_openrave_wrapper(robot, manip_idx)
    elif simulator_type == "mujoco":
        return create_mujoco_wrapper(robot, manip_idx)
    else:
        raise ValueError(f"Unknown simulator type: {simulator_type}. Use 'openrave' or 'mujoco'")

# Add convenience functions to exports
__all__.extend([
    'create_openrave_wrapper',
    'create_mujoco_wrapper', 
    'create_tsr_library'
])
