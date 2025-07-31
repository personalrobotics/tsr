# SPDX-License-Identifier: BSD-2-Clause
# Authors: Siddhartha Srinivasa and contributors to TSR

"""
TSR Wrappers package.

This package provides simulator-specific adapters for the TSR library.
"""

from .base import (
    RobotInterface, 
    ObjectInterface, 
    EnvironmentInterface, 
    TSRWrapperFactory
)

# Import OpenRAVE wrapper
try:
    from .openrave import (
        OpenRAVERobotAdapter,
        OpenRAVEObjectAdapter,
        OpenRAVEEnvironmentAdapter,
        place_object,
        transport_upright,
        cylinder_grasp,
        box_grasp
    )
    
    # Register OpenRAVE wrapper with factory
    TSRWrapperFactory.register_wrapper('openrave', OpenRAVERobotAdapter)
    
    _OPENRAVE_AVAILABLE = True
except ImportError:
    _OPENRAVE_AVAILABLE = False

# Import MuJoCo wrapper (when available)
try:
    from .mujoco import MuJoCoRobotAdapter
    TSRWrapperFactory.register_wrapper('mujoco', MuJoCoRobotAdapter)
    _MUJOCO_AVAILABLE = True
except ImportError:
    _MUJOCO_AVAILABLE = False

__all__ = [
    'RobotInterface',
    'ObjectInterface', 
    'EnvironmentInterface',
    'TSRWrapperFactory'
]

# Add OpenRAVE exports if available
if _OPENRAVE_AVAILABLE:
    __all__.extend([
        'OpenRAVERobotAdapter',
        'OpenRAVEObjectAdapter',
        'OpenRAVEEnvironmentAdapter',
        'place_object',
        'transport_upright',
        'cylinder_grasp',
        'box_grasp'
    ])

# Add MuJoCo exports if available
if _MUJOCO_AVAILABLE:
    __all__.extend([
        'MuJoCoRobotAdapter'
    ]) 