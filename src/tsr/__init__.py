# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""
TSR Library - Task Space Regions for Robotics

Robot-agnostic Task Space Region (TSR) functionality for pose-constrained
manipulation planning.

Usage:
    from tsr import TSR, TSRTemplate, TSRChain
    from tsr import TaskCategory, TaskType, EntityClass
    from tsr import sample_from_tsrs
"""

# Core math
# Template I/O
from .io import (
    get_package_templates,
    list_available_templates,
    load_package_template,
    load_package_templates_by_category,
    load_template,
    load_template_collection,
    save_template,
    save_template_collection,
)

# Placement
from .placement import StablePlacer

# Sampling
from .sampling import (
    choose_tsr,
    choose_tsr_index,
    instantiate_templates,
    sample_from_templates,
    sample_from_tsrs,
    weights_from_tsrs,
)

# Templates
from .template import TSRTemplate
from .tsr import TSR
from .tsr_chain import TSRChain
from .utils import EPSILON, geodesic_distance, geodesic_error, wrap_to_interval
