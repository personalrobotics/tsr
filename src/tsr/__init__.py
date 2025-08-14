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

This library provides robot-agnostic Task Space Region (TSR) functionality.
It includes a core geometric TSR, a neutral TSRTemplate for scene-free storage,
and a relational library for registering/querying TSRs between entities.

Core (robot-agnostic):
    TSR: Core Task Space Region (geometry + sampling)
    TSRTemplate: Neutral, scene-agnostic TSR template (REFERENCE→TSR, TSR→SUBJECT, Bw)
    TSRLibraryRelational: Registry keyed by (subject_entity, reference_entity, task)
    TaskCategory, TaskType, EntityClass: Controlled vocabulary
    Sampling helpers: weights_from_tsrs, choose_tsr_index, choose_tsr, sample_from_tsrs

Usage:
    # Core usage (robot-agnostic)
    from tsr.core.tsr import TSR
    from tsr.core.tsr_template import TSRTemplate
    from tsr.tsr_library_rel import TSRLibraryRelational
    from tsr.schema import TaskCategory, TaskType, EntityClass
    from tsr.sampling import sample_from_tsrs
"""

# Import core classes
from .core import TSR, TSRChain, wrap_to_interval, EPSILON

try:
    from .schema import TaskCategory, TaskType, EntityClass
    from .core.tsr_template import TSRTemplate
    from .tsr_library_rel import TSRLibraryRelational
    from .sampling import (
        weights_from_tsrs,
        choose_tsr_index,
        choose_tsr,
        sample_from_tsrs,
        instantiate_templates,
        sample_from_templates,
    )
    from .template_io import (
        TemplateIO,
        save_template,
        load_template,
        save_template_collection,
        load_template_collection,
    )
    from .generators import (
        generate_cylinder_grasp_template,
        generate_box_grasp_template,
        generate_place_template,
        generate_transport_template,
        generate_mug_grasp_template,
        generate_box_place_template,
    )
    _RELATIONAL_AVAILABLE = True
except Exception:
    _RELATIONAL_AVAILABLE = False

# Export all symbols
__all__ = [
    # Core classes
    'TSR',
    'TSRChain', 
    'wrap_to_interval',
    'EPSILON',

    # Relational / schema / sampling (optional)
    'TSRTemplate',
    'TSRLibraryRelational',
    'TaskCategory',
    'TaskType',
    'EntityClass',
    'weights_from_tsrs',
    'choose_tsr_index',
    'choose_tsr',
    'sample_from_tsrs',
    'instantiate_templates',
    'sample_from_templates',
    
    # Template I/O utilities
    'TemplateIO',
    'save_template',
    'load_template',
    'save_template_collection',
    'load_template_collection',
    
    # Template generators
    'generate_cylinder_grasp_template',
    'generate_box_grasp_template',
    'generate_place_template',
    'generate_transport_template',
    'generate_mug_grasp_template',
    'generate_box_place_template',
]

if not _RELATIONAL_AVAILABLE:
    for _name in (
        'TSRTemplate',
        'TSRLibraryRelational',
        'TaskCategory',
        'TaskType',
        'EntityClass',
        'weights_from_tsrs',
        'choose_tsr_index',
        'choose_tsr',
        'sample_from_tsrs',
        'instantiate_templates',
        'sample_from_templates',
        'TemplateIO',
        'save_template',
        'load_template',
        'save_template_collection',
        'load_template_collection',
        'generate_cylinder_grasp_template',
        'generate_box_grasp_template',
        'generate_place_template',
        'generate_transport_template',
        'generate_mug_grasp_template',
        'generate_box_place_template',
    ):
        if _name in __all__:
            __all__.remove(_name)
