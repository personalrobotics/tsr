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

Robot-agnostic Task Space Region (TSR) functionality for pose-constrained
manipulation planning.

Usage:
    from tsr import TSR, TSRTemplate, TSRChain
    from tsr import TaskCategory, TaskType, EntityClass
    from tsr import sample_from_tsrs
"""

# Core math
from .tsr import TSR
from .tsr_chain import TSRChain
from .utils import EPSILON, wrap_to_interval, geodesic_distance, geodesic_error

# Templates
from .template import TSRTemplate

# Sampling
from .sampling import (
    weights_from_tsrs,
    choose_tsr_index,
    choose_tsr,
    sample_from_tsrs,
    instantiate_templates,
    sample_from_templates,
)

# Placement
from .placement import StablePlacer

# Template I/O
from .io import (
    save_template,
    load_template,
    save_template_collection,
    load_template_collection,
    get_package_templates,
    list_available_templates,
    load_package_template,
    load_package_templates_by_category,
)

