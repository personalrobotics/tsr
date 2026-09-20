# SPDX-License-Identifier: BSD-2-Clause
# Authors: Siddhartha Srinivasa and contributors to TSR

"""TSR — Task Space Regions for robotics.

Robot-agnostic Task Space Region (TSR) functionality for pose-constrained
manipulation planning. The package is layered:

* **core** — the pure-math heart: :class:`TSR`, :class:`TSRChain`, and SE(3)
  helpers. Robot-agnostic, NumPy/SciPy only.
* **template** — :class:`TSRTemplate`: a reusable, serialisable, scene-agnostic
  recipe that instantiates to a concrete TSR at a reference pose.
* **factories** — domain knowledge that *produces* templates:
  :class:`ParallelJawGripper` (and ready-made hands) for grasps, and
  :class:`StablePlacer` for placements.
* **services** — :mod:`tsr.io` (persistence), sampling helpers, and the
  optional :mod:`tsr.viz` renderer (install with the ``viz`` extra).

The spine: a factory produces recipes; a recipe instantiates to math::

    from tsr import ParallelJawGripper
    gripper   = ParallelJawGripper(finger_length=0.055, max_aperture=0.140)
    templates = gripper.grasp_cylinder(cylinder_radius=0.04, cylinder_height=0.12)
    tsr       = templates[0].instantiate(object_pose_in_world)
    pose      = tsr.sample()              # a valid grasp pose
    assert tsr.contains(pose)
"""

# --- Layer 0: core math -------------------------------------------------------
from .core import (
    EPSILON,
    TSR,
    TSRChain,
    geodesic_distance,
    geodesic_error,
    rotation_angle,
    wrap_to_interval,
)

# --- Layer 1: the reusable recipe ---------------------------------------------
from .template import TSRTemplate

# --- Layer 2: factories (domain knowledge -> recipes) -------------------------
from .hands import FrankaHand, GripperBase, ParallelJawGripper, Robotiq2F85, Robotiq2F140
from .placement import StablePlacer

# --- Layer 3: services --------------------------------------------------------
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
from .sampling import (
    choose_tsr,
    choose_tsr_index,
    instantiate_templates,
    sample_from_templates,
    sample_from_tsrs,
    weights_from_tsrs,
)

__version__ = "2.0.1"

__all__ = [
    # core
    "TSR",
    "TSRChain",
    "wrap_to_interval",
    "rotation_angle",
    "geodesic_error",
    "geodesic_distance",
    "EPSILON",
    # template
    "TSRTemplate",
    # factories
    "GripperBase",
    "ParallelJawGripper",
    "Robotiq2F85",
    "Robotiq2F140",
    "FrankaHand",
    "StablePlacer",
    # sampling helpers
    "choose_tsr",
    "choose_tsr_index",
    "instantiate_templates",
    "sample_from_tsrs",
    "sample_from_templates",
    "weights_from_tsrs",
    # template I/O
    "save_template",
    "save_template_collection",
    "load_template",
    "load_template_collection",
    "load_package_template",
    "load_package_templates_by_category",
    "list_available_templates",
    "get_package_templates",
    "__version__",
]
