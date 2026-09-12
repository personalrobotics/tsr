# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Task-space constraints.

A *constraint* is the geometry-only, robot-agnostic object the planner queries to
decide whether an end-effector pose is acceptable, and onto which it projects a
pose when it is not. Every constraint exposes the same small duck-typed seam the
``pycbirrt`` planner depends on:

* ``distance(pose) -> (dist, witness)`` — 0 inside the constraint, otherwise the
  task-space distance to it, plus an opaque *witness* describing the closest
  admissible point (the seed for projection).
* ``to_transform(witness) -> Motor`` — map a witness back to a world-frame
  end-effector ``Motor`` (the pose space of ``forward_kinematics``).
* ``sample() -> Motor`` — draw an admissible world-frame pose.

The legacy :class:`~tsr.tsr.TSR` (a 6-DoF box in the CGA-split frame) is one
concrete :class:`Constraint`. :class:`PlaneConstraint` and
:class:`SphereConstraint` are geometric-primitive siblings that keep the
end-effector origin on a CGA :class:`gafro.Plane` / :class:`gafro.Sphere`.
"""

from .base import Constraint
from .plane import PlaneConstraint
from .sphere import SphereConstraint

__all__ = ["Constraint", "PlaneConstraint", "SphereConstraint"]
