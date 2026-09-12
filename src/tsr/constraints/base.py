# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""The :class:`Constraint` abstract base class.

This formalizes the interface the ``pycbirrt`` planner already relies on
structurally (see ``CBiRRT._satisfies_constraints`` / ``_project_to_constraint``):
a constraint must be able to (a) report how far a pose is from satisfying it and
(b) project a pose onto its manifold. The *witness* returned by ``distance`` is
constraint-specific and is fed straight back into ``to_transform`` to obtain the
projected world-frame pose — for a :class:`~tsr.tsr.TSR` it is the closest ``bw``
6-vector; for a :class:`~tsr.constraints.plane.PlaneConstraint` it is the closest
admissible ``Motor``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from gafro import Motor


class Constraint(ABC):
    """Geometry-only, robot-agnostic task-space constraint.

    A pose is *satisfied* iff :meth:`distance` returns ``0`` (within tolerance).
    Subclasses implement the three abstract methods; everything the planner needs
    is expressed through them, so any subclass is a drop-in path constraint.
    """

    @abstractmethod
    def distance(self, trans, rotation_weight: float = 1.0) -> tuple[float, Any]:
        """Task-space distance from ``trans`` to this constraint.

        @param  trans            a ``Motor`` or 4x4 transform (world-frame EE pose)
        @param  rotation_weight  weight for the rotation component vs translation
        @return ``(dist, witness)`` where ``dist`` is ``0`` when ``trans`` already
                satisfies the constraint, and ``witness`` is the closest admissible
                point in this constraint's own parametrization (the seed consumed
                by :meth:`to_transform`).
        """

    @abstractmethod
    def to_transform(self, witness: Any) -> Motor:
        """Map a witness (as returned by :meth:`distance`) to a world-frame pose.

        The result lives in the same pose space as ``robot.forward_kinematics``,
        so the planner can hand it straight to its IK solver.
        """

    @abstractmethod
    def sample(self) -> Motor:
        """Draw an admissible world-frame end-effector pose as a ``Motor``."""

    def contains(self, trans, tolerance: float = 0.0) -> bool:
        """Whether ``trans`` satisfies this constraint within ``tolerance``."""
        dist, _ = self.distance(trans)
        return dist <= tolerance
