# tsr/src/tsr/bimanual.py
# SPDX-License-Identifier: MIT
"""Bimanual Task Space Regions.

A :class:`BimanualTSR` constrains a dual-arm system in the natural cooperative
coordinates: the *absolute* pose (the object / midpoint frame) and the
*relative* pose (one hand relative to the other, i.e. the grasp). Each is an
ordinary 6-DOF :class:`~tsr.tsr.TSR`; either may be omitted. Presence of a
component means it is *active* — the IK solver and planner key off that.

This does not subclass :class:`~tsr.constraints.base.Constraint`: that ABC is
defined over a single world-frame end-effector pose, whereas a bimanual TSR
speaks in a *pair* of poses (:class:`BimanualPose`).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from gafro import Motor

from tsr.tsr import TSR


@dataclass(frozen=True)
class BimanualPose:
    """A dual-arm pose pair. ``None`` marks an inactive/absent component."""
    absolute: "Motor | None" = None
    relative: "Motor | None" = None


@dataclass(frozen=True)
class BimanualWitness:
    """Per-component closest-``bw`` seeds from :meth:`BimanualTSR.distance`."""
    absolute: "np.ndarray | None" = None
    relative: "np.ndarray | None" = None


def _interval_sum(Bw: np.ndarray) -> float:
    """Summed Bw width with rotation rows (0:3) clamped to 2π; matches TSR.volume."""
    widths = np.asarray(Bw[:, 1] - Bw[:, 0], dtype=float)
    widths[0:3] = np.minimum(widths[0:3], 2.0 * np.pi)
    widths = np.maximum(widths, 0.0)
    return float(np.sum(widths))


class BimanualTSR:
    """Absolute + relative pose constraint for a bimanual system."""

    def __init__(self, absolute: TSR | None = None, relative: TSR | None = None):
        if absolute is None and relative is None:
            raise ValueError("BimanualTSR requires at least one of absolute / relative")
        self.absolute = absolute
        self.relative = relative

    def __repr__(self) -> str:
        active = [name for name, component in (("absolute", self.absolute),
                                               ("relative", self.relative))
                  if component is not None]
        return f"BimanualTSR(active=[{','.join(active)}])"

    def sample(self) -> BimanualPose:
        return BimanualPose(
            absolute=self.absolute.sample() if self.absolute is not None else None,
            relative=self.relative.sample() if self.relative is not None else None,
        )

    def distance(self, pose: BimanualPose, rotation_weight: float = 1.0):
        total = 0.0
        abs_bw = None
        rel_bw = None
        if self.absolute is not None:
            d, abs_bw = self.absolute.distance(pose.absolute, rotation_weight)
            total += d
        if self.relative is not None:
            d, rel_bw = self.relative.distance(pose.relative, rotation_weight)
            total += d
        return total, BimanualWitness(absolute=abs_bw, relative=rel_bw)

    def to_transform(self, witness: BimanualWitness) -> BimanualPose:
        return BimanualPose(
            absolute=(self.absolute.to_transform(witness.absolute)
                      if self.absolute is not None else None),
            relative=(self.relative.to_transform(witness.relative)
                      if self.relative is not None else None),
        )

    def contains(self, pose: BimanualPose, tolerance: float = 0.0) -> bool:
        dist, _ = self.distance(pose)
        return dist <= tolerance

    @property
    def volume(self) -> float:
        v = 0.0
        if self.absolute is not None:
            v += _interval_sum(self.absolute.Bw)
        if self.relative is not None:
            v += _interval_sum(self.relative.Bw)
        return v
