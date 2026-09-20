# SPDX-License-Identifier: BSD-2-Clause
# Authors: Siddhartha Srinivasa and contributors to TSR

"""Structured, machine-readable provenance for generated grasp templates.

Clause 8 of the geometric grasp contract (docs/ARCHITECTURE.md, issue #66)
requires that every template a factory returns declares its grasp mode
*structurally* — tests and oracles must read this record, never parse the
human-readable ``name``. The record is a small frozen dataclass carried on
``TSRTemplate.provenance`` and round-tripped through the template's
dict/JSON/YAML serialization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

# Allowed vocabularies, kept small and closed so the oracle and property tests
# can switch on them exhaustively.
PRIMITIVES = ("box", "cylinder", "sphere", "torus")
# Approach families a factory may emit. "side" wraps radially; "top"/"bottom"
# approach a face along ±z; "face" is a box face; "span" grips a whole ring;
# "equatorial" is a sphere approach in a plane through the center.
MODES = ("side", "top", "bottom", "face", "span", "equatorial")


@dataclass(frozen=True)
class GraspProvenance:
    """Machine-readable description of how a grasp template was generated.

    Attributes:
        primitive: The solid family — one of :data:`PRIMITIVES`.
        mode: The approach family — one of :data:`MODES`.
        approach: Label of the approach side/axis in the reference frame, e.g.
            ``"+z"``, ``"-x"``, or ``"outer"`` (torus). Descriptive, not parsed
            for geometry.
        opening_axis: The finger-opening axis this template grips along, as a
            reference/EE-frame label (``"x"``, ``"y"``, ``"z"``).
        depth_index: 0-based index of this discrete approach depth.
        depth_count: Number of discrete depths requested (``k``); ``depth_index``
            is in ``range(depth_count)``.
        depth: The standoff / finger-reach value baked into this template [m].
        variant: Symmetry variant that produces a geometrically distinct pose at
            the same (mode, depth), e.g. a 180° roll flip. Empty when unused.
        params: Primitive-specific extras that don't fit the common fields, e.g.
            ``{"minor_index": 0, "minor_angle": 0.0}`` for a torus side grasp or
            ``{"slide_axis": "y"}`` for a box face. Values must be JSON/YAML
            scalars (str/int/float/bool) so the record round-trips.
    """

    primitive: str
    mode: str
    approach: str
    opening_axis: str
    depth_index: int
    depth_count: int
    depth: float
    variant: str = ""
    params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.primitive not in PRIMITIVES:
            raise ValueError(f"unknown primitive {self.primitive!r}; expected one of {PRIMITIVES}")
        if self.mode not in MODES:
            raise ValueError(f"unknown mode {self.mode!r}; expected one of {MODES}")
        if not (0 <= self.depth_index < self.depth_count):
            raise ValueError(f"depth_index {self.depth_index} out of range for depth_count {self.depth_count}")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict of JSON/YAML scalars."""
        d = {
            "primitive": self.primitive,
            "mode": self.mode,
            "approach": self.approach,
            "opening_axis": self.opening_axis,
            "depth_index": int(self.depth_index),
            "depth_count": int(self.depth_count),
            "depth": float(self.depth),
        }
        if self.variant:
            d["variant"] = self.variant
        if self.params:
            d["params"] = dict(self.params)
        return d

    @staticmethod
    def from_dict(x: Dict[str, Any]) -> "GraspProvenance":
        """Reconstruct from :meth:`to_dict` output."""
        return GraspProvenance(
            primitive=x["primitive"],
            mode=x["mode"],
            approach=x["approach"],
            opening_axis=x["opening_axis"],
            depth_index=int(x["depth_index"]),
            depth_count=int(x["depth_count"]),
            depth=float(x["depth"]),
            variant=x.get("variant", ""),
            params=dict(x.get("params", {})),
        )
