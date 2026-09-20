# SPDX-License-Identifier: BSD-2-Clause
# Authors: Siddhartha Srinivasa and contributors to TSR

"""Structured, machine-readable provenance for generated grasp templates.

Clause 8 of the geometric grasp contract (docs/ARCHITECTURE.md, issue #66)
requires that every template a factory returns declares its grasp mode
*structurally* — tests and oracles read this record, never parse the
human-readable ``name``. The record is a closed, immutable, lossless value
object carried on ``TSRTemplate.provenance`` and round-tripped through the
template's dict/JSON/YAML serialization.

Field frames and meanings (single definition each; see docs/ARCHITECTURE.md):

* Fingers always open along ``±y_EE`` (a library invariant), so the closing line
  is read from the *pose*, not from this record. The record instead classifies
  the grasp in the **object/reference frame** for coverage and oracle dispatch.
* ``depth`` is the **insertion depth from the approached primitive surface**
  along the approach axis, in metres (``>= 0``). Uniform across every primitive
  and mode; consumed by the analytic oracle.
* ``approach`` is the object-relative side/family the **hand occupies**
  (``"+z"``, ``"-x"``, ``"radial"``, ``"tube"``), not the sign of ``z_EE``.
* ``span_axis`` is the object-relative direction along which the two pad contacts
  are separated (:data:`SPAN_AXES`): an object axis for boxes, or a yaw-free
  family (``"tangential"``/``"diameter"``) for radial/spherical grasps.
* ``primitive``, ``mode``, ``depth``, and torus ``params.minor_angle`` are the
  oracle-consumed fields; ``approach``, ``span_axis``, ``variant``, and
  ``depth_index``/``depth_count`` classify coverage and symmetry only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Dict, Mapping, Union

PRIMITIVES = ("box", "cylinder", "sphere", "torus")

# Allowed modes per primitive — validated as (primitive, mode) pairs, never as
# two independent strings. "side" wraps radially; "top"/"bottom" approach a face
# along the object ±z; "face" is a box face; "span" grips a whole torus ring;
# "surface" is a full-SO(3) sphere grasp (approach from any direction).
_MODES_BY_PRIMITIVE = {
    "box": ("top", "bottom", "face"),
    "cylinder": ("side", "top", "bottom"),
    "sphere": ("surface",),
    "torus": ("side", "span"),
}
MODES = tuple(sorted({m for modes in _MODES_BY_PRIMITIVE.values() for m in modes}))

# Object-relative contact-span directions. Object axes for boxes; yaw-free
# families for radial/spherical grasps (the specific diameter/tangent rotates
# with the free yaw, so no single object axis applies).
SPAN_AXES = ("x", "y", "z", "tangential", "diameter")

JSONScalar = Union[str, int, float, bool]


def _require_int(value: object, name: str) -> int:
    """Return a true (non-boolean) int, else raise. No silent narrowing."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be a non-boolean integer, got {value!r}")
    return value


@dataclass(frozen=True)
class GraspProvenance:
    """Machine-readable description of how a grasp template was generated.

    See the module docstring for each field's coordinate frame and meaning. All
    fields are validated on construction; ``params`` is defensively copied into an
    immutable mapping so a valid record can never be mutated or serialize lossily.
    """

    primitive: str
    mode: str
    approach: str
    span_axis: str
    depth_index: int
    depth_count: int
    depth: float
    variant: str = ""
    params: Mapping[str, JSONScalar] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.primitive not in _MODES_BY_PRIMITIVE:
            raise ValueError(f"unknown primitive {self.primitive!r}; expected one of {PRIMITIVES}")
        allowed = _MODES_BY_PRIMITIVE[self.primitive]
        if self.mode not in allowed:
            raise ValueError(f"mode {self.mode!r} is not valid for primitive {self.primitive!r}; allowed {allowed}")
        if not self.approach:
            raise ValueError("approach must be a non-empty label")
        if self.span_axis not in SPAN_AXES:
            raise ValueError(f"unknown span_axis {self.span_axis!r}; expected one of {SPAN_AXES}")

        _require_int(self.depth_index, "depth_index")
        _require_int(self.depth_count, "depth_count")
        if self.depth_count < 1 or not (0 <= self.depth_index < self.depth_count):
            raise ValueError(f"depth_index {self.depth_index} out of range for depth_count {self.depth_count}")

        depth = float(self.depth)
        if not math.isfinite(depth) or depth < 0.0:
            raise ValueError(f"depth must be finite and >= 0, got {self.depth!r}")

        # Validate params are JSON scalars (finite floats), then freeze a copy so
        # the caller's dict cannot later mutate this "frozen" record.
        frozen: Dict[str, JSONScalar] = {}
        for key, val in dict(self.params).items():
            if not isinstance(key, str):
                raise ValueError(f"params keys must be str, got {key!r}")
            if isinstance(val, bool) or isinstance(val, (str, int)):
                frozen[key] = val
            elif isinstance(val, float):
                if not math.isfinite(val):
                    raise ValueError(f"params[{key!r}] must be a finite float, got {val!r}")
                frozen[key] = val
            else:
                raise ValueError(
                    f"params[{key!r}] must be a JSON scalar (str/int/float/bool), got {type(val).__name__}"
                )
        object.__setattr__(self, "params", MappingProxyType(frozen))

    def to_dict(self) -> Dict[str, JSONScalar]:
        """Serialize to a plain dict of JSON/YAML scalars (lossless)."""
        d: Dict[str, Any] = {
            "primitive": self.primitive,
            "mode": self.mode,
            "approach": self.approach,
            "span_axis": self.span_axis,
            "depth_index": self.depth_index,
            "depth_count": self.depth_count,
            "depth": float(self.depth),
        }
        if self.variant:
            d["variant"] = self.variant
        if self.params:
            d["params"] = dict(self.params)
        return d

    @staticmethod
    def from_dict(x: Mapping[str, Any]) -> "GraspProvenance":
        """Reconstruct from :meth:`to_dict` output, validating (not narrowing)."""
        return GraspProvenance(
            primitive=x["primitive"],
            mode=x["mode"],
            approach=x["approach"],
            span_axis=x["span_axis"],
            depth_index=x["depth_index"],
            depth_count=x["depth_count"],
            depth=x["depth"],
            variant=x.get("variant", ""),
            params=dict(x.get("params", {})),
        )
