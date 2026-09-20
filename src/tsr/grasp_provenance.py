# SPDX-License-Identifier: BSD-2-Clause
# Authors: Siddhartha Srinivasa and contributors to TSR

"""Structured, machine-readable provenance for generated grasp templates.

Clause 8 of the geometric grasp contract (docs/ARCHITECTURE.md, issue #66)
requires that every template a factory returns declares its grasp mode
*structurally* — tests and oracles read this record, never parse the
human-readable ``name``. The record is a closed, immutable, lossless value
object carried on ``TSRTemplate.provenance`` and round-tripped through the
template's dict/JSON/YAML serialization.

Closed: a per-``(primitive, mode)`` schema (:data:`_SCHEMA`) fixes the compatible
``approach``, ``span_axis``, and ``variant`` values and the required mode-specific
``params`` — semantically incompatible combinations are rejected on construction.
Immutable: ``params`` is validated and frozen into a read-only mapping. Lossless:
scalar fields are canonicalized on construction (e.g. ``depth`` is stored as a
float), so ``from_dict(p.to_dict()) == p`` for every accepted record.

Field frames and meanings (single definition each; see docs/ARCHITECTURE.md):

* Fingers always open along ``±y_EE`` (a library invariant), so the closing line
  is read from the *pose*, not from this record. The record instead classifies
  the grasp in the **object/reference frame** for coverage and oracle dispatch.
* ``depth`` is the **insertion depth from the approached primitive surface**
  along the approach axis, in metres (``>= 0``); consumed by the analytic oracle.
* ``approach`` is the object-relative side/family the **hand occupies**, not the
  sign of ``z_EE``.
* ``span_axis`` is the object-relative direction the two pad contacts are
  separated along: an object axis for boxes, or a yaw-free family
  (``"tangential"``/``"diameter"``) for radial/spherical grasps.
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

# Object-relative contact-span directions. Object axes for boxes; yaw-free
# families for radial/spherical grasps.
SPAN_AXES = ("x", "y", "z", "tangential", "diameter")

JSONScalar = Union[str, int, float, bool]


def _is_real(v: object) -> bool:
    """Finite, non-Boolean real number."""
    return not isinstance(v, bool) and isinstance(v, (int, float)) and math.isfinite(float(v))


def _is_str(v: object) -> bool:
    return isinstance(v, str)


# Per-(primitive, mode) schema: compatible approach / span_axis / variant values
# and required params (name -> predicate). This is the closed vocabulary — a
# record whose fields don't satisfy its schema entry is rejected (#80).
_SCHEMA: Dict = {
    ("cylinder", "side"): dict(
        approaches={"radial"}, span_axes={"tangential"}, variants={"roll0", "rollpi"}, params={}
    ),
    ("cylinder", "top"): dict(approaches={"+z"}, span_axes={"diameter"}, variants={""}, params={}),
    ("cylinder", "bottom"): dict(approaches={"-z"}, span_axes={"diameter"}, variants={""}, params={}),
    ("box", "top"): dict(
        approaches={"+z"}, span_axes={"x", "y"}, variants={""}, params={"slide_axis": _is_str, "span": _is_real}
    ),
    ("box", "bottom"): dict(
        approaches={"-z"}, span_axes={"x", "y"}, variants={""}, params={"slide_axis": _is_str, "span": _is_real}
    ),
    ("box", "face"): dict(
        approaches={"+x", "-x", "+y", "-y"},
        span_axes={"x", "y", "z"},
        variants={""},
        params={"slide_axis": _is_str, "span": _is_real},
    ),
    ("sphere", "surface"): dict(approaches={"radial"}, span_axes={"diameter"}, variants={""}, params={}),
    ("torus", "side"): dict(
        approaches={"tube"},
        span_axes={"tangential"},
        variants={"flip0", "flippi"},
        params={
            "minor_index": lambda v: not isinstance(v, bool) and isinstance(v, int),
            "minor_count": lambda v: not isinstance(v, bool) and isinstance(v, int),
            "minor_angle": _is_real,
        },
    ),
    ("torus", "span"): dict(approaches={"+z", "-z"}, span_axes={"diameter"}, variants={""}, params={}),
}

# Derived vocabularies (source of truth is _SCHEMA).
MODES = tuple(sorted({mode for _, mode in _SCHEMA}))
_MODES_BY_PRIMITIVE = {p: tuple(m for (pp, m) in _SCHEMA if pp == p) for p in PRIMITIVES}


def _require_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be a non-boolean integer, got {value!r}")
    return value


@dataclass(frozen=True)
class GraspProvenance:
    """Machine-readable description of how a grasp template was generated.

    See the module docstring for each field's coordinate frame and meaning, and
    for the closed/immutable/lossless guarantees enforced on construction.
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
        key = (self.primitive, self.mode)
        if key not in _SCHEMA:
            allowed_modes = _MODES_BY_PRIMITIVE[self.primitive]
            raise ValueError(
                f"mode {self.mode!r} is not valid for primitive {self.primitive!r}; allowed {allowed_modes}"
            )
        schema = _SCHEMA[key]

        for name, value, allowed in (
            ("approach", self.approach, schema["approaches"]),
            ("span_axis", self.span_axis, schema["span_axes"]),
            ("variant", self.variant, schema["variants"]),
        ):
            if not isinstance(value, str):
                raise ValueError(f"{name} must be a string, got {value!r}")
            if value not in allowed:
                raise ValueError(f"{name}={value!r} is not valid for {key}; allowed {sorted(allowed)}")

        _require_int(self.depth_index, "depth_index")
        _require_int(self.depth_count, "depth_count")
        if self.depth_count < 1 or not (0 <= self.depth_index < self.depth_count):
            raise ValueError(f"depth_index {self.depth_index} out of range for depth_count {self.depth_count}")

        if not _is_real(self.depth) or float(self.depth) < 0.0:
            raise ValueError(f"depth must be a finite, non-boolean real >= 0, got {self.depth!r}")
        # Canonicalize to float so a record round-trips losslessly (#80).
        object.__setattr__(self, "depth", float(self.depth))

        # Validate params: JSON scalars with finite floats, plus the mode's
        # required keys and their types; then freeze a defensive copy.
        frozen: Dict[str, JSONScalar] = {}
        for pkey, pval in dict(self.params).items():
            if not isinstance(pkey, str):
                raise ValueError(f"params keys must be str, got {pkey!r}")
            if isinstance(pval, str) or isinstance(pval, bool):
                frozen[pkey] = pval
            elif isinstance(pval, int):
                frozen[pkey] = pval
            elif isinstance(pval, float):
                if not math.isfinite(pval):
                    raise ValueError(f"params[{pkey!r}] must be a finite float, got {pval!r}")
                frozen[pkey] = pval
            else:
                raise ValueError(
                    f"params[{pkey!r}] must be a JSON scalar (str/int/float/bool), got {type(pval).__name__}"
                )
        for req_name, predicate in schema["params"].items():
            if req_name not in frozen:
                raise ValueError(f"{key} requires params[{req_name!r}]")
            if not predicate(frozen[req_name]):
                raise ValueError(f"params[{req_name!r}]={frozen[req_name]!r} fails the type requirement for {key}")
        object.__setattr__(self, "params", MappingProxyType(frozen))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict of JSON/YAML scalars (lossless)."""
        d: Dict[str, Any] = {
            "primitive": self.primitive,
            "mode": self.mode,
            "approach": self.approach,
            "span_axis": self.span_axis,
            "depth_index": self.depth_index,
            "depth_count": self.depth_count,
            "depth": self.depth,
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
