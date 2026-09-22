# SPDX-License-Identifier: BSD-2-Clause
# Authors: Siddhartha Srinivasa and contributors to TSR

"""Structured, machine-readable provenance for generated grasp templates.

Clause 8 of the geometric grasp contract (docs/ARCHITECTURE.md, issue #66)
requires that every template a factory returns declares its grasp mode
*structurally* — tests and oracles read this record, never parse the
human-readable ``name``.

**Provenance declares intent; the generated pose determines geometric truth**
(issue #84). Three layers are kept separate:

1. :class:`GraspProvenance` — a small, immutable, **extensible** value object
   recording the generator's claim. It validates only *representation*
   invariants (below); it does not know which ``(primitive, mode)`` pairs exist
   or how axes/faces/angles relate. Labels are free strings, so third-party grasp
   generators can use their own vocabulary without modifying sstsr.
2. Native conformance (``tsr.hands._conformance.validate_builtin_provenance``)
   owns the sstsr-specific vocabulary and relational checks; the built-in factory
   tests run it over every emitted template.
3. The analytic oracle (#67) derives contacts/reach/clearance from the concrete
   pose and primitive geometry, and must *not* trust provenance values as evidence
   the pose is correct.

Field frames and meanings (see docs/ARCHITECTURE.md): ``depth`` is the insertion
depth from the approached surface [m]; ``approach`` is the object-relative side
the hand occupies; ``finger_orientation`` is the object-relative direction the
pads are separated along (fingers always close along ``±y_EE``, a library
invariant); ``depth_count`` is the number of **distinct** depth levels emitted for
a *depth family* (at most the requested ``k``; fewer when the feasible band
collapses), and ``depth_index`` in ``0..depth_count-1`` orders them shallow to deep.
A depth family is the set of templates **within one factory result** whose non-depth
provenance fields are identical: ``primitive``, ``mode``, ``approach``,
``finger_orientation``, ``symmetry``, and ``metadata`` are held fixed while only
``depth``, ``depth_index``, and ``depth_count`` vary, so a family holds exactly one
template per depth level. The counters are not a global identifier across
independently generated or concatenated collections;
``symmetry`` distinguishes otherwise-equivalent templates; ``metadata`` is
descriptive only and must never be used as geometric evidence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Dict, Mapping, Union

JSONScalar = Union[str, int, float, bool]


def _require_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be a non-boolean integer, got {value!r}")
    return value


@dataclass(frozen=True)
class GraspProvenance:
    """The generator's structured claim about a grasp template.

    Validates only representation invariants — nonempty string labels, a finite
    nonnegative ``depth``, exact integer depth counters, and JSON-scalar
    ``metadata`` frozen into an immutable mapping — so the record round-trips
    exactly. It intentionally does **not** validate that a ``(primitive, mode)``
    is one sstsr supports or that fields are mutually consistent; that is native
    conformance (``tsr.hands._conformance``), which can evolve with the built-in
    generators without closing this type to external use.
    """

    primitive: str
    mode: str
    approach: str
    finger_orientation: str
    depth: float
    depth_index: int
    depth_count: int
    symmetry: str = ""
    metadata: Mapping[str, JSONScalar] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("primitive", "mode", "approach", "finger_orientation"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string, got {value!r}")
        if not isinstance(self.symmetry, str):
            raise ValueError(f"symmetry must be a string, got {self.symmetry!r}")

        _require_int(self.depth_index, "depth_index")
        _require_int(self.depth_count, "depth_count")
        if self.depth_count < 1 or not (0 <= self.depth_index < self.depth_count):
            raise ValueError(f"depth_index {self.depth_index} out of range for depth_count {self.depth_count}")

        if isinstance(self.depth, bool) or not isinstance(self.depth, (int, float)) or not math.isfinite(self.depth):
            raise ValueError(f"depth must be a finite, non-boolean real number, got {self.depth!r}")
        if float(self.depth) < 0.0:
            raise ValueError(f"depth must be >= 0, got {self.depth!r}")
        object.__setattr__(self, "depth", float(self.depth))  # canonicalize -> lossless round-trip

        frozen: Dict[str, JSONScalar] = {}
        for key, val in dict(self.metadata).items():
            if not isinstance(key, str):
                raise ValueError(f"metadata keys must be str, got {key!r}")
            if isinstance(val, (str, bool, int)):
                frozen[key] = val
            elif isinstance(val, float):
                if not math.isfinite(val):
                    raise ValueError(f"metadata[{key!r}] must be a finite float, got {val!r}")
                frozen[key] = val
            else:
                raise ValueError(f"metadata[{key!r}] must be a JSON scalar, got {type(val).__name__}")
        object.__setattr__(self, "metadata", MappingProxyType(frozen))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict of JSON/YAML scalars (lossless)."""
        d: Dict[str, Any] = {
            "primitive": self.primitive,
            "mode": self.mode,
            "approach": self.approach,
            "finger_orientation": self.finger_orientation,
            "depth": self.depth,
            "depth_index": self.depth_index,
            "depth_count": self.depth_count,
        }
        if self.symmetry:
            d["symmetry"] = self.symmetry
        if self.metadata:
            d["metadata"] = dict(self.metadata)
        return d

    @staticmethod
    def from_dict(x: Mapping[str, Any]) -> "GraspProvenance":
        """Reconstruct from :meth:`to_dict` output, validating (not narrowing)."""
        return GraspProvenance(
            primitive=x["primitive"],
            mode=x["mode"],
            approach=x["approach"],
            finger_orientation=x["finger_orientation"],
            depth=x["depth"],
            depth_index=x["depth_index"],
            depth_count=x["depth_count"],
            symmetry=x.get("symmetry", ""),
            metadata=dict(x.get("metadata", {})),
        )
