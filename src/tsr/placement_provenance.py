# SPDX-License-Identifier: BSD-2-Clause
# Authors: Siddhartha Srinivasa and contributors to TSR

"""Structured, machine-readable provenance for generated placement templates.

The placement counterpart of :mod:`tsr.grasp_provenance`, and the same three layers
(issue #160). Clause 8 of the grasp contract requires a template to declare its
semantics *structurally* and forbids inferring them by parsing ``name``; before this,
placement left a caller no alternative — the resting face's normal, the lever arm behind
the reported tipping angle, and the footprint radius that set the ``Bw`` inset lived
only inside a free-text ``name`` or nowhere at all.

1. :class:`PlacementProvenance` — immutable, extensible, and validating *representation*
   invariants only. Labels are free strings, so a third-party placement generator can
   use its own vocabulary without modifying sstsr.
2. Native conformance (``tsr.placement._conformance``) owns the sstsr vocabulary and the
   relational checks: which primitives exist, that only a sphere may omit a face normal,
   that ``equilibrium`` agrees with ``support_margin``, and that the template's
   ``stability_margin`` really is ``atan2(support_margin, com_height)``.
3. The analytic oracle (``tests/tsr/placement/_placement_oracle.py``) derives resting,
   support and containment from the concrete pose, and must **not** treat these values
   as evidence that the pose is correct.

**Provenance declares intent; the pose determines geometric truth.** A check may read a
record as the subject under test; it may never use one as the expected value.

Field meanings (see the placement contract in ``docs/ARCHITECTURE.md``):

``face_normal``
    The outward normal, in the **object** frame, of the face that ends up on the
    surface. ``None`` only for a sphere, which has no distinguished face.
``support_margin``
    ``d_min`` [m]: the in-plane distance from the projected centre of mass to the
    nearest edge of the support polygon. A cylinder cap gives ``r``, a box face
    ``min(a, b)/2``, a torus lying flat ``R``, a sphere ``0``.
``com_height``
    The lever arm [m]: the perpendicular distance from the centre of mass to the resting
    plane. Together with ``support_margin`` it reconstructs the reported tipping angle,
    ``stability_margin = atan2(support_margin, com_height)``, so the headline number is
    auditable rather than merely asserted. It is **not** the origin height, which is a
    positioning quantity and coincides only when the origin is the centre of mass.
``footprint_radius``
    The largest horizontal distance from the object frame origin to any point of the
    resting object [m] — the radius the free yaw sweeps, and the inset applied to the
    ``Bw`` translation bounds (#150). Unrecoverable from a template before this record.
``equilibrium``
    ``"stable"`` or ``"neutral"``. A sphere reports a zero ``support_margin`` because it
    contacts at a point and *rolls*; a mesh face with a vanishing margin would instead be
    a knife-edge equilibrium about to tip. Both look like ``stability_margin ≈ 0``, and
    nothing distinguished them before.
``face_index`` / ``face_count`` / ``facet_count``
    Mesh only, and all three or none. ``face_index`` is **0-based** (mirroring
    ``depth_index``), while the ``variant`` string is 1-based (``"face-1"``); conformance
    is what keeps the two from drifting. ``face_count`` is how many templates *this call*
    returned, so like ``depth_count`` it is not an identifier across independently
    generated or concatenated collections. ``facet_count`` is how many co-planar hull
    facets were merged into the face (#152) — greater than the two of a triangulated flat
    face means the caller handed over a finely tessellated surface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

from ._provenance_common import JSONScalar, freeze_metadata, require_finite_real, require_int


def _canonical_normal(value: object) -> Optional[Tuple[float, float, float]]:
    """``None``, or a length-3 direction canonicalised to a tuple of plain floats.

    Unit norm is **not** required here: that is sstsr vocabulary, and lives in native
    conformance. What a *representation* can insist on is that the value is a direction
    at all, so the zero vector is rejected.
    """
    if value is None:
        return None
    try:
        components = list(value)  # type: ignore[call-overload]
    except TypeError as e:
        raise ValueError(f"face_normal must be None or a length-3 sequence, got {value!r}") from e
    if len(components) != 3:
        raise ValueError(f"face_normal must have 3 components, got {len(components)}")
    normal = tuple(require_finite_real(c, f"face_normal[{i}]") for i, c in enumerate(components))
    if not any(normal):
        raise ValueError("face_normal must be a direction, got the zero vector")
    return normal  # type: ignore[return-value]


@dataclass(frozen=True)
class PlacementProvenance:
    """The generator's structured claim about a placement template.

    Validates only representation invariants — non-empty string labels, finite
    non-negative lengths, a length-3 non-zero ``face_normal``, exact integer mesh
    counters that are present together, and JSON-scalar ``metadata`` frozen into an
    immutable mapping — so the record round-trips exactly. It deliberately does **not**
    know which primitives sstsr supports, that a normal should be a unit vector, or that
    only a sphere may omit one; that is native conformance
    (``tsr.placement._conformance``), which can evolve with the built-in generators
    without closing this type to external use.
    """

    #: Wire discriminator. Unannotated on purpose: a bare ``KIND: str = "placement"``
    #: would become a dataclass field and change ``__init__`` and ``__eq__``.
    KIND = "placement"

    primitive: str
    support_margin: float
    com_height: float
    footprint_radius: float
    equilibrium: str
    face_normal: Optional[Tuple[float, float, float]] = None
    face_index: Optional[int] = None
    face_count: Optional[int] = None
    facet_count: Optional[int] = None
    metadata: Mapping[str, JSONScalar] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("primitive", "equilibrium"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string, got {value!r}")

        # Lengths are canonicalised to float: a NumPy scalar passes isinstance(x, float)
        # but serializes to a tag safe_load refuses, so it survives a dict round-trip and
        # dies on disk (see require_finite_real).
        for name in ("support_margin", "com_height", "footprint_radius"):
            object.__setattr__(self, name, require_finite_real(getattr(self, name), name, minimum=0.0))

        object.__setattr__(self, "face_normal", _canonical_normal(self.face_normal))

        mesh_fields = ("face_index", "face_count", "facet_count")
        present = [name for name in mesh_fields if getattr(self, name) is not None]
        if present and len(present) != len(mesh_fields):
            missing = [name for name in mesh_fields if getattr(self, name) is None]
            raise ValueError(f"{', '.join(mesh_fields)} are present together or not at all; missing {missing}")
        if present:
            for name in mesh_fields:
                require_int(getattr(self, name), name)
            if self.face_count < 1 or not (0 <= self.face_index < self.face_count):
                raise ValueError(f"face_index {self.face_index} out of range for face_count {self.face_count}")
            if self.facet_count < 1:
                raise ValueError(f"facet_count must be >= 1, got {self.facet_count}")

        object.__setattr__(self, "metadata", freeze_metadata(self.metadata))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict of JSON/YAML scalars (lossless)."""
        d: Dict[str, Any] = {
            "kind": self.KIND,
            "primitive": self.primitive,
            "support_margin": self.support_margin,
            "com_height": self.com_height,
            "footprint_radius": self.footprint_radius,
            "equilibrium": self.equilibrium,
        }
        if self.face_normal is not None:
            d["face_normal"] = list(self.face_normal)
        if self.face_index is not None:
            d["face_index"] = self.face_index
            d["face_count"] = self.face_count
            d["facet_count"] = self.facet_count
        if self.metadata:
            d["metadata"] = dict(self.metadata)
        return d

    @staticmethod
    def from_dict(x: Mapping[str, Any]) -> "PlacementProvenance":
        """Reconstruct from :meth:`to_dict` output, validating (not narrowing).

        Bracket access on the required keys on purpose: a ``.get()`` with a default is
        the one route by which a grasp block could be misread as a placement block.
        """
        kind = x.get("kind", PlacementProvenance.KIND)
        if kind != PlacementProvenance.KIND:
            raise ValueError(f"not a placement provenance block: kind is {kind!r}")
        return PlacementProvenance(
            primitive=x["primitive"],
            support_margin=x["support_margin"],
            com_height=x["com_height"],
            footprint_radius=x["footprint_radius"],
            equilibrium=x["equilibrium"],
            face_normal=x.get("face_normal"),
            face_index=x.get("face_index"),
            face_count=x.get("face_count"),
            facet_count=x.get("facet_count"),
            metadata=dict(x.get("metadata", {})),
        )
