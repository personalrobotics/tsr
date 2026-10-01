# SPDX-License-Identifier: BSD-2-Clause
# Authors: Siddhartha Srinivasa and contributors to TSR

"""Representation helpers shared by the provenance value objects.

:class:`~tsr.grasp_provenance.GraspProvenance` and
:class:`~tsr.placement_provenance.PlacementProvenance` describe different things and
share almost no fields, but they must agree *exactly* on what a well-formed record looks
like: which scalars are acceptable, and what ``metadata`` may hold. Two copies of that
logic drifting apart would mean one record type accepting a value the other rejects, so
a template's serialization would depend on which kind of record it carries. The subtle
part is :func:`freeze_metadata`; these live here so there is one of each.

This module is deliberately field-agnostic. It validates *representation* only, which is
layer 1 of the provenance architecture (see ``GraspProvenance``'s module docstring); the
sstsr-specific vocabulary and the relational checks live in the ``_conformance`` modules.
"""

from __future__ import annotations

import math
from types import MappingProxyType
from typing import Dict, Mapping, Union

JSONScalar = Union[str, int, float, bool]


class ProvenanceKindError(Exception):
    """A serialized provenance block declares a ``kind`` this version cannot read.

    Deliberately **not** a subclass of ``ValueError``, ``KeyError`` or ``TypeError``:
    :func:`tsr.io.load_templates_from_directory` catches those and skips the offending
    file with a warning, which is the right policy for a truncated or malformed file but
    the wrong one for a record written by a newer version — that would turn a
    version-skewed directory into a silently empty list. Being outside that tuple makes
    such a file fail loudly, naming itself.
    """


def require_int(value: object, name: str) -> int:
    """An exact, non-boolean integer. ``True`` is not 1 here."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be a non-boolean integer, got {value!r}")
    return value


def require_finite_real(value: object, name: str, *, minimum: float | None = None) -> float:
    """A finite, non-boolean real, canonicalised to ``float`` for a lossless round-trip.

    The ``float()`` is not cosmetic: a NumPy scalar passes ``isinstance(x, float)``, and
    ``yaml.dump`` writes it as a ``!!python/object/apply`` tag that ``safe_load`` then
    refuses — so an un-canonicalised value survives a dict round-trip and dies on disk.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite, non-boolean real number, got {value!r}")
    if minimum is not None and float(value) < minimum:
        raise ValueError(f"{name} must be >= {minimum:g}, got {value!r}")
    return float(value)


def freeze_metadata(metadata: Mapping[str, JSONScalar]) -> Mapping[str, JSONScalar]:
    """Copy ``metadata`` into an immutable mapping of JSON scalars.

    Descriptive only — it must never be read as geometric evidence. The copy is
    defensive: a caller mutating the dict it passed must not change a frozen record.
    """
    frozen: Dict[str, JSONScalar] = {}
    for key, val in dict(metadata).items():
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
    return MappingProxyType(frozen)
