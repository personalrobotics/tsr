# SPDX-License-Identifier: BSD-2-Clause
# Authors: Siddhartha Srinivasa and contributors to TSR

"""Shared Hypothesis matrix for every placement factory (issue #148).

The placement counterpart of ``tests/tsr/hands/_grasp_matrix.py``: it drives all public
``place_*`` factories through the independent :mod:`_placement_oracle` and provides

* a factory table (which oracle primitive a factory describes, and how to build it
  from the call's arguments);
* Hypothesis strategies over surface extents and object dimensions, spanning the
  regimes where the object fits comfortably, fits exactly, and cannot fit at all;
* **pure** check functions that return failure strings instead of asserting, so the
  property matrix and the mutation gate share one definition of "the suite detects
  this".

Pose sampling comes from the grasp matrix (``bw_samples``): the midpoint, every free
dimension's extrema, all corners and interior points. A placement claim is about the
whole region, so checking the midpoint alone is what let #149 and #150 through.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from hypothesis import strategies as st

from tsr.placement import StablePlacer

from ..hands._grasp_matrix import budget, bw_samples, matrix_settings  # noqa: F401  (re-exported)
from ._placement_oracle import Box, Cylinder, Mesh, Sphere, Torus, certify

# --------------------------------------------------------------------------- #
# Factory table
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Factory:
    name: str
    #: Build the oracle primitive from the factory's keyword arguments.
    oracle: Callable[[Dict[str, Any]], Any]
    #: ``variant`` -> the outward object-frame normal of the face that must face down.
    #: Empty for factories whose variants are not face labels.
    face_normals: Dict[str, Tuple[float, float, float]] = field(default_factory=dict)


_AXES = {
    "-z": (0.0, 0.0, -1.0),
    "+z": (0.0, 0.0, 1.0),
    "-y": (0.0, -1.0, 0.0),
    "+y": (0.0, 1.0, 0.0),
    "-x": (-1.0, 0.0, 0.0),
    "+x": (1.0, 0.0, 0.0),
}

FACTORIES: Dict[str, Factory] = {
    "place_cylinder": Factory(
        "place_cylinder",
        lambda kw: Cylinder(kw["cylinder_radius"], kw["cylinder_height"]),
        {k: _AXES[k] for k in ("-z", "+z")},
    ),
    "place_box": Factory(
        "place_box",
        lambda kw: Box(kw["lx"], kw["ly"], kw["lz"]),
        dict(_AXES),
    ),
    "place_sphere": Factory("place_sphere", lambda kw: Sphere(kw["radius"])),
    "place_torus": Factory(
        "place_torus",
        lambda kw: Torus(kw["major_radius"], kw["minor_radius"]),
        {k: _AXES[k] for k in ("-z", "+z")},
    ),
    "place_mesh": Factory(
        "place_mesh", lambda kw: Mesh(np.asarray(kw["vertices"], float), np.asarray(kw["com"], float))
    ),
}


def public_factories() -> List[str]:
    """Every public ``place_*`` method, so the matrix cannot silently miss one."""
    return sorted(n for n in dir(StablePlacer) if n.startswith("place_") and callable(getattr(StablePlacer, n)))


# --------------------------------------------------------------------------- #
# Cases
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PlacementCase:
    factory: str
    kwargs: Dict[str, Any]
    table: Tuple[float, float]

    @property
    def prim(self):
        return FACTORIES[self.factory].oracle(self.kwargs)

    def placer(self) -> StablePlacer:
        return StablePlacer(table_x=self.table[0], table_y=self.table[1])

    def call(self) -> List[Any]:
        return getattr(self.placer(), self.factory)(**self.kwargs)

    def __str__(self) -> str:  # pragma: no cover - diagnostics only
        return f"{self.factory}({self.kwargs}) on table {self.table}"


def _mesh_vertices(kind: str, scale: float) -> Tuple[np.ndarray, np.ndarray]:
    """A few small vertex clouds with an honestly computed centre of mass.

    ``corner-cube`` has its origin at a corner rather than the centroid, which is the
    case that distinguishes "the origin's resting height" from "half an extent".
    """
    if kind == "cube":
        v = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float) * (scale / 2.0)
        return v, v.mean(axis=0)
    if kind == "corner-cube":
        v = np.array([[x, y, z] for x in (0.0, scale) for y in (0.0, scale) for z in (0.0, scale)])
        return v, v.mean(axis=0)
    if kind == "slab":
        v = np.array(
            [[sx * scale / 2, sy * scale / 3, sz * scale / 10] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
        )
        return v, v.mean(axis=0)
    if kind == "tetra":
        v = np.array([[1.0, 1.0, 1.0], [1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]]) * (scale / 2.0)
        return v, v.mean(axis=0)
    raise AssertionError(kind)


@st.composite
def cases(draw, factories: Optional[Tuple[str, ...]] = None, fit: str = "any") -> PlacementCase:
    """One placement call across scale regimes.

    ``fit`` selects the regime: ``"fits"`` keeps the object comfortably on the surface,
    ``"any"`` also draws objects that cannot fit, which is where the empty-feasible-set
    half of the contract lives.
    """
    factory = draw(st.sampled_from(sorted(factories or FACTORIES)))
    scale = draw(st.sampled_from([1e-3, 1e-2, 0.1, 1.0]))
    table = (draw(st.floats(0.5, 5.0)) * scale, draw(st.floats(0.5, 5.0)) * scale)
    if fit == "fits":
        # Keep the largest possible footprint (the half-diagonal of a cube of side
        # ``scale``, or R + r) inside the surface with room to spare.
        table = (max(table[0], 2.0 * scale), max(table[1], 2.0 * scale))

    def dim(lo: float, hi: float) -> float:
        return draw(st.floats(lo, hi)) * scale

    if factory == "place_cylinder":
        kwargs: Dict[str, Any] = {
            "cylinder_radius": dim(0.05, 0.5),
            "cylinder_height": dim(0.05, 1.0),
        }
    elif factory == "place_box":
        kwargs = {"lx": dim(0.05, 1.0), "ly": dim(0.05, 1.0), "lz": dim(0.05, 1.0)}
    elif factory == "place_sphere":
        kwargs = {"radius": dim(0.05, 0.5)}
    elif factory == "place_torus":
        major = dim(0.1, 0.5)
        kwargs = {"major_radius": major, "minor_radius": major * draw(st.floats(0.05, 0.9))}
    else:
        vertices, com = _mesh_vertices(draw(st.sampled_from(["cube", "corner-cube", "slab", "tetra"])), scale)
        kwargs = {"vertices": vertices, "com": com}
    return PlacementCase(factory, kwargs, table)


# --------------------------------------------------------------------------- #
# Checks (pure: they return failures, they do not assert)
# --------------------------------------------------------------------------- #


def soundness_failures(case: PlacementCase, templates, *, table_pose=None, rng=None) -> List[str]:
    """Every pose the region admits must satisfy all three placement clauses.

    Poses are taken at the ``Bw`` midpoint, every free dimension's extrema, all corners
    and interior points -- the region, not a representative.
    """
    failures: List[str] = []
    prim = case.prim
    T = np.eye(4) if table_pose is None else table_pose
    T_inv = np.linalg.inv(T)
    for t in templates:
        tsr = t.instantiate(T)
        for xi in bw_samples(t, rng=rng):
            # Certify in the surface frame: the oracle's z = 0 is the surface plane.
            pose = T_inv @ tsr.to_transform(xi)
            w = certify(prim, pose, table=case.table)
            if not w.ok:
                failures.append(f"{case} [{t.variant}] xi={np.round(xi, 6).tolist()}: {w.failed}")
    return failures


def face_label_failures(case: PlacementCase, templates) -> List[str]:
    """The face a ``variant`` names must be the face that ends up on the surface.

    Without this a relabelling of the variants is invisible, and "cereal box front-face
    down" is an unbacked promise.
    """
    normals = FACTORIES[case.factory].face_normals
    if not normals:
        return []
    failures: List[str] = []
    down = np.array([0.0, 0.0, -1.0])
    for t in templates:
        n = np.array(normals[t.variant])
        for xi in bw_samples(t):
            pose = t.instantiate(np.eye(4)).to_transform(xi)
            world_n = pose[:3, :3] @ n
            if not np.allclose(world_n, down, atol=1e-9):
                failures.append(f"{case} [{t.variant}]: face normal maps to {np.round(world_n, 6).tolist()}, not -z")
                break
    return failures


def equivariance_failures(case: PlacementCase, templates, T: np.ndarray) -> List[str]:
    """Instantiating on a moved surface must move the poses rigidly, and nothing else."""
    failures: List[str] = []
    for t in templates:
        at_identity = t.instantiate(np.eye(4))
        moved = t.instantiate(T)
        for xi in bw_samples(t):
            expected = T @ at_identity.to_transform(xi)
            got = moved.to_transform(xi)
            if not np.allclose(got, expected, atol=1e-9):
                failures.append(f"{case} [{t.variant}]: instantiate({T.round(3).tolist()}) is not rigid")
                break
    return failures
