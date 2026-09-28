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
from ._placement_oracle import Box, Cylinder, Mesh, Sphere, Torus, certify, tipping_angle

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
    #: How many templates this call must return. Set for the deterministic corpus and
    #: left ``None`` for generated cases. Without it a mutant that *empties* a result
    #: passes every per-pose check vacuously -- which is exactly what the #152 facet
    #: fragmentation does.
    expected: Optional[int] = None
    label: str = ""

    @property
    def prim(self):
        return FACTORIES[self.factory].oracle(self.kwargs)

    def placer(self) -> StablePlacer:
        return StablePlacer(table_x=self.table[0], table_y=self.table[1])

    def call(self) -> List[Any]:
        return getattr(self.placer(), self.factory)(**self.kwargs)

    def __str__(self) -> str:  # pragma: no cover - diagnostics only
        return self.label or f"{self.factory}({self.kwargs}) on table {self.table}"


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


def margin_failures(case: PlacementCase, templates) -> List[str]:
    """Clause 4: a reported ``stability_margin`` is the pose's physical tipping angle.

    Compared on ``tan`` of the angle, which is the ratio the quantity is built from, at
    ``rtol = 2e-3``. The oracle samples a circular contact as a 72-gon, whose apothem is
    ``cos(pi/72) = 0.99905`` of the true radius, so a curved primitive's tipping angle
    carries a known conservative bias of 9.5e-4 -- an order of magnitude below the
    tolerance, and four orders below the 40% error this check exists to catch (#151).
    """
    failures: List[str] = []
    prim = case.prim
    for t in templates:
        if t.stability_margin is None:
            failures.append(f"{case} [{t.variant}]: no stability_margin (#156)")
            continue
        for xi in bw_samples(t):
            pose = t.instantiate(np.eye(4)).to_transform(xi)
            expected = tipping_angle(prim, pose)
            if not np.isclose(np.tan(t.stability_margin), np.tan(expected), rtol=2e-3, atol=1e-9):
                failures.append(
                    f"{case} [{t.variant}]: margin {np.degrees(t.stability_margin):.4f}° "
                    f"but the posed object tips at {np.degrees(expected):.4f}°"
                )
                break
    return failures


#: Factories that promise an order. ``place_mesh`` documents most-stable-first; the
#: primitives return their faces in a fixed labelled order instead, which is the more
#: useful contract when the caller cares which face is down.
ORDERED_FACTORIES = ("place_mesh",)


def ordering_failures(case: PlacementCase, templates) -> List[str]:
    """``place_mesh`` promises most-stable-first, judged by the reported angles."""
    if case.factory not in ORDERED_FACTORIES:
        return []
    reported = [t.stability_margin for t in templates]
    if any(m is None for m in reported):
        return []
    if reported != sorted(reported, reverse=True):
        return [f"{case}: margins {np.degrees(reported).round(3).tolist()} are not descending"]
    return []


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


# --------------------------------------------------------------------------- #
# The deterministic corpus, and the one definition of "the suite detects this"
# --------------------------------------------------------------------------- #


def octahedron(scale: float = 0.10) -> np.ndarray:
    """Every face tilted with respect to the object frame, which is what makes a
    foreshortened in-plane projection visible (#151)."""
    return np.array([[s * scale if i == j else 0.0 for j in range(3)] for i in range(3) for s in (-1, 1)])


def tessellated_cylinder(radius: float, height: float, n: int) -> np.ndarray:
    """A curved surface cut into ``n`` flat sides, each a real resting face at
    ``180/n`` degrees. Merging them would invent a face the caller never supplied."""
    a = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    rim = np.stack([radius * np.cos(a), radius * np.sin(a)], axis=-1)
    return np.vstack([np.c_[rim, np.full(n, z)] for z in (-height / 2, height / 2)])


def float32_box(lx: float, ly: float, lz: float, seed: int = 0) -> np.ndarray:
    """A rotated box whose vertices round-tripped through float32 -- the ordinary
    result of loading a mesh, since STL, OBJ, glTF and MuJoCo all store float32."""
    from scipy.spatial.transform import Rotation

    Q = Rotation.random(random_state=seed).as_matrix()
    box = np.array([[sx * lx, sy * ly, sz * lz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) / 2.0
    return (box.astype(np.float32) @ Q.T.astype(np.float32)).astype(np.float64)


#: Its centroid projects outside the ``z = 0`` facet, so that facet is not a rest.
_OBTUSE_TETRA = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.2, 0.15, 0.0], [0.9, 0.1, 1.0]])


def corpus() -> List[PlacementCase]:
    """A fixed case per factory plus the shapes that make a specific defect visible.

    Deterministic on purpose: the gate has to attribute a detection to its mutation, so
    it cannot depend on what Hypothesis happened to draw.
    """
    cube = _mesh_vertices("cube", 0.10)
    corner = _mesh_vertices("corner-cube", 0.10)
    tetra = _mesh_vertices("tetra", 0.10)
    f32 = float32_box(0.20, 0.10, 0.30)
    return [
        PlacementCase(
            "place_cylinder", {"cylinder_radius": 0.04, "cylinder_height": 0.12}, (0.30, 0.20), 2, "cylinder"
        ),
        # A disc far larger than the surface: a valid request with no feasible pose.
        PlacementCase("place_cylinder", {"cylinder_radius": 2.0, "cylinder_height": 0.10}, (0.30, 0.20), 0, "4 m disc"),
        PlacementCase("place_box", {"lx": 0.20, "ly": 0.10, "lz": 0.30}, (0.60, 0.60), 6, "box"),
        PlacementCase("place_box", {"lx": 0.01, "ly": 0.01, "lz": 0.30}, (0.60, 0.60), 6, "needle"),
        PlacementCase("place_sphere", {"radius": 0.05}, (0.30, 0.20), 1, "sphere"),
        PlacementCase("place_torus", {"major_radius": 0.05, "minor_radius": 0.012}, (0.30, 0.20), 2, "torus"),
        PlacementCase("place_mesh", {"vertices": cube[0], "com": cube[1]}, (0.60, 0.60), 6, "mesh cube"),
        PlacementCase("place_mesh", {"vertices": corner[0], "com": corner[1]}, (0.60, 0.60), 6, "corner-origin cube"),
        PlacementCase("place_mesh", {"vertices": tetra[0], "com": tetra[1]}, (0.60, 0.60), 4, "tetrahedron"),
        PlacementCase("place_mesh", {"vertices": octahedron(), "com": np.zeros(3)}, (0.60, 0.60), 8, "octahedron"),
        PlacementCase("place_mesh", {"vertices": f32, "com": f32.mean(axis=0)}, (1.0, 1.0), 6, "float32 box"),
        PlacementCase(
            "place_mesh",
            {"vertices": tessellated_cylinder(0.04, 0.12, 24), "com": np.zeros(3)},
            (0.60, 0.60),
            26,  # two caps and twenty-four sides: the prism the caller supplied
            "24-gon cylinder",
        ),
        # An obtuse tetrahedron whose centroid projects outside its bottom facet, at a
        # scale where an absolute containment tolerance stops discriminating (#153).
        PlacementCase(
            "place_mesh",
            {"vertices": _OBTUSE_TETRA * 1e-5, "com": (_OBTUSE_TETRA * 1e-5).mean(axis=0)},
            (2e-5, 2e-5),
            3,  # four facets, and the one the centroid overhangs is not a rest
            "obtuse tetrahedron at 1e-5 m",
        ),
        # The centre of mass exactly above a support edge: a critical equilibrium, which
        # leaves only the face it is squarely above (#153).
        PlacementCase(
            "place_mesh",
            {"vertices": cube[0], "com": np.array([0.0, 0.05, 0.0])},
            (0.60, 0.60),
            1,
            "cube with its COM on a support edge",
        ),
    ]


def check_failures(case: PlacementCase, templates) -> List[str]:
    """Every check the placement suite makes about one call, as failure strings.

    The property matrix asserts this is empty; the mutation gate asserts a corrupted
    generator makes it non-empty. Sharing the definition is what makes "the suite
    detects this" mean the same thing in both places.
    """
    failures: List[str] = []
    if case.expected is not None and len(templates) != case.expected:
        failures.append(f"{case}: {len(templates)} templates, expected {case.expected}")
    rng = np.random.default_rng(0)
    failures += soundness_failures(case, templates, rng=rng)
    failures += margin_failures(case, templates)
    failures += face_label_failures(case, templates)
    failures += ordering_failures(case, templates)
    return failures


def corpus_failures() -> List[str]:
    """Run every check over the whole corpus."""
    return [f for case in corpus() for f in check_failures(case, case.call())]
