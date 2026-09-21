# SPDX-License-Identifier: BSD-2-Clause
# Authors: Siddhartha Srinivasa and contributors to TSR

"""Independent analytic oracle for primitive parallel-jaw grasps (issue #67).

This is a **test-side** oracle. Given only

* the primitive type and dimensions,
* the idealized gripper dimensions (finger length ``L``, max aperture ``A``),
* the requested preshape ``a`` and clearance ``c``,
* a concrete end-effector pose ``E`` (in the primitive's reference frame), and
* the grasp ``(primitive, mode)`` label, used **only** to select the SDF model,

it certifies the geometric-soundness clauses of ``docs/ARCHITECTURE.md``.

**Provenance independence (#84).** Every quantity the oracle certifies -- palm
clearance, usable finger depth, jaw contacts, contact normals, and object span --
is derived from the pose and an independent signed-distance representation of the
primitive. The oracle never reads the generator's private standoff/depth formulas
and never uses ``provenance.depth``, a recorded minor angle, or ``metadata`` as an
input to the calculation that certifies those same values. ``primitive``/``mode``
select which analytic model to apply; everything else is checked against the pose.

Frame convention (library invariant): ``z_EE`` = approach, ``y_EE`` = finger
opening (jaws always close along ``±y_EE``), ``x_EE = y_EE × z_EE``. The palm is the
translation of ``E``; the fingertips are ``palm + L·z_EE``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

# Scale-aware tolerances (docs/ARCHITECTURE.md "Tolerances and conventions").
ANGLE_ATOL = 1e-6  # rad, contact-normal opposition


def length_atol(scale: float) -> float:
    """Scale-aware length tolerance ``1e-9 + 1e-6 · scale``."""
    return 1e-9 + 1e-6 * float(scale)


# --------------------------------------------------------------------------- #
# Independent primitive signed-distance representations.
#
# These are standard closed-form SDFs (negative inside, zero on the surface,
# positive outside), written from the object-frame conventions in
# tsr.hands.base -- NOT from any generator standoff/contact formula.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Sphere:
    """Sphere of ``radius`` centered at the object-frame origin."""

    radius: float

    @property
    def scale(self) -> float:
        return self.radius

    def sdf(self, p: np.ndarray) -> float:
        return float(np.linalg.norm(p) - self.radius)


@dataclass(frozen=True)
class Cylinder:
    """Finite cylinder: axis ``+z``, radius ``radius``, ``z ∈ [0, height]``."""

    radius: float
    height: float

    @property
    def scale(self) -> float:
        return max(self.radius, self.height)

    def sdf(self, p: np.ndarray) -> float:
        radial = float(np.hypot(p[0], p[1])) - self.radius
        axial = abs(p[2] - self.height / 2.0) - self.height / 2.0
        outside = np.array([max(radial, 0.0), max(axial, 0.0)])
        return float(min(max(radial, axial), 0.0) + np.linalg.norm(outside))


@dataclass(frozen=True)
class Box:
    """Axis-aligned box: ``x ∈ [-dx/2, dx/2]``, ``y ∈ [-dy/2, dy/2]``, ``z ∈ [0, dz]``."""

    dx: float
    dy: float
    dz: float

    @property
    def scale(self) -> float:
        return max(self.dx, self.dy, self.dz)

    def sdf(self, p: np.ndarray) -> float:
        center = np.array([0.0, 0.0, self.dz / 2.0])
        half = np.array([self.dx / 2.0, self.dy / 2.0, self.dz / 2.0])
        q = np.abs(p - center) - half
        return float(np.linalg.norm(np.maximum(q, 0.0)) + min(max(q[0], q[1], q[2]), 0.0))


@dataclass(frozen=True)
class Torus:
    """Torus: axis ``+z``, tube-center radius ``major``, tube radius ``minor``."""

    major: float
    minor: float

    @property
    def scale(self) -> float:
        return self.major + self.minor

    def sdf(self, p: np.ndarray) -> float:
        q = np.array([float(np.hypot(p[0], p[1])) - self.major, p[2]])
        return float(np.linalg.norm(q) - self.minor)


Primitive = object  # any of the dataclasses above (duck-typed .sdf/.scale)


def outward_normal(prim, point: np.ndarray) -> np.ndarray:
    """Unit outward surface normal at ``point``, from the SDF gradient.

    Central differences on the independent SDF -- no primitive-specific normal
    formula. Valid on face/lateral interiors (undefined only exactly on an edge).
    """
    eps = 1e-7 * (prim.scale + 1.0)
    grad = np.zeros(3)
    for i in range(3):
        step = np.zeros(3)
        step[i] = eps
        grad[i] = prim.sdf(point + step) - prim.sdf(point - step)
    norm = np.linalg.norm(grad)
    if norm == 0.0:
        return grad
    return grad / norm


def _extent_along(prim, base: np.ndarray, close: np.ndarray, s_max: float) -> Optional[Tuple[float, float]]:
    """Outermost object extent ``(s_lo, s_hi)`` along the closing line through ``base``.

    Models parallel jaws closing from *outside* along ``±close``: the contacts are
    the object's extreme offsets on the line ``base + s·close`` (``sdf ≤ 0``),
    regardless of whether ``base`` is inside the solid (it is not, for an enveloping
    grasp whose approach axis passes through a torus hole). Returns ``None`` if the
    line misses the object, or if the object reaches the search window (no clean
    outer contact).
    """
    # A coarse scan only needs to BRACKET the object's boundaries; the bisection
    # below refines each to full precision, so resolution here is a speed knob.
    steps = 768
    ss = np.linspace(-s_max, s_max, steps + 1)
    inside = np.array([prim.sdf(base + s * close) < 0.0 for s in ss])
    if not inside.any():
        return None
    idx = np.where(inside)[0]
    i_lo, i_hi = int(idx[0]), int(idx[-1])
    if i_lo == 0 or i_hi == steps:
        return None

    def _bisect(s_out: float, s_in: float) -> float:
        for _ in range(100):
            mid = 0.5 * (s_out + s_in)
            if prim.sdf(base + mid * close) < 0.0:
                s_in = mid
            else:
                s_out = mid
        return 0.5 * (s_out + s_in)

    s_lo = _bisect(ss[i_lo - 1], ss[i_lo])
    s_hi = _bisect(ss[i_hi + 1], ss[i_hi])
    return s_lo, s_hi


def _best_closing(
    prim, palm: np.ndarray, approach: np.ndarray, close: np.ndarray, reach: float, s_max: float
) -> Optional[Tuple[float, Tuple[float, float], Tuple[float, float]]]:
    """Closing plane depth ``t* ∈ [0, reach]`` maximizing the object's span.

    The jaws contact the widest reachable part of the object first, so the sound
    closing plane is where the ``±close`` extent is largest within finger reach.
    Returns ``(t*, (s_lo, s_hi), (t_first, t_last))`` where the last pair is the
    depth band over which the object is present on the closing line, or ``None`` if
    no closing plane within reach meets the object.
    """
    ts = np.linspace(0.0, reach, 129)
    present: List[float] = []
    spans: List[float] = []
    for t in ts:
        ext = _extent_along(prim, palm + t * approach, close, s_max)
        if ext is None:
            continue
        present.append(float(t))
        spans.append(ext[1] - ext[0])
    if not present:
        return None

    # Contact where the object is widest within reach. The max-span set is often a
    # flat plateau (a cylinder/box side is constant over its height); take the
    # plateau's midpoint so contacts land on a clean surface interior, not on an
    # extreme rim edge where the normal is an ambiguous corner.
    best_span = max(spans)
    plateau_tol = max(length_atol(prim.scale), 1e-6 * best_span)
    band = [t for t, s in zip(present, spans) if s >= best_span - plateau_tol]
    t_star = 0.5 * (min(band) + max(band))
    ext = _extent_along(prim, palm + t_star * approach, close, s_max)
    if ext is None:  # non-contiguous band fallback: nearest present plane
        t_star = min(present, key=lambda t: abs(t - t_star))
        ext = _extent_along(prim, palm + t_star * approach, close, s_max)
    return t_star, ext, (min(present), max(present))


@dataclass
class GraspWitness:
    """Structured geometric witness for one concrete pose (see #67).

    ``ok`` is True iff every soundness clause holds. ``failed`` lists
    ``(clause, message)`` for each violated clause, so a failure reports geometry,
    not just a Boolean.
    """

    ok: bool
    palm_clearance: float
    contact_depth: float
    usable_depth_interval: Tuple[float, float]
    span: float
    contact_neg: Optional[np.ndarray]
    contact_pos: Optional[np.ndarray]
    normal_neg: Optional[np.ndarray]
    normal_pos: Optional[np.ndarray]
    failed: List[Tuple[int, str]] = field(default_factory=list)


def certify(
    prim,
    pose: np.ndarray,
    *,
    finger_length: float,
    max_aperture: float,
    preshape: float,
    clearance: float,
    mode: str,
) -> GraspWitness:
    """Certify the soundness clauses for ``prim`` at end-effector ``pose``.

    ``pose`` is the 4×4 end-effector transform in the primitive's reference frame
    (``E(ξ) = T_ref_tsr · xyzrpy_to_trans(ξ) · Tw_e``). ``mode`` is accepted for
    parity with provenance/model selection; the SDF is chosen by ``prim``'s type.
    """
    atol = length_atol(prim.scale)
    failed: List[Tuple[int, str]] = []

    # Clause 1: finiteness / well-formed pose.
    finite = np.all(np.isfinite(pose)) and pose.shape == (4, 4) and np.allclose(pose[3], [0, 0, 0, 1], atol=1e-9)
    R = pose[:3, :3]
    if finite and not np.allclose(R.T @ R, np.eye(3), atol=1e-6):
        finite = False
    if not finite:
        failed.append((1, "pose is not a finite, well-formed SE(3) transform"))
        return GraspWitness(False, np.nan, np.nan, (np.nan, np.nan), np.nan, None, None, None, None, failed)

    palm = pose[:3, 3]
    approach = pose[:3, 2]
    close = pose[:3, 1]

    # Clause 3: palm outside S with the promised clearance.
    palm_clearance = prim.sdf(palm)
    if palm_clearance < clearance - atol:
        failed.append((3, f"palm clearance {palm_clearance:.6g} < requested c {clearance:.6g}"))

    # Closing geometry: parallel jaws close from OUTSIDE along ±y_EE and contact the
    # object's widest reachable extent (this handles both penetrating grasps, whose
    # approach axis enters the solid, and enveloping grasps, whose approach axis
    # passes through a hole). t* is the closing-plane depth along the approach axis.
    s_max = 4.0 * prim.scale + max(preshape, 0.0) + 1.0
    best = _best_closing(prim, palm, approach, close, finger_length, s_max)
    span = np.nan
    contact_depth = np.nan
    usable = (np.nan, np.nan)
    contact_pos = contact_neg = normal_pos = normal_neg = None
    if best is None:
        failed.append((5, "closing along ±y_EE meets no object within finger reach"))
    else:
        t_star, (s_lo, s_hi), usable = best
        base = palm + t_star * approach
        contact_pos = base + s_hi * close
        contact_neg = base + s_lo * close
        normal_pos = outward_normal(prim, contact_pos)
        normal_neg = outward_normal(prim, contact_neg)
        span = float(s_hi - s_lo)
        contact_depth = float(t_star)

        # Clause 4/5: the contact plane is reachable and strictly ahead of the palm.
        if not (atol < t_star <= finger_length + atol):
            failed.append((5, f"closing plane depth {t_star:.6g} not within usable band (0, {finger_length:.6g}]"))

        # Clause 6: inward contact normals oppose the closing directions.
        ang_pos = float(np.arccos(np.clip(normal_pos @ close, -1.0, 1.0)))
        ang_neg = float(np.arccos(np.clip(normal_neg @ (-close), -1.0, 1.0)))
        if ang_pos > ANGLE_ATOL or ang_neg > ANGLE_ATOL:
            failed.append((6, f"contact normals do not oppose closing (angles {ang_pos:.2e}, {ang_neg:.2e} rad)"))

    # Clause 2: aperture validity and that the object fits strictly between the pads.
    if not (0.0 < preshape <= max_aperture + atol):
        failed.append((2, f"preshape {preshape:.6g} not in (0, A={max_aperture:.6g}]"))
    if np.isfinite(span) and span >= preshape - atol:
        failed.append((2, f"object span {span:.6g} does not fit within preshape {preshape:.6g}"))

    return GraspWitness(
        ok=not failed,
        palm_clearance=float(palm_clearance),
        contact_depth=contact_depth,
        usable_depth_interval=usable,
        span=span,
        contact_neg=contact_neg,
        contact_pos=contact_pos,
        normal_neg=normal_neg,
        normal_pos=normal_pos,
        failed=failed,
    )


def pose_from(palm: np.ndarray, approach: np.ndarray, close: np.ndarray) -> np.ndarray:
    """Build an SE(3) EE pose from a palm point and approach/close axes.

    ``approach`` → ``z_EE`` and ``close`` → ``y_EE`` (re-orthonormalized against
    the approach), with ``x_EE = y_EE × z_EE``. Test helper for hand-derived poses.
    """
    z = np.asarray(approach, dtype=float)
    z = z / np.linalg.norm(z)
    y = np.asarray(close, dtype=float)
    y = y - (y @ z) * z
    y = y / np.linalg.norm(y)
    x = np.cross(y, z)
    T = np.eye(4)
    T[:3, 0] = x
    T[:3, 1] = y
    T[:3, 2] = z
    T[:3, 3] = np.asarray(palm, dtype=float)
    return T
