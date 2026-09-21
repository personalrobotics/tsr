# SPDX-License-Identifier: BSD-2-Clause
# Authors: Siddhartha Srinivasa and contributors to TSR

"""Independent analytic oracle for primitive parallel-jaw grasps (issue #67).

This is a **test-side** oracle. Given only

* the primitive type and dimensions,
* the idealized gripper dimensions (finger length ``L``, max aperture ``A``),
* the requested preshape ``a`` and clearance ``c``,
* a concrete end-effector pose ``E`` (in the primitive's reference frame), and
* the structural grasp selectors ``mode`` (and optional ``approach`` /
  ``finger_orientation``), used **only** to select and check the analytic model,

it certifies the geometric-soundness clauses of ``docs/ARCHITECTURE.md``.

**Exact, not sampled (#93).** Contacts, spans, normals, realized insertion depth,
and torus minor angle are derived in **closed form** per ``(primitive, mode)`` from
the concrete pose; there are no approach/closing rasters, so results do not depend
on any grid resolution and each certification is O(1) arithmetic. The independent
signed-distance representation is used only as a *verifier* -- it confirms the
analytic contacts lie on the surface and the analytic normals match the SDF
gradient -- never as the search itself.

**Provenance independence (#84).** ``mode``/``approach``/``finger_orientation`` are
structural claim labels that select which analytic model to apply and are *checked*
against the realized pose geometry (a mislabeled mode fails). The oracle never
consumes recorded numeric provenance (``depth``, minor angle, span, metadata) as
evidence; every certified quantity is derived from the pose and primitive geometry.

Frame convention (library invariant): ``z_EE`` = approach, ``y_EE`` = finger opening
(jaws always close along ``±y_EE``), ``x_EE = y_EE × z_EE``. The palm is the
translation of ``E``; the fingertips are ``palm + L·z_EE``.

Clause codes on :class:`GraspWitness`:

* ``1`` model/pose validity (finite, proper SE(3); see also :func:`certify` which
  raises ``ValueError`` for a malformed *model*);
* ``2`` aperture validity and the object fits strictly between the pads;
* ``3`` palm outside the solid with the requested clearance;
* ``4`` the clearance-aware usable band (finger depth, cylinder ends, box edges);
* ``5`` reach: two opposing contacts exist within finger reach;
* ``6`` contact normals oppose the closing directions;
* ``8`` the declared ``mode``/``approach``/``finger_orientation`` match the pose.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

ANGLE_ATOL = 1e-6  # rad, contact-normal opposition (docs/ARCHITECTURE.md)

_EX = np.array([1.0, 0.0, 0.0])
_EY = np.array([0.0, 1.0, 0.0])
_EZ = np.array([0.0, 0.0, 1.0])

SUPPORTED_MODES = {
    "sphere": ("surface",),
    "cylinder": ("side", "top", "bottom"),
    "box": ("top", "bottom", "face"),
    "torus": ("span", "side"),
}


def length_atol(scale: float) -> float:
    """Scale-aware length tolerance ``1e-9 + 1e-6 · scale``."""
    return 1e-9 + 1e-6 * float(scale)


# --------------------------------------------------------------------------- #
# Independent primitive signed-distance representations (used only as a VERIFIER
# for the analytic contacts below -- never as the search). Standard closed-form
# SDFs from the object-frame conventions in tsr.hands.base.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Sphere:
    radius: float

    @property
    def scale(self) -> float:
        return self.radius

    def sdf(self, p: np.ndarray) -> float:
        return float(np.linalg.norm(p) - self.radius)


@dataclass(frozen=True)
class Cylinder:
    radius: float
    height: float  # axis +z, z in [0, height]

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
    dx: float
    dy: float
    dz: float  # x,y centered; z in [0, dz]

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
    major: float  # tube-center ring radius R
    minor: float  # tube radius r

    @property
    def scale(self) -> float:
        return self.major + self.minor

    def sdf(self, p: np.ndarray) -> float:
        q = np.array([float(np.hypot(p[0], p[1])) - self.major, p[2]])
        return float(np.linalg.norm(q) - self.minor)


def _primitive_name(prim) -> str:
    return type(prim).__name__.lower()


def sdf_normal(prim, point: np.ndarray) -> np.ndarray:
    """Independent outward unit normal from the SDF gradient (central difference).

    The step is scaled to the primitive only (no additive one-metre constant), so
    the verifier is scale-equivariant (#93).
    """
    eps = 1e-6 * prim.scale
    grad = np.empty(3)
    for i in range(3):
        step = np.zeros(3)
        step[i] = eps
        grad[i] = prim.sdf(point + step) - prim.sdf(point - step)
    n = np.linalg.norm(grad)
    return grad / n if n > 0 else grad


# --------------------------------------------------------------------------- #
# Model / pose validation (clause 1, issue #95).
# --------------------------------------------------------------------------- #


def _validate_model(prim, finger_length, max_aperture, preshape, clearance, mode) -> None:
    """Validate the oracle *model*; raise ``ValueError`` if malformed (#95).

    A malformed model means the test itself is wrong, so this raises rather than
    returning a clause-1 failure (which is reserved for a malformed candidate pose).
    """
    name = _primitive_name(prim)
    if name == "sphere":
        dims = {"radius": prim.radius}
    elif name == "cylinder":
        dims = {"radius": prim.radius, "height": prim.height}
    elif name == "box":
        dims = {"dx": prim.dx, "dy": prim.dy, "dz": prim.dz}
    elif name == "torus":
        dims = {"major": prim.major, "minor": prim.minor}
    else:
        raise ValueError(f"unsupported primitive type {type(prim).__name__!r}")
    for label, value in dims.items():
        if not (np.isfinite(value) and value > 0.0):
            raise ValueError(f"{name}.{label} must be finite and > 0, got {value!r}")
    if name == "torus" and not (prim.major > prim.minor):
        raise ValueError(f"torus requires major > minor, got major={prim.major!r}, minor={prim.minor!r}")

    for label, value in (("finger_length", finger_length), ("max_aperture", max_aperture), ("preshape", preshape)):
        if not (np.isfinite(value) and value > 0.0):
            raise ValueError(f"{label} must be finite and > 0, got {value!r}")
    if not (np.isfinite(clearance) and clearance >= 0.0):
        raise ValueError(f"clearance must be finite and >= 0, got {clearance!r}")

    if not isinstance(mode, str) or not mode:
        raise ValueError(f"mode must be a non-empty string, got {mode!r}")
    if mode not in SUPPORTED_MODES[name]:
        raise ValueError(f"mode {mode!r} is not supported for a {name} (expected one of {SUPPORTED_MODES[name]})")


def _validate_pose(pose) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """Return ``(pose_float, None)`` for a proper SE(3) pose, else ``(None, reason)``."""
    try:
        p = np.asarray(pose, dtype=float)
    except (ValueError, TypeError):
        return None, "pose is not convertible to a float array"
    if p.shape != (4, 4):
        return None, f"pose must be shape (4, 4), got {p.shape}"
    if not np.all(np.isfinite(p)):
        return None, "pose has non-finite entries"
    if not np.allclose(p[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
        return None, f"pose homogeneous row is {p[3]!r}, expected [0, 0, 0, 1]"
    R = p[:3, :3]
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-6):
        return None, "pose rotation is not orthonormal"
    det = float(np.linalg.det(R))
    if not np.isclose(det, 1.0, atol=1e-6):
        return None, f"pose rotation has det(R)={det:.6g}, not a proper rotation (+1)"
    return p, None


# --------------------------------------------------------------------------- #
# Closed-form line/surface intersections (exact contact geometry).
# --------------------------------------------------------------------------- #


def _line_quadric_roots(o2: np.ndarray, d2: np.ndarray, radius: float) -> Optional[Tuple[float, float]]:
    """Roots of ``|o2 + s·d2|² = radius²`` in 2-D (unit ``d2``), or ``None`` if it misses."""
    b = float(o2 @ d2)
    c = float(o2 @ o2 - radius * radius)
    disc = b * b - c
    if disc < 0.0:
        return None
    root = float(np.sqrt(disc))
    return -b - root, -b + root


@dataclass
class _Contacts:
    base: np.ndarray  # closing-plane point on the approach axis
    contact_lo: np.ndarray
    contact_hi: np.ndarray
    normal_lo: np.ndarray  # analytic outward normals
    normal_hi: np.ndarray
    span: float
    realized_depth: float
    boundary_clearance: float  # min margin to a free edge along the object extent
    minor_angle: Optional[float]
    failed: List[Tuple[int, str]]  # mode/reach failures from the handler (clauses 4/5/6/8)


def _axis_component(vec: np.ndarray, axis: np.ndarray) -> float:
    return float(vec @ axis)


# --------------------------------------------------------------------------- #
# Per-mode analytic handlers. Each derives contacts/normals/depth/boundary from
# the concrete pose and returns a _Contacts (with any mode-consistency, reach, or
# boundary-clearance failures already attached).
# --------------------------------------------------------------------------- #


def _handle_sphere_surface(prim: Sphere, p, a, y, x, L, c, atol) -> _Contacts:
    r = prim.radius
    failed: List[Tuple[int, str]] = []
    # Approach must point toward the centre (origin).
    t_center = float(-(p @ a))
    if t_center <= 0.0:
        failed.append((8, "sphere approach does not point toward the centre"))
    # The clean diameter closing plane is at the centre's projection, capped by reach.
    t_base = min(max(t_center, 0.0), L)
    base = p + t_base * a
    roots = _line_quadric_roots(base, y, r)
    if roots is None:
        failed.append((5, "closing line misses the sphere"))
        return _Contacts(base, base, base, y, -y, np.nan, np.nan, np.inf, None, failed)
    s_lo, s_hi = roots
    contact_lo = base + s_lo * y
    contact_hi = base + s_hi * y
    span = s_hi - s_lo
    # Near-surface entry along the approach ray, for realized insertion depth.
    entry = _line_quadric_roots(p, a, r)
    realized_depth = np.nan if entry is None else L - max(entry[0], 0.0)
    if entry is not None and entry[0] > L + atol:
        failed.append((5, "sphere surface is beyond finger reach"))
    return _Contacts(
        base, contact_lo, contact_hi, contact_lo / r, contact_hi / r, span, realized_depth, np.inf, None, failed
    )


def _handle_cylinder_side(prim: Cylinder, p, a, y, x, L, c, atol) -> _Contacts:
    r, h = prim.radius, prim.height
    failed: List[Tuple[int, str]] = []
    if abs(_axis_component(a, _EZ)) > 1e-6:
        failed.append((8, "cylinder side approach is not radial (perpendicular to the axis)"))
    if abs(_axis_component(y, _EZ)) > 1e-6:
        failed.append((8, "cylinder side closing axis is not perpendicular to the cylinder axis"))
    # Closest approach-ray point to the cylinder (z) axis.
    a_xy = a[:2]
    denom = float(a_xy @ a_xy)
    if denom < 1e-12:
        failed.append((8, "cylinder side approach is parallel to the axis"))
        return _Contacts(p, p, p, y, -y, np.nan, np.nan, np.inf, None, failed)
    t_base = float(-(p[:2] @ a_xy) / denom)
    if t_base > L + atol:
        failed.append((5, "cylinder axis crossing is beyond finger reach"))
    t_base = min(max(t_base, 0.0), L)
    base = p + t_base * a
    roots = _line_quadric_roots(base[:2], y[:2], r)
    if roots is None:
        failed.append((5, "closing line misses the cylinder"))
        return _Contacts(base, base, base, y, -y, np.nan, np.nan, np.inf, None, failed)
    s_lo, s_hi = roots
    contact_lo = base + s_lo * y
    contact_hi = base + s_hi * y
    span = s_hi - s_lo
    z0 = base[2]
    boundary = min(z0 - 0.0, h - z0)  # distance to the nearer end
    if boundary < c - atol:
        failed.append((4, f"side grasp height {z0:.6g} within {c:.6g} of a cylinder end (margin {boundary:.6g})"))
    entry = _line_quadric_roots(p[:2], a[:2], r)
    realized_depth = np.nan if entry is None else L - max(entry[0], 0.0)
    n_lo = np.array([contact_lo[0], contact_lo[1], 0.0]) / r
    n_hi = np.array([contact_hi[0], contact_hi[1], 0.0]) / r
    return _Contacts(base, contact_lo, contact_hi, n_lo, n_hi, span, realized_depth, boundary, None, failed)


def _handle_cylinder_cap(prim: Cylinder, p, a, y, x, L, c, atol, *, top: bool) -> _Contacts:
    r, h = prim.radius, prim.height
    failed: List[Tuple[int, str]] = []
    want = -1.0 if top else 1.0  # top approaches -z, bottom approaches +z
    if not np.isclose(_axis_component(a, _EZ), want, atol=1e-6):
        failed.append((8, f"cylinder {'top' if top else 'bottom'} approach is not along {'-z' if top else '+z'}"))
    if abs(_axis_component(y, _EZ)) > 1e-6:
        failed.append((8, "cylinder cap closing axis is not perpendicular to the cylinder axis"))
    face_z = h if top else 0.0
    # Insertion below the approached face; the closing plane sits at the fingertip.
    standoff = (p[2] - face_z) if top else (face_z - p[2])
    realized_depth = L - standoff
    z0 = min(max(p[2] + want * L, 0.0), h)  # fingertip height, capped to the solid
    base = np.array([p[0], p[1], z0])
    roots = _line_quadric_roots(base[:2], y[:2], r)
    if roots is None:
        failed.append((5, "closing line misses the cylinder cross-section"))
        return _Contacts(base, base, base, y, -y, np.nan, realized_depth, np.inf, None, failed)
    if realized_depth < atol:
        failed.append((5, "fingertips do not reach below the approached cap"))
    if realized_depth > h + atol:
        failed.append((4, "insertion would exit the far end of the cylinder"))
    s_lo, s_hi = roots
    contact_lo = base + s_lo * y
    contact_hi = base + s_hi * y
    span = s_hi - s_lo
    n_lo = np.array([contact_lo[0], contact_lo[1], 0.0]) / r
    n_hi = np.array([contact_hi[0], contact_hi[1], 0.0]) / r
    boundary = min(realized_depth, h - realized_depth)
    return _Contacts(base, contact_lo, contact_hi, n_lo, n_hi, span, realized_depth, boundary, None, failed)


def _box_axis_bounds(prim: Box, axis_index: int) -> Tuple[float, float]:
    if axis_index == 0:
        return -prim.dx / 2.0, prim.dx / 2.0
    if axis_index == 1:
        return -prim.dy / 2.0, prim.dy / 2.0
    return 0.0, prim.dz


def _match_axis(vec: np.ndarray) -> Optional[Tuple[int, float]]:
    """If ``vec`` is (near) ``±`` an object axis, return ``(index, sign)``, else ``None``."""
    for i, axis in enumerate((_EX, _EY, _EZ)):
        d = float(vec @ axis)
        if abs(abs(d) - 1.0) <= 1e-6:
            return i, float(np.sign(d))
    return None


def _handle_box(prim: Box, p, a, y, x, L, c, atol, *, mode: str) -> _Contacts:
    failed: List[Tuple[int, str]] = []
    a_match = _match_axis(a)
    y_match = _match_axis(y)
    if a_match is None or y_match is None:
        failed.append((8, "box grasp approach/closing axes are not aligned with object axes"))
        return _Contacts(p, p, p, y, -y, np.nan, np.nan, np.inf, None, failed)
    a_ax, a_sign = a_match
    y_ax, y_sign = y_match
    if a_ax == y_ax:
        failed.append((8, "box approach and closing axes coincide"))
        return _Contacts(p, p, p, y, -y, np.nan, np.nan, np.inf, None, failed)
    # Mode must match the approached face.
    if mode == "top" and not (a_ax == 2 and a_sign < 0):
        failed.append((8, "box top approach is not -z"))
    if mode == "bottom" and not (a_ax == 2 and a_sign > 0):
        failed.append((8, "box bottom approach is not +z"))
    if mode == "face" and a_ax == 2:
        failed.append((8, "box face approach must be along ±x or ±y, not the z faces"))

    slide_ax = ({0, 1, 2} - {a_ax, y_ax}).pop()
    lo_s, hi_s = _box_axis_bounds(prim, y_ax)  # span-axis (closing) bounds
    lo_a, hi_a = _box_axis_bounds(prim, a_ax)  # approach-axis bounds
    lo_g, hi_g = _box_axis_bounds(prim, slide_ax)  # slide-axis bounds
    span = hi_s - lo_s

    # Approached face plane along the approach axis, and realized insertion depth.
    face_val = hi_a if a_sign < 0 else lo_a  # -z approach hits the +z (upper) face, etc.
    standoff = (p[a_ax] - face_val) * (-a_sign)
    realized_depth = L - standoff
    if realized_depth < atol:
        failed.append((5, "fingertips do not reach past the approached box face"))
    if realized_depth > (hi_a - lo_a) + atol:
        failed.append((4, "insertion would exit the far box face"))

    # The closing plane sits at the fingertip depth; the slide coordinate is the
    # palm's. Fingertip = palm + L·approach, so along the approach axis it moves by
    # ``a_sign · L`` (approach column's component on that object axis).
    depth_coord = min(max(p[a_ax] + a_sign * L, lo_a), hi_a)
    base = np.array(p, dtype=float)
    base[a_ax] = depth_coord
    base[slide_ax] = p[slide_ax]
    base[y_ax] = (lo_s + hi_s) / 2.0  # contacts are symmetric about the span-axis centre

    contact_hi = base.copy()
    contact_hi[y_ax] = hi_s
    contact_lo = base.copy()
    contact_lo[y_ax] = lo_s
    n_hi = np.zeros(3)
    n_hi[y_ax] = 1.0
    n_lo = -n_hi
    # Order lo/hi along +y_EE so normals oppose ±y_EE consistently.
    if y_sign < 0:
        contact_lo, contact_hi = contact_hi, contact_lo
        n_lo, n_hi = n_hi, n_lo

    # Boundary clearance: the pads must stay on the face, clear of the slide edges,
    # and the closing faces must be hit within the approach extent.
    slide_margin = min(p[slide_ax] - lo_g, hi_g - p[slide_ax])
    if slide_margin < c - atol:
        failed.append((4, f"grasp within {c:.6g} of a lateral box edge (slide margin {slide_margin:.6g})"))
    if not (lo_a - atol <= depth_coord <= hi_a + atol):
        failed.append((5, "closing plane falls outside the box along the approach axis"))
    return _Contacts(base, contact_lo, contact_hi, n_lo, n_hi, span, realized_depth, slide_margin, None, failed)


def _handle_torus_span(prim: Torus, p, a, y, x, L, c, atol) -> _Contacts:
    R, r = prim.major, prim.minor
    failed: List[Tuple[int, str]] = []
    if abs(abs(_axis_component(a, _EZ)) - 1.0) > 1e-6:
        failed.append((8, "torus span approach is not along ±z"))
    if abs(_axis_component(y, _EZ)) > 1e-6:
        failed.append((8, "torus span closing axis is not horizontal"))
    # Reach the equator plane (z = 0), where the outer diameter is widest.
    if abs(a[2]) < 1e-12:
        failed.append((8, "torus span approach has no axial component"))
        return _Contacts(p, p, p, y, -y, np.nan, np.nan, np.inf, None, failed)
    t_equator = float((0.0 - p[2]) / a[2])
    if t_equator > L + atol:
        failed.append((5, "fingertips do not reach the torus equator plane"))
    t_base = min(max(t_equator, 0.0), L)
    base = p + t_base * a
    roots = _line_quadric_roots(base[:2], y[:2], R + r)  # outer equator radius
    if roots is None:
        failed.append((5, "closing line misses the torus outer equator"))
        return _Contacts(base, base, base, y, -y, np.nan, np.nan, np.inf, None, failed)
    s_lo, s_hi = roots
    contact_lo = base + s_lo * y
    contact_hi = base + s_hi * y
    span = s_hi - s_lo
    # Insertion below the approached extremal tube plane (z = ±r, nearest to palm).
    standoff = abs(p[2]) - r
    realized_depth = L - standoff
    n_lo = _torus_normal(prim, contact_lo)
    n_hi = _torus_normal(prim, contact_hi)
    return _Contacts(base, contact_lo, contact_hi, n_lo, n_hi, span, realized_depth, np.inf, 0.0, failed)


def _torus_normal(prim: Torus, point: np.ndarray) -> np.ndarray:
    R = prim.major
    rho = float(np.hypot(point[0], point[1]))
    if rho < 1e-12:
        return np.array([0.0, 0.0, np.sign(point[2]) or 1.0])
    ring = np.array([point[0] / rho * R, point[1] / rho * R, 0.0])  # nearest tube-centre point
    out = point - ring
    n = np.linalg.norm(out)
    return out / n if n > 0 else out


def _handle_torus_side(prim: Torus, p, a, y, x, L, c, atol) -> _Contacts:
    R, r = prim.major, prim.minor
    failed: List[Tuple[int, str]] = []
    if abs(_axis_component(a, _EZ)) > 1e-6:
        failed.append((8, "torus side approach is not radial (perpendicular to the torus axis)"))
    # Azimuth of the grasp from the approach direction (radial inward).
    a_xy = a[:2]
    if float(a_xy @ a_xy) < 1e-12:
        failed.append((8, "torus side approach has no radial component"))
        return _Contacts(p, p, p, y, -y, np.nan, np.nan, np.inf, None, failed)
    radial = -a_xy / np.linalg.norm(a_xy)  # inward approach -> outward radial
    phi = float(np.arctan2(radial[1], radial[0]))
    tube_center = np.array([R * np.cos(phi), R * np.sin(phi), 0.0])
    radial3 = np.array([np.cos(phi), np.sin(phi), 0.0])
    # Contacts across the minor-circle diameter: tube_center ± r·y_EE.
    contact_hi = tube_center + r * y
    contact_lo = tube_center - r * y
    span = 2.0 * r
    # Minor angle of the +y contact within the tube cross-section (radial, z).
    off = contact_hi - tube_center
    minor_angle = float(np.arctan2(off[2], off @ radial3))
    # Reach: the fingertip must reach the tube centre radius.
    rho_p = float(np.hypot(p[0], p[1]))
    standoff = rho_p - (R + r)
    realized_depth = L - standoff
    if (rho_p - R) > L + atol:
        failed.append((5, "torus tube is beyond finger reach"))
    n_lo = _torus_normal(prim, contact_lo)
    n_hi = _torus_normal(prim, contact_hi)
    return _Contacts(tube_center, contact_lo, contact_hi, n_lo, n_hi, span, realized_depth, np.inf, minor_angle, failed)


# --------------------------------------------------------------------------- #
# Public witness + certify.
# --------------------------------------------------------------------------- #


@dataclass
class GraspWitness:
    """Structured geometric witness for one concrete pose (issue #67).

    ``ok`` is True iff every soundness clause holds. ``failed`` lists
    ``(clause, message)`` for each violation so a failure reports geometry, not a
    bare Boolean. All quantities are pose-derived: ``realized_depth`` is the
    insertion from the approached surface [m] (the ``GraspProvenance.depth`` frame),
    ``boundary_clearance`` is the min margin to a free edge along the object extent,
    ``minor_angle`` is the torus tube cross-section angle (torus only, else None),
    and ``contact_plane_distance`` is the palm-to-closing-plane distance.
    """

    ok: bool
    palm_clearance: float
    realized_depth: float
    boundary_clearance: float
    span: float
    contact_neg: Optional[np.ndarray]
    contact_pos: Optional[np.ndarray]
    normal_neg: Optional[np.ndarray]
    normal_pos: Optional[np.ndarray]
    minor_angle: Optional[float] = None
    contact_plane_distance: float = np.nan
    failed: List[Tuple[int, str]] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.failed is None:
            self.failed = []


def certify(
    prim,
    pose,
    *,
    finger_length: float,
    max_aperture: float,
    preshape: float,
    clearance: float,
    mode: str,
    approach: Optional[str] = None,
    finger_orientation: Optional[str] = None,
) -> GraspWitness:
    """Certify the soundness clauses for ``prim`` at end-effector ``pose``.

    ``pose`` is the 4×4 end-effector transform in the primitive's reference frame
    (``E(ξ) = T_ref_tsr · xyzrpy_to_trans(ξ) · Tw_e``). Raises ``ValueError`` for a
    malformed *model* (bad dimensions/gripper/clearance or an unsupported mode,
    #95); a malformed *pose* returns a clause-1 :class:`GraspWitness` failure.

    ``mode`` selects the analytic model; ``approach``/``finger_orientation`` are
    optional structural labels checked against the realized geometry. None of these
    supply numeric evidence -- contacts are derived from the pose.
    """
    _validate_model(prim, finger_length, max_aperture, preshape, clearance, mode)
    pose_f, reason = _validate_pose(pose)
    if pose_f is None:
        return GraspWitness(False, np.nan, np.nan, np.nan, np.nan, None, None, None, None, failed=[(1, reason)])

    atol = length_atol(prim.scale)
    p = pose_f[:3, 3]
    a = pose_f[:3, 2]
    y = pose_f[:3, 1]
    x = pose_f[:3, 0]
    name = _primitive_name(prim)

    if name == "sphere":
        cg = _handle_sphere_surface(prim, p, a, y, x, finger_length, clearance, atol)
    elif name == "cylinder":
        if mode == "side":
            cg = _handle_cylinder_side(prim, p, a, y, x, finger_length, clearance, atol)
        else:
            cg = _handle_cylinder_cap(prim, p, a, y, x, finger_length, clearance, atol, top=(mode == "top"))
    elif name == "box":
        cg = _handle_box(prim, p, a, y, x, finger_length, clearance, atol, mode=mode)
    else:  # torus
        cg = (
            _handle_torus_span(prim, p, a, y, x, finger_length, clearance, atol)
            if mode == "span"
            else _handle_torus_side(prim, p, a, y, x, finger_length, clearance, atol)
        )

    failed: List[Tuple[int, str]] = list(cg.failed)

    # Clause 3: palm outside the solid with the requested clearance (SDF is the
    # authoritative signed distance here).
    palm_clearance = prim.sdf(p)
    if palm_clearance < clearance - atol:
        failed.append((3, f"palm clearance {palm_clearance:.6g} < requested c {clearance:.6g}"))

    span = cg.span
    contact_distance = float((cg.base - p) @ a) if np.all(np.isfinite(cg.base)) else np.nan

    if np.isfinite(span):
        # Independent SDF verification of the analytic contacts and normals (#93).
        verify_atol = 10.0 * atol
        for label, contact, normal in (("+", cg.contact_hi, cg.normal_hi), ("-", cg.contact_lo, cg.normal_lo)):
            if abs(prim.sdf(contact)) > verify_atol:
                failed.append((5, f"analytic {label} contact is not on the surface (sdf={prim.sdf(contact):.2e})"))
            elif float(sdf_normal(prim, contact) @ normal) < np.cos(1e-3):
                failed.append((6, f"analytic {label} normal disagrees with the SDF gradient"))

        # Clause 6: inward contact normals oppose the closing directions.
        ang_hi = float(np.arccos(np.clip(cg.normal_hi @ y, -1.0, 1.0)))
        ang_lo = float(np.arccos(np.clip(cg.normal_lo @ (-y), -1.0, 1.0)))
        if ang_hi > ANGLE_ATOL or ang_lo > ANGLE_ATOL:
            failed.append((6, f"contact normals do not oppose closing (angles {ang_hi:.2e}, {ang_lo:.2e} rad)"))

        # Clause 2: aperture validity and strict fit between the pads.
        if span >= preshape - atol:
            failed.append((2, f"object span {span:.6g} does not fit within preshape {preshape:.6g}"))

    # Clause 2: aperture bound (independent of whether contacts were found).
    if preshape > max_aperture + atol:
        failed.append((2, f"preshape {preshape:.6g} exceeds max aperture {max_aperture:.6g}"))

    # Optional declared-selector consistency (#94): checked, never trusted.
    if approach is not None or finger_orientation is not None:
        failed.extend(_check_declared_selectors(name, mode, a, y, approach, finger_orientation))

    return GraspWitness(
        ok=not failed,
        palm_clearance=float(palm_clearance),
        realized_depth=float(cg.realized_depth),
        boundary_clearance=float(cg.boundary_clearance),
        span=float(span) if np.isfinite(span) else np.nan,
        contact_neg=cg.contact_lo if np.isfinite(span) else None,
        contact_pos=cg.contact_hi if np.isfinite(span) else None,
        normal_neg=cg.normal_lo if np.isfinite(span) else None,
        normal_pos=cg.normal_hi if np.isfinite(span) else None,
        minor_angle=cg.minor_angle,
        contact_plane_distance=contact_distance,
        failed=failed,
    )


def _check_declared_selectors(name, mode, a, y, approach, finger_orientation) -> List[Tuple[int, str]]:
    """Check declared structural labels against realized pose axes (clause 8)."""
    out: List[Tuple[int, str]] = []
    axis_names = {0: "x", 1: "y", 2: "z"}
    if finger_orientation is not None and name == "box":
        y_match = _match_axis(y)
        if y_match is None or axis_names[y_match[0]] != finger_orientation:
            out.append((8, f"declared finger_orientation {finger_orientation!r} != realized closing axis"))
    if approach is not None and name == "box" and mode == "face":
        a_match = _match_axis(a)
        if a_match is not None:
            sign = "+" if a_match[1] > 0 else "-"
            realized = f"{sign}{axis_names[a_match[0]]}"
            if realized != approach:
                out.append((8, f"declared approach {approach!r} != realized approach {realized!r}"))
    return out


def pose_from(palm, approach, close) -> np.ndarray:
    """Build an SE(3) EE pose from a palm point and approach/close axes.

    ``approach`` → ``z_EE`` and ``close`` → ``y_EE`` (re-orthonormalized against the
    approach), with ``x_EE = y_EE × z_EE``. Test helper for hand-derived poses.
    """
    z = np.asarray(approach, dtype=float)
    z = z / np.linalg.norm(z)
    yv = np.asarray(close, dtype=float)
    yv = yv - (yv @ z) * z
    yv = yv / np.linalg.norm(yv)
    xv = np.cross(yv, z)
    T = np.eye(4)
    T[:3, 0] = xv
    T[:3, 1] = yv
    T[:3, 2] = z
    T[:3, 3] = np.asarray(palm, dtype=float)
    return T
