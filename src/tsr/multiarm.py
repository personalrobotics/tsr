# tsr/src/tsr/multiarm.py
# SPDX-License-Identifier: MIT
"""Task Space Regions for three- and four-arm cooperative systems.

Where a dual-arm system is described by an (absolute, relative) *Motor* pair
(see :mod:`tsr.bimanual`), gafro describes a three- or four-arm system by a
single :class:`~gafro.SimilarityTransformation`: the transform carrying a
canonical primitive onto the one the end-effectors currently span. Three points
span a **circle**, four span a **sphere**, and the transform between two such
primitives is a *similarity* — it carries a dilation as well as a translation
and rotation, because the arms can grow or shrink the shape they hold.

**Coordinates.** A similarity transform decomposes (via
``get_canonical_decomposition``) into translator, rotor and dilator: 3 + 3 + 1 =
7 numbers. The TSRs here do not use all seven, because the spanned primitive is
symmetric under some of the rotations and constraining a symmetric DOF is
meaningless:

``SphereTSR`` (four arms) — **4 DOF** ``[tx, ty, tz, dilation]``
    A sphere is isotropic: no rotation of a sphere about its own centre changes
    it, so all three rotation components are dropped. What remains is where the
    sphere's centre is and how big it is.

``CircleTSR`` (three arms) — **6 DOF** ``[tx, ty, tz, dilation, n1, n2]``
    A circle is symmetric only about its own axis. Two rotation components
    orient the circle's plane (they carry the plane normal); the third — the
    spin *about* that normal — leaves the circle unchanged and is dropped.

``dilation`` is the **logarithm** of the scale factor, so that 0 means "same
size", it is signed and symmetric about 0 (halving and doubling are equally far
from unity), and it composes additively like the other log coordinates. Recover
the raw factor with ``exp(dilation)``.

Both classes follow the :class:`~tsr.tsr.TSR` protocol the planner already
speaks — ``sample`` / ``distance`` / ``to_transform`` / ``contains`` /
``volume`` — so they drop into the same machinery. They are not
:class:`~tsr.constraints.base.Constraint` subclasses for the same reason
:class:`~tsr.bimanual.BimanualTSR` is not: that ABC is defined over a single
world-frame end-effector Motor, and these speak in similarity transforms.
"""
from __future__ import annotations

import numpy as np
from gafro import Circle, Dilator, Motor, Point, Rotor, Sphere, Vector
from gafro import SimilarityTransformation as _Similarity

__all__ = [
    "CircleTSR",
    "SphereTSR",
    "circle_from_center_radius_normal",
    "decompose_similarity",
    "dilation_log",
    "is_degenerate",
    "sphere_from_center_radius",
]

# Canonical primitives the task-space similarity is measured against: the unit
# circle in the z = 0 plane and the unit sphere, both centred at the origin.
_UNIT_NORMAL = np.array([0.0, 0.0, 1.0])
# How close the dilator ratio may come to the atanh pole before the spanned
# primitive counts as degenerate (|log scale| beyond ~log(4e3)).
_DILATION_RATIO_MARGIN = 1e-7


def dilation_log(dilator: "Dilator") -> float:
    """Log of the scale factor carried by a ``Dilator``.

    A ``Dilator``'s two coefficients ``[a0, a1]`` satisfy
    ``scale = exp(2 * atanh(a1 / a0))``; this returns that exponent, so a
    dilator of scale ``s`` gives ``log(s)`` and the identity gives ``0``.

    Raises:
        ValueError: if the ratio has reached +/-1, where ``atanh`` diverges.
            That means the spanned primitive has degenerated -- most often four
            *coplanar* end-effectors, which have no finite circumsphere. Break
            the symmetry of the arm configuration and the scale becomes finite
            again. Failing here beats returning an infinity that silently turns
            a Jacobian into NaN.
    """
    a = np.asarray(dilator.to_array(), dtype=float)
    if a[0] == 0.0:
        raise ValueError("degenerate dilator (zero leading coefficient)")
    ratio = a[1] / a[0]
    # atanh loses all precision well before the exact pole: at 1 - 1e-9 the
    # implied scale is already ~1e9, which is a degenerate primitive, not a
    # measurement. Reject the whole ill-conditioned neighbourhood.
    if abs(ratio) >= 1.0 - _DILATION_RATIO_MARGIN:
        raise ValueError(
            f"degenerate dilation: dilator ratio {ratio!r} is at the atanh pole, so the "
            "spanned primitive has no finite scale (four coplanar end-effectors have no "
            "circumsphere). Perturb the arm configuration to break the symmetry.")
    return float(2.0 * np.arctanh(ratio))


def dilator_from_log(value: float) -> "Dilator":
    """Inverse of :func:`dilation_log`: a ``Dilator`` of scale ``exp(value)``."""
    return Dilator(float(np.exp(value)))


def is_degenerate(similarity) -> bool:
    """Whether a similarity transform's spanned primitive has no finite scale.

    Sampling-based planners visit configurations where the end-effectors are
    (nearly) coplanar, and those have no circumsphere. Such a configuration is
    simply *not usable* rather than an error, so callers that explore
    configuration space should test with this instead of catching the
    ``ValueError`` from :func:`dilation_log`.
    """
    try:
        dilation_log(similarity.get_canonical_decomposition().get_dilator())
    except ValueError:
        return True
    return False


def sphere_from_center_radius(center, radius: float) -> "Sphere":
    """Sphere with the given centre and radius."""
    c = np.asarray(center, dtype=float)
    return Sphere(Point(*(float(x) for x in c)), float(radius))


def circle_from_center_radius_normal(center, radius: float, normal) -> "Circle":
    """Circle with the given centre, radius and plane normal.

    gafro builds a ``Circle`` from three points, so three equally spaced points
    are placed on the requested circle.
    """
    c = np.asarray(center, dtype=float)
    n = np.asarray(normal, dtype=float)
    norm = np.linalg.norm(n)
    if norm < 1e-12:
        raise ValueError("circle normal must be non-zero")
    n = n / norm
    # Any vector not parallel to n gives a starting in-plane direction.
    seed = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n, seed)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    angles = (0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0)
    points = [c + radius * (np.cos(t) * u + np.sin(t) * v) for t in angles]
    return Circle(*(Point(*(float(x) for x in p)) for p in points))


def decompose_similarity(similarity) -> tuple[np.ndarray, np.ndarray, float]:
    """Split a ``SimilarityTransformation`` into ``(translation, rotor_log, dilation)``.

    Returns the translation as ``[tx, ty, tz]``, the rotor bivector log as a
    3-vector in ``Rotor.log()`` order, and the dilation log as a scalar.
    """
    decomposition = similarity.get_canonical_decomposition()
    translator = decomposition.get_translator()
    translation = np.array([translator.x(), translator.y(), translator.z()], dtype=float)
    rotor_log = np.asarray(decomposition.get_rotor().log().to_array(), dtype=float)
    return translation, rotor_log, dilation_log(decomposition.get_dilator())


def _rotor_from_log(bivector) -> "Rotor":
    """Rotor whose bivector log is ``bivector`` (3-vector, ``Rotor.log()`` order)."""
    b = np.asarray(bivector, dtype=float)
    return Rotor.exp(float(b[0]), float(b[1]), float(b[2]))


def _clip_to_box(values: np.ndarray, box: np.ndarray) -> np.ndarray:
    return np.clip(values, box[:, 0], box[:, 1])


def _displacement_to_box(values: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Signed distance from each component to its interval (0 when inside)."""
    below = np.minimum(values - box[:, 0], 0.0)
    above = np.maximum(values - box[:, 1], 0.0)
    return below + above


def _interval_sum(box: np.ndarray, angular_rows=()) -> float:
    """Summed interval width, with angular rows clamped to 2π (cf. ``TSR.volume``)."""
    widths = np.asarray(box[:, 1] - box[:, 0], dtype=float)
    for row in angular_rows:
        widths[row] = min(widths[row], 2.0 * np.pi)
    return float(np.sum(np.maximum(widths, 0.0)))


class _SimilarityTSR:
    """Shared machinery for the similarity-transform TSRs.

    Subclasses declare their coordinate layout (``_DOF``, ``_LABELS``,
    ``_ANGULAR_ROWS``) and implement the primitive round-trip
    (``_primitive_from_bw`` / ``_bw_from_similarity``).
    """

    _DOF: int = 0
    _LABELS: tuple[str, ...] = ()
    _ANGULAR_ROWS: tuple[int, ...] = ()

    def __init__(self, Bw=None):
        if Bw is None:
            Bw = np.zeros((self._DOF, 2))
        self.Bw = np.array(Bw, dtype=float)
        if self.Bw.shape != (self._DOF, 2):
            raise ValueError(
                f"{type(self).__name__} Bw must be shape ({self._DOF},2) "
                f"for {list(self._LABELS)}, got {self.Bw.shape}")
        if np.any(self.Bw[:, 0] > self.Bw[:, 1] + 1e-9):
            raise ValueError("Bw bounds must be [min, max] for every row", Bw)
        Bw_cont = np.copy(self.Bw)
        for row in self._ANGULAR_ROWS:
            Bw_cont[row, :] = np.clip(Bw_cont[row, :], -np.pi, np.pi)
        self._Bw_cont = Bw_cont

    def __repr__(self) -> str:
        free = [label for i, label in enumerate(self._LABELS)
                if not np.isclose(self.Bw[i, 0], self.Bw[i, 1])]
        return f"{type(self).__name__}(free=[{','.join(free) if free else 'fixed'}])"

    @property
    def dof(self) -> int:
        return self._DOF

    def sample_bw(self, bw=None) -> np.ndarray:
        """Sample a ``bw`` vector; NaN (or omitted) components are drawn uniformly."""
        if bw is None:
            bw = np.full(self._DOF, np.nan)
        bw = np.asarray(bw, dtype=float)
        if bw.shape != (self._DOF,):
            raise ValueError(f"bw must have length {self._DOF}, got {bw.shape}")
        low, high = self._Bw_cont[:, 0], self._Bw_cont[:, 1]
        drawn = low + (high - low) * np.random.random_sample(self._DOF)
        return np.where(np.isnan(bw), drawn, bw)

    def sample(self, bw=None):
        """Sample a ``SimilarityTransformation`` from the region."""
        return self.to_transform(self.sample_bw(bw))

    def to_transform(self, bw):
        """Build the ``SimilarityTransformation`` for a ``bw`` vector."""
        bw = np.asarray(bw, dtype=float)
        if bw.shape != (self._DOF,):
            raise ValueError(f"bw must have length {self._DOF}, got {bw.shape}")
        return _Similarity.between(self._canonical_primitive(), self._primitive_from_bw(bw))

    def to_bw(self, similarity) -> np.ndarray:
        """Coordinates of a ``SimilarityTransformation`` in this region's layout."""
        return self._bw_from_similarity(similarity)

    def distance(self, similarity, rotation_weight: float = 1.0):
        """Distance from the region to a similarity transform.

        A *degenerate* pose (end-effectors with no finite circumsphere) is
        infinitely far from any region: sampling planners hand these in while
        exploring, and they must read as "does not satisfy" rather than raising.

        @return ``(dist, bwopt)`` — 0 when inside, and the closest admissible
                ``bw`` (the witness the planner feeds back to ``to_transform``).
        """
        if is_degenerate(similarity):
            return float("inf"), self._clip_centre()
        bw = self._bw_from_similarity(similarity)
        displacement = _displacement_to_box(bw, self._Bw_cont)
        for row in self._ANGULAR_ROWS:
            displacement[row] *= rotation_weight
        return float(np.linalg.norm(displacement)), _clip_to_box(bw, self._Bw_cont)

    def _clip_centre(self) -> np.ndarray:
        """Midpoint of the box; the witness offered for an unusable pose."""
        return 0.5 * (self._Bw_cont[:, 0] + self._Bw_cont[:, 1])

    def contains(self, similarity, tolerance: float = 1e-9) -> bool:
        dist, _ = self.distance(similarity)
        return dist <= tolerance

    @property
    def volume(self) -> float:
        return _interval_sum(self.Bw, self._ANGULAR_ROWS)


class SphereTSR(_SimilarityTSR):
    """Four-arm region over ``[tx, ty, tz, dilation]``.

    The four end-effectors span a sphere; this constrains where that sphere's
    centre sits and how large it is. A sphere is unchanged by rotation about its
    centre, so no rotation coordinate appears.

    Args:
        Bw: (4,2) bounds over ``[tx, ty, tz, dilation]``. ``dilation`` is the
            log scale, so ``[0, 0]`` pins the sphere to the canonical radius and
            ``[-0.7, 0.7]`` admits roughly half to double that size.
        radius: radius of the canonical sphere the transform is measured
            against (default 1).
    """

    _DOF = 4
    _LABELS = ("tx", "ty", "tz", "dilation")
    _ANGULAR_ROWS = ()

    def __init__(self, Bw=None, radius: float = 1.0):
        super().__init__(Bw)
        self.radius = float(radius)

    def _canonical_primitive(self):
        return sphere_from_center_radius(np.zeros(3), self.radius)

    def _primitive_from_bw(self, bw: np.ndarray):
        return sphere_from_center_radius(bw[0:3], self.radius * float(np.exp(bw[3])))

    def _bw_from_similarity(self, similarity) -> np.ndarray:
        translation, _rotor_log, dilation = decompose_similarity(similarity)
        return np.hstack((translation, dilation))


class CircleTSR(_SimilarityTSR):
    """Three-arm region over ``[tx, ty, tz, dilation, n1, n2]``.

    The three end-effectors span a circle; this constrains its centre, its size
    and the orientation of its plane. Of the rotor's three bivector components
    the first is the spin *about* the circle's own normal, which leaves the
    circle unchanged — so it is dropped and the remaining two (``n1``, ``n2``)
    carry the plane normal.

    Args:
        Bw: (6,2) bounds over ``[tx, ty, tz, dilation, n1, n2]``. ``dilation``
            is the log scale; ``n1``/``n2`` are radians and are clamped to
            ``[-pi, pi]``.
        radius: radius of the canonical circle the transform is measured
            against (default 1).
    """

    _DOF = 6
    _LABELS = ("tx", "ty", "tz", "dilation", "n1", "n2")
    # n1 / n2 are the retained rotor-bivector components (rows 4 and 5).
    _ANGULAR_ROWS = (4, 5)
    # The dropped spin component is the first of Rotor.log(); the two retained
    # ones follow it.
    _SPIN_ROW = 0

    def __init__(self, Bw=None, radius: float = 1.0):
        super().__init__(Bw)
        self.radius = float(radius)

    def _canonical_primitive(self):
        return circle_from_center_radius_normal(np.zeros(3), self.radius, _UNIT_NORMAL)

    def _primitive_from_bw(self, bw: np.ndarray):
        # Rebuild the rotor with a zero spin component, then carry the canonical
        # normal through it to get this circle's plane.
        rotor = _rotor_from_log((0.0, bw[4], bw[5]))
        rotated = Motor.from_rotor(rotor).transform_vector(
            Vector(*(float(x) for x in _UNIT_NORMAL)))
        normal = np.array([rotated.x(), rotated.y(), rotated.z()], dtype=float)
        return circle_from_center_radius_normal(
            bw[0:3], self.radius * float(np.exp(bw[3])), normal)

    def _bw_from_similarity(self, similarity) -> np.ndarray:
        translation, rotor_log, dilation = decompose_similarity(similarity)
        # Drop the spin component; keep the two that orient the plane.
        plane = np.delete(rotor_log, self._SPIN_ROW)
        return np.hstack((translation, dilation, plane))
