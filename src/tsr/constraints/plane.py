# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Plane constraint: keep the end-effector origin on a CGA plane.

Wraps a :class:`gafropy.Plane` (a grade-1 vector in CGA, the dual of the
"flat" point-at-infinity object). The constraint is satisfied when the
end-effector *position* lies on the plane; orientation can be left free or
pinned to a reference rotor.

The witness exchanged with the planner is the projected world-frame ``Motor``:
:meth:`distance` returns the pose slid onto the plane (and, if pinned, with the
reference orientation), and :meth:`to_transform` simply returns it. This keeps
the seam identical to :class:`~tsr.tsr.TSR` (``distance`` -> witness ->
``to_transform``) while using the native CGA primitive for the geometry.
"""

from __future__ import annotations

import numpy as np

from gafropy import Motor, Plane, Point, Rotor

from .base import Constraint


def _motor_from_translation_rotor(xyz, rotor):
    """``Translator(xyz) * Rotor`` as a Motor (the projected-pose witness)."""
    return Motor.from_translation_rotor(float(xyz[0]), float(xyz[1]), float(xyz[2]), rotor)


class PlaneConstraint(Constraint):
    """Constrain the end-effector origin to a :class:`gafropy.Plane`.

    @param plane          the gafropy ``Plane`` defining the manifold.
    @param orientation    optional reference orientation as a rotor-bivector log
                          3-vector ``[b12, b13, b23]`` (axis*angle). When given,
                          the constraint also pins orientation to this rotor and
                          the rotation error contributes to :meth:`distance`.
                          When ``None`` (default), orientation is unconstrained.
    """

    def __init__(self, plane: Plane, orientation: np.ndarray | None = None):
        self.plane = plane
        if orientation is None:
            self._rotor = None
        else:
            biv = np.asarray(orientation, dtype=float).reshape(3)
            self._rotor = Rotor.exp(float(biv[0]), float(biv[1]), float(biv[2]))

    # ------------------------------------------------------------------
    # factories
    # ------------------------------------------------------------------

    @classmethod
    def from_point_normal(cls, point, normal, orientation=None) -> "PlaneConstraint":
        """Build a plane through ``point`` with the given ``normal`` 3-vector.

        Constructed via the three-point CGA constructor (``Plane(p0, p1, p2)``)
        rather than raw coefficient packing, so it is independent of the
        ``Plane.from_array`` coefficient convention: we span the plane with two
        orthonormal in-plane directions through ``point``.
        """
        p = np.asarray(point, dtype=float).reshape(3)
        n = np.asarray(normal, dtype=float).reshape(3)
        n = n / np.linalg.norm(n)
        # Two in-plane axes orthogonal to n.
        helper = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        u = np.cross(n, helper)
        u /= np.linalg.norm(u)
        v = np.cross(n, u)
        plane = Plane(
            Point(*(float(c) for c in p)),
            Point(*(float(c) for c in p + u)),
            Point(*(float(c) for c in p + v)),
        )
        return cls(plane, orientation=orientation)

    # ------------------------------------------------------------------
    # constraint interface
    # ------------------------------------------------------------------

    def _ee_point(self, trans) -> tuple[Motor, Point, np.ndarray]:
        """Return ``(motor, ee_point, ee_xyz)`` for a pose ``trans``."""
        m = Motor(trans)
        xyz = np.asarray(m.get_translator().to_array(), dtype=float)
        return m, Point(float(xyz[0]), float(xyz[1]), float(xyz[2])), xyz

    def _project_point(self, xyz: np.ndarray) -> np.ndarray:
        """Foot of the perpendicular from ``xyz`` onto the plane.

        Uses the native CGA ``Plane.project`` formula
        (``(point | plane) * plane.inverse()``), which returns the exact
        orthogonal projection of the point onto the plane.
        """
        foot = self.plane.project(Point(*(float(c) for c in xyz)))
        return np.asarray(foot.to_array(), dtype=float)

    def distance(self, trans, rotation_weight: float = 1.0) -> tuple[float, Motor]:
        m, ee, xyz = self._ee_point(trans)

        # Translation error: perpendicular distance to the plane; the projected
        # foot is the orthogonal projection onto the plane (see _project_point).
        d_trans = self.plane.get_distance_to_point(ee)
        proj_xyz = self._project_point(xyz)

        rotor = m.get_rotor() if self._rotor is None else self._rotor
        witness = _motor_from_translation_rotor(proj_xyz, rotor)

        if self._rotor is None:
            dist = abs(d_trans)
        else:
            # Rotation error: minimal angle of the relative rotor between pose and
            # pin, via its bivector-log norm (smooth, wrapped to [0, pi]).
            rel = m.get_rotor().inverse().multiply(self._rotor)
            d_rot = float(np.linalg.norm(rel.log().to_array()))
            dist = float(np.hypot(d_trans, rotation_weight * d_rot))

        return float(dist), witness

    def to_transform(self, witness: Motor) -> Motor:
        return Motor(witness)

    def sample(self) -> Motor:
        """Draw a world-frame pose on the plane.

        Samples a point on the plane by projecting a random point onto it, and
        uses the pinned orientation (or identity when unconstrained).
        """
        rand = np.random.uniform(-1.0, 1.0, size=3)
        xyz = self._project_point(rand)
        rotor = self._rotor if self._rotor is not None else Rotor.exp(0.0, 0.0, 0.0)
        return _motor_from_translation_rotor(xyz, rotor)

    # ------------------------------------------------------------------
    # serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        d = {
            "format": "plane-constraint-v1",
            "plane": np.asarray(self.plane.to_array(), dtype=float).tolist(),
        }
        if self._rotor is not None:
            d["orientation"] = np.asarray(self._rotor.log().to_array(), dtype=float).tolist()
        return d

    @staticmethod
    def from_dict(x: dict) -> "PlaneConstraint":
        fmt = x.get("format")
        if fmt != "plane-constraint-v1":
            raise ValueError(f"Unsupported PlaneConstraint serialization format: {fmt!r}")
        plane = Plane.from_array(np.asarray(x["plane"], dtype=float))
        return PlaneConstraint(plane, orientation=x.get("orientation"))

    def __repr__(self) -> str:
        coeffs = np.asarray(self.plane.to_array(), dtype=float).round(3)
        pinned = "free" if self._rotor is None else "pinned"
        return f"PlaneConstraint(plane={coeffs.tolist()}, orientation={pinned})"
