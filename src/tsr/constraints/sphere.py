# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Sphere constraint: keep the end-effector origin on a CGA sphere shell.

Wraps a :class:`gafro.Sphere` (a grade-1 vector in CGA). The constraint is
satisfied when the end-effector *position* lies on the spherical surface (a
fixed distance ``radius`` from the center); orientation can be left free or
pinned to a reference rotor.

This is the spherical sibling of :class:`~tsr.constraints.plane.PlaneConstraint`
and uses the native CGA ``Sphere.project`` formula (which returns the point on
the shell nearest the query). The witness exchanged with the planner is the
projected world-frame ``Motor``, so the seam is identical to ``PlaneConstraint``
and :class:`~tsr.tsr.TSR` (``distance`` -> witness -> ``to_transform``).
"""

from __future__ import annotations

import numpy as np
from gafro import Motor, Point, Rotor, Sphere

from ..utils import as_motor, position
from .base import Constraint


def _motor_from_translation_rotor(xyz, rotor):
    """``Translator(xyz) * Rotor`` as a Motor (the projected-pose witness)."""
    return Motor.from_translation_rotor(float(xyz[0]), float(xyz[1]), float(xyz[2]), rotor)


class SphereConstraint(Constraint):
    """Constrain the end-effector origin to a :class:`gafro.Sphere` shell.

    @param sphere         the gafro ``Sphere`` defining the manifold.
    @param orientation    optional reference orientation as a rotor-bivector log
                          3-vector ``[b12, b13, b23]`` (axis*angle). When given,
                          the constraint also pins orientation to this rotor and
                          the rotation error contributes to :meth:`distance`.
                          When ``None`` (default), orientation is unconstrained.
    """

    def __init__(self, sphere: Sphere, orientation: np.ndarray | None = None):
        self.sphere = sphere
        if orientation is None:
            self._rotor = None
        else:
            biv = np.asarray(orientation, dtype=float).reshape(3)
            self._rotor = Rotor.exp(float(biv[0]), float(biv[1]), float(biv[2]))

    # ------------------------------------------------------------------
    # factories
    # ------------------------------------------------------------------

    @classmethod
    def from_center_radius(cls, center, radius: float, orientation=None) -> "SphereConstraint":
        """Build a sphere of ``radius`` about ``center`` (3-vector)."""
        c = np.asarray(center, dtype=float).reshape(3)
        sphere = Sphere(Point(float(c[0]), float(c[1]), float(c[2])), float(radius))
        return cls(sphere, orientation=orientation)

    # ------------------------------------------------------------------
    # constraint interface
    # ------------------------------------------------------------------

    def _ee_point(self, trans) -> tuple[Motor, Point, np.ndarray]:
        """Return ``(motor, ee_point, ee_xyz)`` for a pose ``trans``."""
        m = as_motor(trans)
        xyz = position(m.get_translator())
        return m, Point(float(xyz[0]), float(xyz[1]), float(xyz[2])), xyz

    def _project_point(self, xyz: np.ndarray) -> np.ndarray:
        """Nearest point on the sphere shell to ``xyz``.

        Uses the native CGA ``Sphere.project`` formula, which intersects the
        line center->point with the sphere and returns the nearer of the two
        antipodal roots -- i.e. the closest point on the spherical surface.

        A query at the exact center is degenerate (equidistant from the whole
        shell, ``project`` returns NaN); there we pick an arbitrary but
        deterministic shell point along +x.
        """
        center = position(self.sphere.get_center())
        if np.linalg.norm(xyz - center) < 1e-9:
            radius = float(self.sphere.get_radius())
            return center + np.array([radius, 0.0, 0.0])
        foot = self.sphere.project(Point(*(float(c) for c in xyz)))
        return position(foot)

    def distance(self, trans, rotation_weight: float = 1.0) -> tuple[float, Motor]:
        m, _, xyz = self._ee_point(trans)

        # Translation error: distance from the point to its projection on the
        # shell (the foot from Sphere.project). This is the unsigned radial
        # offset |‖xyz - center‖ - radius| and never depends on inside/outside.
        proj_xyz = self._project_point(xyz)
        d_trans = float(np.linalg.norm(xyz - proj_xyz))

        rotor = m.get_rotor() if self._rotor is None else self._rotor
        witness = _motor_from_translation_rotor(proj_xyz, rotor)

        if self._rotor is None:
            dist = d_trans
        else:
            # Rotation error: minimal angle of the relative rotor between pose and
            # pin, via its bivector-log norm (smooth, wrapped to [0, pi]).
            rel = m.get_rotor().inverse().multiply(self._rotor)
            d_rot = float(np.linalg.norm(rel.log().to_array()))
            dist = float(np.hypot(d_trans, rotation_weight * d_rot))

        return float(dist), witness

    def to_transform(self, witness: Motor) -> Motor:
        return as_motor(witness)

    def sample(self) -> Motor:
        """Draw a world-frame pose on the sphere shell.

        Samples a point on the shell by projecting a random point (offset from
        the center) onto it, and uses the pinned orientation (or identity when
        unconstrained).
        """
        center = position(self.sphere.get_center())
        # Random direction away from the center; projecting it lands on the shell.
        offset = np.random.uniform(-1.0, 1.0, size=3)
        if np.linalg.norm(offset) < 1e-9:
            offset = np.array([1.0, 0.0, 0.0])
        xyz = self._project_point(center + offset)
        rotor = self._rotor if self._rotor is not None else Rotor.exp(0.0, 0.0, 0.0)
        return _motor_from_translation_rotor(xyz, rotor)

    # ------------------------------------------------------------------
    # serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        d = {
            "format": "sphere-constraint-v1",
            "sphere": np.asarray(self.sphere.to_array(), dtype=float).tolist(),
        }
        if self._rotor is not None:
            d["orientation"] = np.asarray(self._rotor.log().to_array(), dtype=float).tolist()
        return d

    @staticmethod
    def from_dict(x: dict) -> "SphereConstraint":
        fmt = x.get("format")
        if fmt != "sphere-constraint-v1":
            raise ValueError(f"Unsupported SphereConstraint serialization format: {fmt!r}")
        sphere = Sphere.from_array(np.asarray(x["sphere"], dtype=float))
        return SphereConstraint(sphere, orientation=x.get("orientation"))

    def __repr__(self) -> str:
        c = position(self.sphere.get_center()).round(3)
        r = round(float(self.sphere.get_radius()), 3)
        pinned = "free" if self._rotor is None else "pinned"
        return f"SphereConstraint(center={c.tolist()}, radius={r}, orientation={pinned})"
