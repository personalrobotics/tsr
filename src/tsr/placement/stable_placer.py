# SPDX-License-Identifier: BSD-2-Clause
# Authors: Siddhartha Srinivasa and contributors to TSR

"""StablePlacer: generate stable placement TSRs for objects on a flat surface.

The placement contract is stated in ``docs/ARCHITECTURE.md``. In short, for **every**
pose a returned region admits, the object rests on the surface without penetrating it,
its centre of mass projects inside the contact patch, and the whole object stays within
the surface footprint.
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np

from ..core.utils import length_atol
from ..template import TSRTemplate
from ._stable_poses import _rotation_to_align, stable_poses_mesh

logger = logging.getLogger(__name__)

_NEG_Z = np.array([0.0, 0.0, -1.0])


def _check_positive_finite(name: str, value: float) -> float:
    """Validate a finite, strictly positive scalar, as the grasp factories do (#154)."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a finite positive number, got {value!r}")
    if not (np.isfinite(value) and value > 0.0):
        raise ValueError(f"{name} must be a finite positive number, got {value!r}")
    return float(value)


def _check_finite(name: str, value: float) -> float:
    """Validate a finite scalar of any sign (#154)."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    if not np.isfinite(value):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    return float(value)


def _check_finite_array(name: str, values: np.ndarray) -> np.ndarray:
    """Validate that every entry of an array is finite (#154)."""
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be finite, got {np.count_nonzero(~np.isfinite(values))} non-finite entry(s)")
    return values


class StablePlacer:
    """Generate stable placement TSRs for objects on a flat surface.

    Frame convention:
        Surface z points up; surface origin at the centre of the surface.
        The object frame is the caller's own frame: its origin is the point the
        returned poses place, which is the geometric centre for the primitives below
        and an arbitrary point for :meth:`place_mesh`.

    Surface extents:
        ``table_x`` and ``table_y`` bound the **object**, not just its origin. The
        sliding bounds are inset by the object's worst-case footprint over the free
        yaw, so every admitted pose keeps the whole object on the surface (#150). An
        object too large for the surface is a valid request with an empty feasible
        set, and the factory returns ``[]`` with the reason logged at ``DEBUG``.

    Args:
        table_x: Surface half-extent along x (m).
        table_y: Surface half-extent along y (m).
        reference: Reference frame name (default ``"table"``).

    Example::

        placer    = StablePlacer(table_x=0.3, table_y=0.2)
        templates = placer.place_cylinder(cylinder_radius=0.04,
                                          cylinder_height=0.12,
                                          subject="mug")
        tsr  = templates[0].instantiate(surface_pose)
        pose = tsr.sample()
    """

    def __init__(self, table_x: float, table_y: float, reference: str = "table"):
        self.table_x = _check_positive_finite("table_x", table_x)
        self.table_y = _check_positive_finite("table_y", table_y)
        self.reference = reference

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _bw(self, footprint_radius: float, roll_range=None, pitch_range=None) -> np.ndarray:
        """Build the 6×2 ``Bw``: inset xy slide, z fixed, yaw free.

        ``footprint_radius`` is the largest horizontal distance from the object frame
        origin to any point of the resting object. Because yaw is free, the footprint
        sweeps a disc of that radius about the origin, so insetting each axis by it is
        exactly the condition that the object stays on the surface at every yaw (#150).
        """
        half_x = max(self.table_x - footprint_radius, 0.0)
        half_y = max(self.table_y - footprint_radius, 0.0)
        bw = np.array(
            [
                [-half_x, half_x],
                [-half_y, half_y],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [-np.pi, np.pi],
            ]
        )
        if roll_range is not None:
            bw[3] = roll_range
        if pitch_range is not None:
            bw[4] = pitch_range
        return bw

    def _fits(self, footprint_radius: float, scale: float) -> bool:
        """Whether a footprint of this radius fits on the surface at every yaw.

        Equality is feasible, as everywhere else in the library: a footprint exactly
        the size of the surface leaves the single pose at the centre, which is an exact
        interval of zero width rather than an empty one.
        """
        atol = length_atol(scale)
        return footprint_radius <= self.table_x + atol and footprint_radius <= self.table_y + atol

    def _log_empty(self, method: str, reason: str, context: str) -> List[TSRTemplate]:
        """Report an empty feasible set with a machine-readable reason (#150).

        The factory contract in ``docs/ARCHITECTURE.md``: an exception reports an
        invalid request, an empty list reports a valid request with no feasible pose.
        """
        logger.debug("%s.%s: empty feasible set (%s; %s)", type(self).__name__, method, reason, context)
        return []

    def _template(
        self,
        name,
        description,
        variant,
        R,
        origin_height,
        Bw,
        subject,
        stability_margin=None,
    ) -> TSRTemplate:
        """Assemble one template from a resting rotation and the origin's resting height.

        The height goes in ``T_ref_tsr``, **not** in ``Tw_e``: a pose is
        ``T_ref_tsr @ xyzrpy_to_trans(ξ) @ Tw_e``, so any height carried by ``Tw_e``
        would be rotated by ``ξ``'s roll and pitch and the object would sink (#149).
        With the height ahead of ``ξ`` and ``Tw_e`` a pure rotation, the resting height
        is invariant under every rotation ``Bw`` admits, so freeing roll or pitch
        rotates the object in place, as a tilt tolerance should.
        """
        T_ref_tsr = np.eye(4)
        T_ref_tsr[2, 3] = _check_finite("origin_height", origin_height)
        Tw_e = np.eye(4)
        Tw_e[:3, :3] = R
        return TSRTemplate(
            T_ref_tsr=T_ref_tsr,
            Tw_e=Tw_e,
            Bw=Bw,
            task="place",
            subject=subject,
            reference=self.reference,
            name=name,
            description=description,
            variant=variant,
            stability_margin=stability_margin,
        )

    # ------------------------------------------------------------------
    # Primitive placement methods
    # ------------------------------------------------------------------

    def place_cylinder(
        self,
        cylinder_radius: float,
        cylinder_height: float,
        subject: str = "object",
    ) -> List[TSRTemplate]:
        """Return 2 placement templates: cylinder on each circular face.

        Object frame: origin at center, z = cylinder axis pointing up.
        Sideways (on the curved surface) is not stable and not returned.

        Both caps are returned because which cap faces the surface is caller-visible in
        ``variant`` and semantically distinct for a labelled object (a mug's opening up
        or down), even though the two occupy the same volume.

        Args:
            cylinder_radius: Cylinder radius (m).
            cylinder_height: Cylinder height (m).
            subject: Name of the object frame.

        Returns:
            Two templates, or ``[]`` if the cylinder cannot fit on the surface.
        """
        radius = _check_positive_finite("cylinder_radius", cylinder_radius)
        height = _check_positive_finite("cylinder_height", cylinder_height)

        # Resting on a cap the axis is vertical, so the footprint is the cap itself.
        footprint = radius
        scale = max(2.0 * radius, height)
        if not self._fits(footprint, scale):
            return self._log_empty(
                "place_cylinder",
                "exceeds_surface",
                f"footprint radius {footprint:.4g} m exceeds surface half-extent "
                f"min({self.table_x:.4g}, {self.table_y:.4g}) m",
            )

        candidates = [
            (np.array([0.0, 0.0, -1.0]), "-z"),
            (np.array([0.0, 0.0, +1.0]), "+z"),
        ]
        return [
            self._template(
                name=f"Place cylinder {label}-face down ({subject} on {self.reference})",
                description=(
                    f"Cylinder (r={radius:.3f} m, h={height:.3f} m) "
                    f"resting on {label} face on {self.reference}. Yaw free."
                ),
                variant=label,
                R=_rotation_to_align(n, _NEG_Z),
                origin_height=height / 2.0,
                Bw=self._bw(footprint),
                subject=subject,
            )
            for n, label in candidates
        ]

    def place_box(
        self,
        lx: float,
        ly: float,
        lz: float,
        subject: str = "object",
    ) -> List[TSRTemplate]:
        """Return 6 placement templates: one for each face of the box.

        Object frame: origin at center, axes aligned with box extents.
        All 6 faces are returned because opposite faces are semantically
        distinct (e.g. front vs back of a cereal box).

        Args:
            lx: Box x-extent (m).
            ly: Box y-extent (m).
            lz: Box z-extent (m).
            subject: Name of the object frame.

        Returns:
            Up to six templates: the faces whose footprint fits on the surface. ``[]``
            if no face fits.
        """
        lx = _check_positive_finite("lx", lx)
        ly = _check_positive_finite("ly", ly)
        lz = _check_positive_finite("lz", lz)
        scale = max(lx, ly, lz)

        # (outward normal of the resting face, origin height, footprint radius, label).
        # Each face is listed with its outward normal pointing toward the surface. The
        # footprint radius is the half-diagonal of the resting face, which is what the
        # free yaw sweeps -- an axis-aligned half-extent would still overhang (#150).
        candidates = [
            (np.array([0.0, 0.0, -1.0]), lz / 2.0, np.hypot(lx, ly) / 2.0, "-z"),
            (np.array([0.0, 0.0, +1.0]), lz / 2.0, np.hypot(lx, ly) / 2.0, "+z"),
            (np.array([0.0, -1.0, 0.0]), ly / 2.0, np.hypot(lx, lz) / 2.0, "-y"),
            (np.array([0.0, +1.0, 0.0]), ly / 2.0, np.hypot(lx, lz) / 2.0, "+y"),
            (np.array([-1.0, 0.0, 0.0]), lx / 2.0, np.hypot(ly, lz) / 2.0, "-x"),
            (np.array([+1.0, 0.0, 0.0]), lx / 2.0, np.hypot(ly, lz) / 2.0, "+x"),
        ]
        usable = [c for c in candidates if self._fits(c[2], scale)]
        if not usable:
            smallest = min(c[2] for c in candidates)
            return self._log_empty(
                "place_box",
                "exceeds_surface",
                f"smallest face footprint radius {smallest:.4g} m exceeds surface half-extent "
                f"min({self.table_x:.4g}, {self.table_y:.4g}) m",
            )
        if len(usable) < len(candidates):
            logger.debug(
                "%s.place_box: %d of %d faces dropped (exceeds_surface)",
                type(self).__name__,
                len(candidates) - len(usable),
                len(candidates),
            )

        return [
            self._template(
                name=f"Place box {label}-face down ({subject} on {self.reference})",
                description=(
                    f"Box ({lx:.3f}×{ly:.3f}×{lz:.3f} m) resting on {label} face on {self.reference}. Yaw free."
                ),
                variant=label,
                R=_rotation_to_align(n, _NEG_Z),
                origin_height=origin_h,
                Bw=self._bw(footprint),
                subject=subject,
            )
            for n, origin_h, footprint, label in usable
        ]

    def place_sphere(
        self,
        radius: float,
        subject: str = "object",
    ) -> List[TSRTemplate]:
        """Return one placement template: sphere on a flat surface.

        Every orientation is equally stable so roll and pitch are also free. The centre
        stays at ``z = radius`` for every admitted orientation, which the construction
        in :meth:`_template` guarantees structurally rather than only at zero tilt
        (#149).

        Args:
            radius: Sphere radius (m).
            subject: Name of the object frame.

        Returns:
            One template, or ``[]`` if the sphere cannot fit on the surface.
        """
        radius = _check_positive_finite("radius", radius)
        if not self._fits(radius, 2.0 * radius):
            return self._log_empty(
                "place_sphere",
                "exceeds_surface",
                f"footprint radius {radius:.4g} m exceeds surface half-extent "
                f"min({self.table_x:.4g}, {self.table_y:.4g}) m",
            )

        return [
            self._template(
                name=f"Place sphere ({subject} on {self.reference})",
                description=(f"Sphere (r={radius:.3f} m) on {self.reference}. All orientations free."),
                variant="upright",
                R=np.eye(3),
                origin_height=radius,
                Bw=self._bw(
                    radius,
                    roll_range=np.array([-np.pi, np.pi]),
                    pitch_range=np.array([-np.pi, np.pi]),
                ),
                subject=subject,
            )
        ]

    def place_torus(
        self,
        major_radius: float,
        minor_radius: float,
        subject: str = "object",
    ) -> List[TSRTemplate]:
        """Return 2 placement templates: torus flat on surface, each side down.

        Object frame: origin at center, z = torus symmetry axis pointing up.
        The torus rests on the bottom of the tube ring at z = -minor_radius.

        As for the cylinder's two caps, both sides are returned because which side faces
        the surface is caller-visible in ``variant`` and semantically distinct for a
        labelled object, even though a bare solid torus is invariant under the flip
        that separates them (#155).

        Args:
            major_radius: Distance from torus center to tube center (m).
            minor_radius: Tube radius (m).
            subject: Name of the object frame.

        Returns:
            Two templates, or ``[]`` if the torus cannot fit on the surface.
        """
        major_radius = _check_positive_finite("major_radius", major_radius)
        minor_radius = _check_positive_finite("minor_radius", minor_radius)
        if minor_radius >= major_radius:
            raise ValueError("minor_radius must be less than major_radius")

        footprint = major_radius + minor_radius  # the outer equator of the ring
        if not self._fits(footprint, 2.0 * footprint):
            return self._log_empty(
                "place_torus",
                "exceeds_surface",
                f"footprint radius {footprint:.4g} m exceeds surface half-extent "
                f"min({self.table_x:.4g}, {self.table_y:.4g}) m",
            )

        candidates = [
            (np.array([0.0, 0.0, -1.0]), "-z"),
            (np.array([0.0, 0.0, +1.0]), "+z"),
        ]
        return [
            self._template(
                name=f"Place torus {label}-face down ({subject} on {self.reference})",
                description=(
                    f"Torus (R={major_radius:.3f} m, r={minor_radius:.3f} m) "
                    f"flat, {label} face down on {self.reference}. Yaw free."
                ),
                variant=label,
                R=_rotation_to_align(n, _NEG_Z),
                origin_height=minor_radius,
                Bw=self._bw(footprint),
                subject=subject,
            )
            for n, label in candidates
        ]

    def place_mesh(
        self,
        vertices: np.ndarray,
        com: np.ndarray,
        subject: str = "object",
        min_margin_deg: float = 0.0,
    ) -> List[TSRTemplate]:
        """Return one template per stable resting face of the mesh.

        Uses the convex hull of ``vertices`` for stable-pose detection.
        Works for non-convex objects: the support polygon is the convex
        hull of the contact points.  Results are sorted by descending
        stability margin (most stable face first).

        Args:
            vertices:       (N, 3) array of object vertices in the object frame.
            com:            (3,) center of mass in the same frame.
            subject:        Name of the object frame.
            min_margin_deg: Discard faces whose stability margin is below this
                            threshold (degrees). Default 0 returns all stable faces.

        Returns:
            One template per stable face that fits on the surface, or ``[]`` when no
            face is both stable and small enough, with the reason logged at ``DEBUG``.
        """
        vertices = np.asarray(vertices, dtype=float)
        com = np.asarray(com, dtype=float)
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError("vertices must have shape (N, 3)")
        if com.shape != (3,):
            raise ValueError("com must be a length-3 array")
        _check_finite_array("vertices", vertices)
        _check_finite_array("com", com)
        _check_finite("min_margin_deg", min_margin_deg)

        scale = float(np.ptp(vertices, axis=0).max()) if len(vertices) else 0.0
        min_margin_rad = float(np.radians(min_margin_deg))
        stable = sorted(stable_poses_mesh(vertices, com), key=lambda x: -x[2])  # descending stability margin
        poses = [p for p in stable if p[2] >= min_margin_rad]
        if not poses:
            return self._log_empty(
                "place_mesh",
                "no_stable_face" if not stable else "below_min_margin",
                f"{len(stable)} stable face(s), none with margin >= {min_margin_deg:.4g}°",
            )

        # The footprint is measured from the object frame origin, which Bw slides, and
        # not from the centroid: place_mesh accepts an arbitrary origin (#150).
        usable = []
        for R, origin_height, margin_rad in poses:
            footprint = float(np.linalg.norm((vertices @ R.T)[:, :2], axis=1).max())
            if self._fits(footprint, scale):
                usable.append((R, origin_height, margin_rad, footprint))
        if not usable:
            return self._log_empty(
                "place_mesh",
                "exceeds_surface",
                f"all {len(poses)} stable face(s) exceed surface half-extent "
                f"min({self.table_x:.4g}, {self.table_y:.4g}) m",
            )
        if len(usable) < len(poses):
            logger.debug(
                "%s.place_mesh: %d of %d stable faces dropped (exceeds_surface)",
                type(self).__name__,
                len(poses) - len(usable),
                len(poses),
            )

        templates = []
        for idx, (R, origin_height, margin_rad, footprint) in enumerate(usable):
            deg = float(np.degrees(margin_rad))
            templates.append(
                self._template(
                    name=(
                        f"Place mesh face {idx + 1}/{len(usable)} ({subject} on {self.reference}, margin {deg:.1f}°)"
                    ),
                    description=(
                        f"Mesh resting on stable face {idx + 1} of {len(usable)} "
                        f"(stability margin {deg:.1f}°) on {self.reference}."
                    ),
                    variant=f"face-{idx + 1}",
                    R=R,
                    origin_height=origin_height,
                    Bw=self._bw(footprint),
                    subject=subject,
                    stability_margin=float(margin_rad),
                )
            )
        return templates
