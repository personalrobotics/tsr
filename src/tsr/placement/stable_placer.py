"""StablePlacer: generate stable placement TSRs for objects on a flat surface."""
from __future__ import annotations

from typing import List

import numpy as np

from ..template import TSRTemplate
from ._stable_poses import _rotation_to_align, stable_poses_mesh


class StablePlacer:
    """Generate stable placement TSRs for objects on a flat surface.

    Frame convention:
        Surface z points up; surface origin at the center.
        Object frame origin at geometric center (= COM for uniform density).

    Args:
        table_x: Surface half-extent along x (m). Sampled poses slide ±table_x.
        table_y: Surface half-extent along y (m). Sampled poses slide ±table_y.
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
        if table_x <= 0 or table_y <= 0:
            raise ValueError("table_x and table_y must be positive")
        self.table_x = float(table_x)
        self.table_y = float(table_y)
        self.reference = reference

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _bw(self, roll_range=None, pitch_range=None) -> np.ndarray:
        """Build 6×2 Bw: surface extents for xy, z/roll/pitch fixed, yaw free."""
        bw = np.array([
            [-self.table_x,  self.table_x],
            [-self.table_y,  self.table_y],
            [0.0,            0.0],
            [0.0,            0.0],
            [0.0,            0.0],
            [-np.pi,         np.pi],
        ])
        if roll_range is not None:
            bw[3] = roll_range
        if pitch_range is not None:
            bw[4] = pitch_range
        return bw

    def _tw_e(self, R: np.ndarray, com_height: float) -> np.ndarray:
        """Build 4×4 Tw_e from rotation R and COM height above the surface."""
        T = np.eye(4)
        T[:3, :3] = R
        T[2, 3] = float(com_height)
        return T

    def _template(self, name, description, variant, Tw_e, Bw, subject,
                  stability_margin=None) -> TSRTemplate:
        return TSRTemplate(
            T_ref_tsr=np.eye(4),
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

        Args:
            cylinder_radius: Cylinder radius (m).
            cylinder_height: Cylinder height (m).
            subject: Name of the object frame.
        """
        if cylinder_radius <= 0:
            raise ValueError("cylinder_radius must be positive")
        if cylinder_height <= 0:
            raise ValueError("cylinder_height must be positive")

        _neg_z = np.array([0.0, 0.0, -1.0])
        candidates = [
            (np.array([0.0, 0.0, -1.0]), "-z"),
            (np.array([0.0, 0.0, +1.0]), "+z"),
        ]
        return [
            self._template(
                name=f"Place cylinder {label}-face down ({subject} on {self.reference})",
                description=(
                    f"Cylinder (r={cylinder_radius:.3f} m, h={cylinder_height:.3f} m) "
                    f"resting on {label} face on {self.reference}. Yaw free."
                ),
                variant=label,
                Tw_e=self._tw_e(_rotation_to_align(n, _neg_z), cylinder_height / 2.0),
                Bw=self._bw(),
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
        """
        for name, val in [("lx", lx), ("ly", ly), ("lz", lz)]:
            if val <= 0:
                raise ValueError(f"{name} must be positive")

        _neg_z = np.array([0.0, 0.0, -1.0])
        # (outward normal of the resting face, COM height, variant label)
        # Each face is listed with its outward normal pointing toward the table.
        candidates = [
            (np.array([ 0.0,  0.0, -1.0]), lz / 2.0, "-z"),
            (np.array([ 0.0,  0.0, +1.0]), lz / 2.0, "+z"),
            (np.array([ 0.0, -1.0,  0.0]), ly / 2.0, "-y"),
            (np.array([ 0.0, +1.0,  0.0]), ly / 2.0, "+y"),
            (np.array([-1.0,  0.0,  0.0]), lx / 2.0, "-x"),
            (np.array([+1.0,  0.0,  0.0]), lx / 2.0, "+x"),
        ]

        return [
            self._template(
                name=f"Place box {label}-face down ({subject} on {self.reference})",
                description=(
                    f"Box ({lx:.3f}×{ly:.3f}×{lz:.3f} m) resting on {label} face "
                    f"on {self.reference}. Yaw free."
                ),
                variant=label,
                Tw_e=self._tw_e(_rotation_to_align(n, _neg_z), com_h),
                Bw=self._bw(),
                subject=subject,
            )
            for n, com_h, label in candidates
        ]

    def place_sphere(
        self,
        radius: float,
        subject: str = "object",
    ) -> List[TSRTemplate]:
        """Return one placement template: sphere on a flat surface.

        Every orientation is equally stable so roll and pitch are also free.

        Args:
            radius: Sphere radius (m).
            subject: Name of the object frame.
        """
        if radius <= 0:
            raise ValueError("radius must be positive")

        return [self._template(
            name=f"Place sphere ({subject} on {self.reference})",
            description=(
                f"Sphere (r={radius:.3f} m) on {self.reference}. "
                f"All orientations free."
            ),
            variant="upright",
            Tw_e=self._tw_e(np.eye(3), float(radius)),
            Bw=self._bw(
                roll_range=np.array([-np.pi, np.pi]),
                pitch_range=np.array([-np.pi, np.pi]),
            ),
            subject=subject,
        )]

    def place_torus(
        self,
        major_radius: float,
        minor_radius: float,
        subject: str = "object",
    ) -> List[TSRTemplate]:
        """Return one placement template: torus flat on surface (axis = z).

        Object frame: origin at center, z = torus symmetry axis pointing up.
        The torus rests on the bottom of the tube ring at z = -minor_radius.

        Args:
            major_radius: Distance from torus center to tube center (m).
            minor_radius: Tube radius (m).
            subject: Name of the object frame.
        """
        if major_radius <= 0:
            raise ValueError("major_radius must be positive")
        if minor_radius <= 0:
            raise ValueError("minor_radius must be positive")
        if minor_radius >= major_radius:
            raise ValueError("minor_radius must be less than major_radius")

        _neg_z = np.array([0.0, 0.0, -1.0])
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
                Tw_e=self._tw_e(_rotation_to_align(n, _neg_z), float(minor_radius)),
                Bw=self._bw(),
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
        """
        vertices = np.asarray(vertices, dtype=float)
        com = np.asarray(com, dtype=float)
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError("vertices must have shape (N, 3)")
        if com.shape != (3,):
            raise ValueError("com must be a length-3 array")

        min_margin_rad = float(np.radians(min_margin_deg))
        poses = sorted(stable_poses_mesh(vertices, com),
                       key=lambda x: -x[2])  # descending stability margin
        poses = [p for p in poses if p[2] >= min_margin_rad]

        templates = []
        for idx, (R, com_height, margin_rad) in enumerate(poses):
            deg = float(np.degrees(margin_rad))
            templates.append(self._template(
                name=(
                    f"Place mesh face {idx + 1}/{len(poses)} "
                    f"({subject} on {self.reference}, margin {deg:.1f}°)"
                ),
                description=(
                    f"Mesh resting on stable face {idx + 1} of {len(poses)} "
                    f"(stability margin {deg:.1f}°) on {self.reference}."
                ),
                variant=f"face-{idx + 1}",
                Tw_e=self._tw_e(R, com_height),
                Bw=self._bw(),
                subject=subject,
                stability_margin=float(margin_rad),
            ))
        return templates
