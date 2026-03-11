"""TablePlacer: generate stable placement TSRs for objects on a flat surface."""
from __future__ import annotations

from typing import List

import numpy as np

from ..template import TSRTemplate
from ._stable_poses import _rotation_to_align, stable_poses_mesh


class TablePlacer:
    """Generate stable placement TSRs for objects placed on a flat table.

    Frame convention:
        Table z points up; table origin at the surface center.
        Object frame origin at geometric center (= COM for uniform density).

    Args:
        table_x: Table half-extent along x (m). Sampled poses slide ±table_x.
        table_y: Table half-extent along y (m). Sampled poses slide ±table_y.
        reference: Reference frame name (default ``"table"``).

    Example::

        placer    = TablePlacer(table_x=0.3, table_y=0.2)
        templates = placer.place_cylinder(cylinder_radius=0.04,
                                          cylinder_height=0.12,
                                          subject="mug")
        tsr  = templates[0].instantiate(table_pose)
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
        """Build 6×2 Bw: table extents for xy, z/roll/pitch fixed, yaw free."""
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
        """Build 4×4 Tw_e from rotation R and COM height above table surface."""
        T = np.eye(4)
        T[:3, :3] = R
        T[2, 3] = float(com_height)
        return T

    def _template(self, name, description, variant, Tw_e, Bw, subject) -> TSRTemplate:
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
        """Return one placement template: cylinder upright on its base.

        Object frame: origin at center, z = cylinder axis pointing up.

        Args:
            cylinder_radius: Cylinder radius (m).
            cylinder_height: Cylinder height (m).
            subject: Name of the object frame.
        """
        if cylinder_radius <= 0:
            raise ValueError("cylinder_radius must be positive")
        if cylinder_height <= 0:
            raise ValueError("cylinder_height must be positive")

        return [self._template(
            name=f"Place cylinder upright ({subject} on {self.reference})",
            description=(
                f"Cylinder (r={cylinder_radius:.3f} m, h={cylinder_height:.3f} m) "
                f"upright on {self.reference}. Yaw free."
            ),
            variant="upright",
            Tw_e=self._tw_e(np.eye(3), cylinder_height / 2.0),
            Bw=self._bw(),
            subject=subject,
        )]

    def place_box(
        self,
        lx: float,
        ly: float,
        lz: float,
        subject: str = "object",
    ) -> List[TSRTemplate]:
        """Return up to 3 placement templates: one per unique stable face.

        Object frame: origin at center, axes aligned with box extents.
        Faces with equal half-extents yield identical templates and are
        deduplicated.

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
        # (outward normal of resting face, COM height, label)
        candidates = [
            (np.array([0.0, 0.0, -1.0]), lz / 2.0, "z-face"),
            (np.array([0.0, -1.0, 0.0]), ly / 2.0, "y-face"),
            (np.array([-1.0, 0.0, 0.0]), lx / 2.0, "x-face"),
        ]

        seen: set = set()
        templates: List[TSRTemplate] = []
        for n, com_h, label in candidates:
            key = round(com_h, 10)
            if key in seen:
                continue
            seen.add(key)
            templates.append(self._template(
                name=f"Place box on {label} ({subject} on {self.reference})",
                description=(
                    f"Box ({lx:.3f}×{ly:.3f}×{lz:.3f} m) resting on {label} "
                    f"on {self.reference}. Yaw free."
                ),
                variant=label,
                Tw_e=self._tw_e(_rotation_to_align(n, _neg_z), com_h),
                Bw=self._bw(),
                subject=subject,
            ))
        return templates

    def place_sphere(
        self,
        radius: float,
        subject: str = "object",
    ) -> List[TSRTemplate]:
        """Return one placement template: sphere on table.

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
        """Return one placement template: torus flat on table (axis = z).

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

        return [self._template(
            name=f"Place torus flat ({subject} on {self.reference})",
            description=(
                f"Torus (R={major_radius:.3f} m, r={minor_radius:.3f} m) "
                f"flat on {self.reference}. Yaw free."
            ),
            variant="flat",
            Tw_e=self._tw_e(np.eye(3), float(minor_radius)),
            Bw=self._bw(),
            subject=subject,
        )]

    def place_mesh(
        self,
        vertices: np.ndarray,
        com: np.ndarray,
        subject: str = "object",
    ) -> List[TSRTemplate]:
        """Return one template per stable resting face of the mesh.

        Uses the convex hull of ``vertices`` for stable-pose detection.
        Works for non-convex objects: the support polygon is the convex
        hull of the contact points.  Results are sorted by descending
        stability margin (most stable face first).

        Args:
            vertices: (N, 3) array of object vertices in the object frame.
            com:      (3,) center of mass in the same frame.
            subject:  Name of the object frame.
        """
        vertices = np.asarray(vertices, dtype=float)
        com = np.asarray(com, dtype=float)
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError("vertices must have shape (N, 3)")
        if com.shape != (3,):
            raise ValueError("com must be a length-3 array")

        poses = sorted(stable_poses_mesh(vertices, com),
                       key=lambda x: -x[2])  # descending stability margin

        templates = []
        for idx, (R, com_height, stability_margin) in enumerate(poses):
            deg = float(np.degrees(stability_margin))
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
            ))
        return templates
