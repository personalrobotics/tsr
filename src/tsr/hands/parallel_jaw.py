"""Parallel jaw gripper hand models."""
from __future__ import annotations

import dataclasses
from typing import List, Optional, Tuple

import numpy as np

from tsr.template import TSRTemplate
from .base import GripperBase

_DEPTH_LABELS = {1: ["mid"], 2: ["shallow", "deep"], 3: ["shallow", "mid", "deep"]}


def _depth_label(k: int, i: int) -> str:
    return (_DEPTH_LABELS.get(k) or [f"depth {j+1}/{k}" for j in range(k)])[i]


class ParallelJawGripper(GripperBase):
    """Parallel jaw gripper: generates TSRTemplates from object geometry.

    Poses sampled from these TSRs are pre-grasp configurations — the hand is
    open at ``preshape``, positioned so that closing the fingers achieves
    stable contact with the object. The TSR constrains where the hand must be
    before closing; it does not explicitly verify force closure.

    Frame convention:
        z = approach direction (toward object surface)
        y = finger opening direction
        x = palm normal (right-hand rule: x = y × z)

    AnyGrasp / GraspNet uses x=approach — convert with::

        R = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])

    Args:
        finger_length: Distance from palm to fingertip [m].
        max_aperture:  Maximum jaw opening [m].
    """

    def __init__(self, finger_length: float, max_aperture: float):
        self.finger_length = finger_length
        self.max_aperture  = max_aperture

    def _validate(self, cylinder_radius: float, preshape: float) -> None:
        if cylinder_radius <= 0:
            raise ValueError("cylinder_radius must be > 0")
        if preshape > self.max_aperture:
            raise ValueError(
                f"preshape {preshape:.3f}m > max_aperture {self.max_aperture:.3f}m"
            )

    def grasp_cylinder_side(
        self,
        cylinder_radius: float,
        cylinder_height: float,
        preshape: Optional[float] = None,
        k: int = 3,
        clearance: Optional[float] = None,
        angle_range: Tuple[float, float] = (0., 2 * np.pi),
        subject: str = "gripper",
        reference: str = "cylinder",
        name: str = "",
        description: str = "",
    ) -> List[TSRTemplate]:
        """Side grasp templates for a cylinder — 2*k templates.

        Returns k depth levels × 2 roll orientations. Each template has a fixed
        radial offset baked into Tw_e, covering the full pre-grasp volume:

          depth 1/k : fingertips a clearance inside the surface (shallowest)
          depth k/k : palm a clearance from the surface (deepest)

        Two roll orientations per depth (180° apart around z_EE):
          roll=0 : palm normal = +y_world, fingers open in +tangential
          roll=π : palm normal = -y_world, fingers open in -tangential
        A symmetric hand produces identical poses; asymmetric hands need both.

        The radial approach direction couples with yaw and cannot be encoded in
        Bw directly. Instead, k discrete depths are baked into Tw_e so the full
        pre-grasp volume is covered without post-processing.

        Args:
            clearance:  Safety buffer [m] applied to: height ends, fingertip
                        start depth, and palm stop depth. Default: 10% of finger_length.

        Returns [] if preshape <= cylinder diameter. Raises ValueError for
        invalid geometry.
        """
        if clearance is None:
            clearance = 0.1 * self.finger_length
        if preshape is None:
            preshape = 2. * cylinder_radius + clearance
        self._validate(cylinder_radius, preshape)
        if preshape <= 2. * cylinder_radius:
            return []
        h0, h1 = clearance, cylinder_height - clearance
        if h1 <= h0:
            raise ValueError("cylinder_height too small for the given clearance")

        if not name:
            name = f"{reference.title()} Cylinder Side Grasp"

        z_mid, z_half = (h0 + h1) / 2., (h1 - h0) / 2.
        T_ref_tsr = np.eye(4)
        T_ref_tsr[2, 3] = z_mid

        Bw = np.array([
            [0.,             0.            ],  # x: no radial freedom
            [0.,             0.            ],  # y: no tangential freedom
            [-z_half,        z_half        ],  # z: height range (symmetric)
            [0.,             0.            ],  # roll: fixed (encoded in Tw_e)
            [0.,             0.            ],  # pitch
            [angle_range[0], angle_range[1]],  # yaw: angular freedom
        ])

        approach_max = min(self.finger_length, cylinder_radius) - clearance
        depths = np.linspace(clearance, approach_max, max(k, 1))

        common = dict(
            T_ref_tsr=T_ref_tsr, Bw=Bw,
            task="grasp", subject=subject, reference=reference,
            preshape=np.array([preshape]),
        )
        templates = []
        for i, d in enumerate(depths):
            ro = cylinder_radius + self.finger_length - d
            dlabel = _depth_label(k, i)
            Tw_e_0 = np.array([
                [ 0.,  0., -1., ro],
                [ 0.,  1.,  0., 0.],
                [ 1.,  0.,  0., 0.],
                [ 0.,  0.,  0., 1.],
            ])
            Tw_e_pi = np.array([
                [ 0.,  0., -1., ro],
                [ 0., -1.,  0., 0.],
                [-1.,  0.,  0., 0.],
                [ 0.,  0.,  0., 1.],
            ])
            for Tw_e, roll_label in ((Tw_e_0, "roll 0°"), (Tw_e_pi, "roll 180°")):
                t_desc = description or (
                    f"{dlabel.capitalize()} side grasp on {reference}: "
                    f"standoff {ro*1000:.0f}mm from axis, {roll_label}, "
                    f"preshape {preshape*1000:.0f}mm"
                )
                templates.append(TSRTemplate(
                    Tw_e=Tw_e,
                    name=f"{name} — {dlabel}, {roll_label}",
                    description=t_desc,
                    **common,
                ))
        return templates

    def grasp_cylinder_top(
        self,
        cylinder_radius: float,
        cylinder_height: float,
        preshape: Optional[float] = None,
        k: int = 3,
        clearance: Optional[float] = None,
        angle_range: Tuple[float, float] = (0., 2 * np.pi),
        subject: str = "gripper",
        reference: str = "cylinder",
        name: str = "",
        description: str = "",
    ) -> List[TSRTemplate]:
        """Top-down grasp templates for a cylinder — k templates.

        Gripper approaches from above (z_EE = [0,0,-1]). TSR origin at
        z = cylinder_height. Depth ranges from fingertips a clearance inside
        the rim (shallowest) to palm a clearance above the rim (deepest).
        Full yaw covers all finger orientations.
        """
        if clearance is None:
            clearance = 0.1 * self.finger_length
        if preshape is None:
            preshape = 2. * cylinder_radius + clearance
        self._validate(cylinder_radius, preshape)
        if preshape <= 2. * cylinder_radius:
            return []

        if not name:
            name = f"{reference.title()} Cylinder Top Grasp"

        T_ref_tsr = np.eye(4)
        T_ref_tsr[2, 3] = cylinder_height

        Bw = np.array([
            [0.,             0.            ],  # x
            [0.,             0.            ],  # y
            [0.,             0.            ],  # z: fixed at top face
            [0.,             0.            ],  # roll
            [0.,             0.            ],  # pitch
            [angle_range[0], angle_range[1]],  # yaw: full rotation
        ])

        depths = np.linspace(clearance, self.finger_length - clearance, max(k, 1))
        common = dict(
            T_ref_tsr=T_ref_tsr, Bw=Bw,
            task="grasp", subject=subject, reference=reference,
            preshape=np.array([preshape]),
        )
        templates = []
        for i, d in enumerate(depths):
            h_palm = self.finger_length - d
            dlabel = _depth_label(k, i)
            t_desc = description or (
                f"{dlabel.capitalize()} top grasp on {reference}: "
                f"palm {h_palm*1000:.0f}mm above rim, preshape {preshape*1000:.0f}mm"
            )
            # z_EE = [0,0,-1] (approach down); x = y × z = [0,1,0]×[0,0,-1] = [-1,0,0]
            Tw_e = np.array([
                [-1.,  0.,  0.,  0.     ],
                [ 0.,  1.,  0.,  0.     ],
                [ 0.,  0., -1.,  h_palm ],
                [ 0.,  0.,  0.,  1.     ],
            ])
            templates.append(TSRTemplate(
                Tw_e=Tw_e,
                name=f"{name} — {dlabel}",
                description=t_desc,
                **common,
            ))
        return templates

    def grasp_cylinder_bottom(
        self,
        cylinder_radius: float,
        cylinder_height: float,
        preshape: Optional[float] = None,
        k: int = 3,
        clearance: Optional[float] = None,
        angle_range: Tuple[float, float] = (0., 2 * np.pi),
        subject: str = "gripper",
        reference: str = "cylinder",
        name: str = "",
        description: str = "",
    ) -> List[TSRTemplate]:
        """Bottom-up grasp templates for a cylinder — k templates.

        Gripper approaches from below (z_EE = [0,0,+1]). TSR origin at z = 0
        (bottom face). Depth ranges from fingertips a clearance inside the
        bottom face (shallowest) to palm a clearance below it (deepest).
        Full yaw covers all finger orientations.
        """
        if clearance is None:
            clearance = 0.1 * self.finger_length
        if preshape is None:
            preshape = 2. * cylinder_radius + clearance
        self._validate(cylinder_radius, preshape)
        if preshape <= 2. * cylinder_radius:
            return []

        if not name:
            name = f"{reference.title()} Cylinder Bottom Grasp"

        del cylinder_height  # bottom face is always at z=0; accepted for interface symmetry
        T_ref_tsr = np.eye(4)

        Bw = np.array([
            [0.,             0.            ],  # x
            [0.,             0.            ],  # y
            [0.,             0.            ],  # z: fixed at bottom face
            [0.,             0.            ],  # roll
            [0.,             0.            ],  # pitch
            [angle_range[0], angle_range[1]],  # yaw: full rotation
        ])

        depths = np.linspace(clearance, self.finger_length - clearance, max(k, 1))
        common = dict(
            T_ref_tsr=T_ref_tsr, Bw=Bw,
            task="grasp", subject=subject, reference=reference,
            preshape=np.array([preshape]),
        )
        templates = []
        for i, d in enumerate(depths):
            h_palm = self.finger_length - d
            dlabel = _depth_label(k, i)
            t_desc = description or (
                f"{dlabel.capitalize()} bottom grasp on {reference}: "
                f"palm {h_palm*1000:.0f}mm below bottom, preshape {preshape*1000:.0f}mm"
            )
            # z_EE = [0,0,+1] (approach up); identity rotation
            Tw_e = np.array([
                [ 1.,  0.,  0.,  0.     ],
                [ 0.,  1.,  0.,  0.     ],
                [ 0.,  0.,  1., -h_palm ],
                [ 0.,  0.,  0.,  1.     ],
            ])
            templates.append(TSRTemplate(
                Tw_e=Tw_e,
                name=f"{name} — {dlabel}",
                description=t_desc,
                **common,
            ))
        return templates

    # ── Box primitives ────────────────────────────────────────────────────────

    def _validate_box(self, box_x: float, box_y: float, box_z: float) -> None:
        for dim, label in ((box_x, "box_x"), (box_y, "box_y"), (box_z, "box_z")):
            if dim <= 0:
                raise ValueError(f"{label} must be > 0")

    def _box_face_templates(
        self,
        T_ref_tsr: np.ndarray,
        y_ee: np.ndarray,
        z_ee: np.ndarray,
        span_dim: float,
        slide_bw_row: int,
        slide_half: float,
        preshape_user: Optional[float],
        k: int,
        clearance: float,
        subject: str,
        reference: str,
        name_prefix: str,
        description: str,
        face_label: str,
    ) -> List[TSRTemplate]:
        """k depth templates for one face × finger-orientation combo.

        Fingers open along y_ee, spanning span_dim.  The gripper slides in
        slide_bw_row (world axis 0/1/2 = x/y/z) ± slide_half from the TSR
        origin.  Returns [] if the finger span can't fit around the object or
        the required preshape exceeds max_aperture.
        """
        if preshape_user is not None:
            preshape = preshape_user
            if preshape <= span_dim:
                return []           # fingers can't straddle the object
        else:
            preshape = span_dim + clearance
            if preshape > self.max_aperture:
                return []           # geometry exceeds hardware — skip silently

        Bw = np.zeros((6, 2))
        Bw[slide_bw_row, 0] = -slide_half
        Bw[slide_bw_row, 1] =  slide_half

        R = np.column_stack([np.cross(y_ee, z_ee), y_ee, z_ee])

        depths = np.linspace(clearance, self.finger_length - clearance, max(k, 1))
        common = dict(
            T_ref_tsr=T_ref_tsr, Bw=Bw,
            task="grasp", subject=subject, reference=reference,
            preshape=np.array([preshape]),
        )
        templates = []
        for i, d in enumerate(depths):
            h_palm = self.finger_length - d
            dlabel = _depth_label(k, i)
            Tw_e = np.eye(4)
            Tw_e[:3, :3] = R
            Tw_e[:3, 3]  = -z_ee * h_palm   # palm is h_palm outside the face
            t_desc = description or (
                f"{dlabel.capitalize()} {face_label} grasp on {reference}: "
                f"standoff {h_palm*1000:.0f}mm, preshape {preshape*1000:.0f}mm"
            )
            templates.append(TSRTemplate(
                Tw_e=Tw_e,
                name=f"{name_prefix} {face_label} — {dlabel}",
                description=t_desc,
                **common,
            ))
        return templates

    def grasp_box_top(
        self,
        box_x: float,
        box_y: float,
        box_z: float,
        preshape: Optional[float] = None,
        k: int = 3,
        clearance: Optional[float] = None,
        subject: str = "gripper",
        reference: str = "box",
        name: str = "",
        description: str = "",
    ) -> List[TSRTemplate]:
        """Top-down grasp templates for a box — up to 2*k templates.

        Gripper approaches from above (z_EE = [0,0,-1]). TSR origin at
        z = box_z (top face). Two finger orientations:
          - fingers along x: spans box_x, slides in y
          - fingers along y: spans box_y, slides in x
        Each orientation generates k depth templates if max_aperture allows.
        """
        if clearance is None:
            clearance = 0.1 * self.finger_length
        self._validate_box(box_x, box_y, box_z)
        if preshape is not None and preshape > self.max_aperture:
            raise ValueError(
                f"preshape {preshape:.3f}m > max_aperture {self.max_aperture:.3f}m"
            )
        hx, hy = box_x / 2. - clearance, box_y / 2. - clearance
        if hx <= 0 or hy <= 0:
            raise ValueError("box face too small for the given clearance")

        if not name:
            name = f"{reference.title()} Box Top Grasp"

        T = np.eye(4)
        T[2, 3] = box_z
        z_ee = np.array([0., 0., -1.])

        kw = dict(preshape_user=preshape, k=k, clearance=clearance,
                  subject=subject, reference=reference,
                  name_prefix=name, description=description)
        return (
            self._box_face_templates(T, np.array([1., 0., 0.]), z_ee,
                                     box_x, 1, hy, **kw,
                                     face_label="+z (span-x)")
            + self._box_face_templates(T, np.array([0., 1., 0.]), z_ee,
                                       box_y, 0, hx, **kw,
                                       face_label="+z (span-y)")
        )

    def grasp_box_bottom(
        self,
        box_x: float,
        box_y: float,
        box_z: float,
        preshape: Optional[float] = None,
        k: int = 3,
        clearance: Optional[float] = None,
        subject: str = "gripper",
        reference: str = "box",
        name: str = "",
        description: str = "",
    ) -> List[TSRTemplate]:
        """Bottom-up grasp templates for a box — up to 2*k templates.

        Gripper approaches from below (z_EE = [0,0,+1]). TSR origin at
        z = 0 (bottom face). Two finger orientations:
          - fingers along x: spans box_x, slides in y
          - fingers along y: spans box_y, slides in x
        Each orientation generates k depth templates if max_aperture allows.
        """
        if clearance is None:
            clearance = 0.1 * self.finger_length
        self._validate_box(box_x, box_y, box_z)
        if preshape is not None and preshape > self.max_aperture:
            raise ValueError(
                f"preshape {preshape:.3f}m > max_aperture {self.max_aperture:.3f}m"
            )
        hx, hy = box_x / 2. - clearance, box_y / 2. - clearance
        if hx <= 0 or hy <= 0:
            raise ValueError("box face too small for the given clearance")

        if not name:
            name = f"{reference.title()} Box Bottom Grasp"

        del box_z  # bottom face is always at z=0; accepted for API symmetry
        T = np.eye(4)
        z_ee = np.array([0., 0., 1.])

        kw = dict(preshape_user=preshape, k=k, clearance=clearance,
                  subject=subject, reference=reference,
                  name_prefix=name, description=description)
        return (
            self._box_face_templates(T, np.array([1., 0., 0.]), z_ee,
                                     box_x, 1, hy, **kw,
                                     face_label="-z (span-x)")
            + self._box_face_templates(T, np.array([0., 1., 0.]), z_ee,
                                       box_y, 0, hx, **kw,
                                       face_label="-z (span-y)")
        )

    def grasp_box_face_x(
        self,
        box_x: float,
        box_y: float,
        box_z: float,
        preshape: Optional[float] = None,
        k: int = 3,
        clearance: Optional[float] = None,
        subject: str = "gripper",
        reference: str = "box",
        name: str = "",
        description: str = "",
    ) -> List[TSRTemplate]:
        """Grasp templates for the ±x faces of a box — up to 4*k templates.

        Two approach directions (+x and -x), each with two finger orientations:
          - fingers along y: spans box_y, slides in z
          - fingers along z: spans box_z, slides in y
        Each valid orientation (max_aperture allows the span) generates k templates.
        """
        if clearance is None:
            clearance = 0.1 * self.finger_length
        self._validate_box(box_x, box_y, box_z)
        if preshape is not None and preshape > self.max_aperture:
            raise ValueError(
                f"preshape {preshape:.3f}m > max_aperture {self.max_aperture:.3f}m"
            )
        hy = box_y / 2. - clearance
        hz_half = box_z / 2. - clearance
        if hy <= 0:
            raise ValueError("box_y too small for the given clearance")
        if hz_half <= 0:
            raise ValueError("box_z too small for the given clearance")

        if not name:
            name = f"{reference.title()} Box X-Face Grasp"

        T_pos = np.eye(4); T_pos[0, 3] =  box_x / 2.; T_pos[2, 3] = box_z / 2.
        T_neg = np.eye(4); T_neg[0, 3] = -box_x / 2.; T_neg[2, 3] = box_z / 2.

        kw = dict(preshape_user=preshape, k=k, clearance=clearance,
                  subject=subject, reference=reference,
                  name_prefix=name, description=description)
        templates = []
        for T_ref, z_ee, sign in (
            (T_pos, np.array([-1., 0., 0.]), "+x"),
            (T_neg, np.array([ 1., 0., 0.]), "-x"),
        ):
            templates += self._box_face_templates(
                T_ref, np.array([0., 1., 0.]), z_ee,
                box_y, 2, hz_half, **kw, face_label=f"{sign} (span-y)")
            templates += self._box_face_templates(
                T_ref, np.array([0., 0., 1.]), z_ee,
                box_z, 1, hy,      **kw, face_label=f"{sign} (span-z)")
        return templates

    def grasp_box_face_y(
        self,
        box_x: float,
        box_y: float,
        box_z: float,
        preshape: Optional[float] = None,
        k: int = 3,
        clearance: Optional[float] = None,
        subject: str = "gripper",
        reference: str = "box",
        name: str = "",
        description: str = "",
    ) -> List[TSRTemplate]:
        """Grasp templates for the ±y faces of a box — up to 4*k templates.

        Two approach directions (+y and -y), each with two finger orientations:
          - fingers along x: spans box_x, slides in z
          - fingers along z: spans box_z, slides in x
        Each valid orientation (max_aperture allows the span) generates k templates.
        """
        if clearance is None:
            clearance = 0.1 * self.finger_length
        self._validate_box(box_x, box_y, box_z)
        if preshape is not None and preshape > self.max_aperture:
            raise ValueError(
                f"preshape {preshape:.3f}m > max_aperture {self.max_aperture:.3f}m"
            )
        hx = box_x / 2. - clearance
        hz_half = box_z / 2. - clearance
        if hx <= 0:
            raise ValueError("box_x too small for the given clearance")
        if hz_half <= 0:
            raise ValueError("box_z too small for the given clearance")

        if not name:
            name = f"{reference.title()} Box Y-Face Grasp"

        T_pos = np.eye(4); T_pos[1, 3] =  box_y / 2.; T_pos[2, 3] = box_z / 2.
        T_neg = np.eye(4); T_neg[1, 3] = -box_y / 2.; T_neg[2, 3] = box_z / 2.

        kw = dict(preshape_user=preshape, k=k, clearance=clearance,
                  subject=subject, reference=reference,
                  name_prefix=name, description=description)
        templates = []
        for T_ref, z_ee, sign in (
            (T_pos, np.array([0., -1., 0.]), "+y"),
            (T_neg, np.array([0.,  1., 0.]), "-y"),
        ):
            templates += self._box_face_templates(
                T_ref, np.array([1., 0., 0.]), z_ee,
                box_x, 2, hz_half, **kw, face_label=f"{sign} (span-x)")
            templates += self._box_face_templates(
                T_ref, np.array([0., 0., 1.]), z_ee,
                box_z, 0, hx,      **kw, face_label=f"{sign} (span-z)")
        return templates

    # ── Sphere primitives ─────────────────────────────────────────────────────

    def grasp_sphere(
        self,
        object_radius: float,
        preshape: Optional[float] = None,
        k: int = 3,
        clearance: Optional[float] = None,
        angle_range: Tuple[float, float] = (0., 2 * np.pi),
        subject: str = "gripper",
        reference: str = "sphere",
        name: str = "",
        description: str = "",
    ) -> List[TSRTemplate]:
        """Full-sphere grasp templates — k templates.

        Approach from any direction in 3-D. Each template has continuous SO(3)
        freedom in Bw so sampling produces uniformly distributed approach
        directions over the sphere:

          roll  ∈ [0, 2π]        — finger orientation around the approach axis
          pitch ∈ [-π/2, π/2]   — elevation (covers full hemisphere, no double-cover)
          yaw   ∈ angle_range    — azimuth (default full 360°)

        TSR origin at sphere center. k discrete depths baked into Tw_e.

        Returns [] if preshape <= sphere diameter. Raises ValueError for
        invalid geometry.
        """
        if clearance is None:
            clearance = 0.1 * self.finger_length
        if preshape is None:
            preshape = 2. * object_radius + clearance
        if object_radius <= 0:
            raise ValueError("object_radius must be > 0")
        if preshape > self.max_aperture:
            raise ValueError(
                f"preshape {preshape:.3f}m > max_aperture {self.max_aperture:.3f}m"
            )
        if preshape <= 2. * object_radius:
            return []

        if not name:
            name = f"{reference.title()} Sphere Grasp"

        T_ref_tsr = np.eye(4)   # origin at sphere center

        Bw = np.array([
            [0.,              0.           ],  # x: no translational freedom
            [0.,              0.           ],  # y
            [0.,              0.           ],  # z
            [0.,              2 * np.pi    ],  # roll: full rotation around approach
            [-np.pi / 2,      np.pi / 2    ],  # pitch: full elevation (no double-cover)
            [angle_range[0],  angle_range[1]],  # yaw: azimuthal freedom
        ])

        approach_max = min(self.finger_length, object_radius) - clearance
        depths = np.linspace(clearance, approach_max, max(k, 1))

        common = dict(
            T_ref_tsr=T_ref_tsr, Bw=Bw,
            task="grasp", subject=subject, reference=reference,
            preshape=np.array([preshape]),
        )
        templates = []
        for i, d in enumerate(depths):
            ro = object_radius + self.finger_length - d
            dlabel = _depth_label(k, i)
            # Approach along -x in TSR frame; standoff ro baked into Tw_e.
            # Bw roll/pitch/yaw rotates this to any direction on the sphere.
            Tw_e = np.array([
                [ 0.,  0., -1., ro],
                [ 0.,  1.,  0., 0.],
                [ 1.,  0.,  0., 0.],
                [ 0.,  0.,  0., 1.],
            ])
            t_desc = description or (
                f"{dlabel.capitalize()} sphere grasp on {reference}: "
                f"standoff {ro*1000:.0f}mm from center, full SO(3), "
                f"preshape {preshape*1000:.0f}mm"
            )
            templates.append(TSRTemplate(
                Tw_e=Tw_e,
                name=f"{name} — {dlabel}",
                description=t_desc,
                **common,
            ))
        return templates

    # ── Torus primitives ─────────────────────────────────────────────────────

    def _validate_torus(self, torus_radius: float, tube_radius: float) -> None:
        if torus_radius <= 0:
            raise ValueError("torus_radius must be > 0")
        if tube_radius <= 0:
            raise ValueError("tube_radius must be > 0")
        if tube_radius >= torus_radius:
            raise ValueError(
                f"tube_radius ({tube_radius}) must be < torus_radius ({torus_radius}) "
                "to avoid a self-intersecting torus"
            )

    def grasp_torus_side(
        self,
        torus_radius: float,
        tube_radius: float,
        preshape: Optional[float] = None,
        k: int = 3,
        n_minor: int = 5,
        clearance: Optional[float] = None,
        angle_range: Tuple[float, float] = (0., 2 * np.pi),
        subject: str = "gripper",
        reference: str = "torus",
        name: str = "",
        description: str = "",
    ) -> List[TSRTemplate]:
        """Side grasp templates for a torus tube — 2 * k * n_minor templates.

        The gripper approaches the tube from n_minor discrete angles α in the
        tube cross-section plane (the plane containing the radial and vertical
        axes), ranging from directly below (α = −π/2) to directly above
        (α = +π/2) via the outside equator (α = 0):

            α = −π/2  from below    (matches span-bottom geometry)
            α = −π/4  from below-outside
            α =  0    from outside  (pure radial equatorial approach)
            α = +π/4  from above-outside
            α = +π/2  from above    (matches span-top geometry)

        At each α the Bw yaw samples the full azimuth around the torus ring
        (major radius), giving complete coverage of the tube surface. Two hand
        flip variants (fingers open in ±tangential direction) are generated per
        (α, depth) combination.

        Gripper position in TSR frame at depth d, minor angle α:
            tx = R + (r + fl − d) · cos α   (radial offset from torus axis)
            tz =     (r + fl − d) · sin α   (height above equatorial plane)
        z_EE = (−cos α, 0, −sin α)   (points toward tube center)

        Args:
            n_minor:   Discrete approach angles around the tube cross-section
                       (default 5: evenly spaced in [−π/2, +π/2]).
            clearance: Safety buffer [m]. Defaults to 10% of finger_length.

        Returns 2*k*n_minor TSRTemplates. Returns [] if preshape ≤ tube
        diameter. Raises ValueError for invalid geometry.
        """
        if clearance is None:
            clearance = 0.1 * self.finger_length
        if preshape is None:
            preshape = 2. * tube_radius + clearance
        self._validate_torus(torus_radius, tube_radius)
        if preshape > self.max_aperture:
            raise ValueError(
                f"preshape {preshape:.3f}m > max_aperture {self.max_aperture:.3f}m"
            )
        if preshape <= 2. * tube_radius:
            return []

        if not name:
            name = f"{reference.title()} Torus Side Grasp"

        T_ref_tsr = np.eye(4)  # origin at torus center

        Bw = np.array([
            [0.,             0.            ],  # x: no freedom
            [0.,             0.            ],  # y: no freedom
            [0.,             0.            ],  # z: baked into Tw_e
            [0.,             0.            ],  # roll: baked into Tw_e
            [0.,             0.            ],  # pitch: baked into Tw_e
            [angle_range[0], angle_range[1]],  # yaw: full azimuthal freedom
        ])

        approach_max = min(self.finger_length, tube_radius) - clearance
        depths = np.linspace(clearance, approach_max, max(k, 1))
        minor_angles = np.linspace(-np.pi / 2, np.pi / 2, max(n_minor, 1))

        common = dict(
            T_ref_tsr=T_ref_tsr, Bw=Bw,
            task="grasp", subject=subject, reference=reference,
            preshape=np.array([preshape]),
        )
        templates = []
        for alpha in minor_angles:
            ca, sa = np.cos(alpha), np.sin(alpha)
            a_label = f"α={np.degrees(alpha):.0f}°"
            for i, d in enumerate(depths):
                # Distance from tube center to palm
                ro_minor = tube_radius + self.finger_length - d
                # Gripper position in TSR frame (before yaw rotation)
                tx = torus_radius + ro_minor * ca
                tz = ro_minor * sa
                dlabel = _depth_label(k, i)
                # z_EE = (−cosα, 0, −sinα); y_EE = (0, ±1, 0); x_EE = y_EE × z_EE
                # Flip 0:  y_EE=(0,+1,0) → x_EE=(−sinα, 0, cosα)
                # Flip π:  y_EE=(0,−1,0) → x_EE=(+sinα, 0, −cosα)
                Tw_e_0 = np.array([
                    [-sa,  0., -ca,  tx],
                    [ 0.,  1.,  0.,  0.],
                    [ ca,  0., -sa,  tz],
                    [ 0.,  0.,  0.,  1.],
                ])
                Tw_e_pi = np.array([
                    [ sa,  0., -ca,  tx],
                    [ 0., -1.,  0.,  0.],
                    [-ca,  0., -sa,  tz],
                    [ 0.,  0.,  0.,  1.],
                ])
                for Tw_e, flip_label in ((Tw_e_0, "flip 0°"), (Tw_e_pi, "flip 180°")):
                    t_desc = description or (
                        f"{dlabel.capitalize()} torus side grasp on {reference}: "
                        f"{a_label}, ro={ro_minor*1000:.0f}mm from tube center, "
                        f"{flip_label}, preshape {preshape*1000:.0f}mm"
                    )
                    templates.append(TSRTemplate(
                        Tw_e=Tw_e,
                        name=f"{name} — {a_label}, {dlabel}, {flip_label}",
                        description=t_desc,
                        **common,
                    ))
        return templates

    def grasp_torus_span(
        self,
        torus_radius: float,
        tube_radius: float,
        preshape: Optional[float] = None,
        k: int = 3,
        clearance: Optional[float] = None,
        subject: str = "gripper",
        reference: str = "torus",
        name: str = "",
        description: str = "",
    ) -> List[TSRTemplate]:
        """Span grasp templates for a torus — up to 2*k templates.

        Approach from above and below, fingers spanning the outer diameter
        2*(R+r). Full yaw freedom covers all finger orientations. Silently
        returns [] if the outer diameter + clearance exceeds max_aperture.
        """
        if clearance is None:
            clearance = 0.1 * self.finger_length
        if preshape is None:
            preshape = 2. * (torus_radius + tube_radius) + clearance
            if preshape > self.max_aperture:
                return []       # torus too large for hardware — skip silently
        else:
            if preshape <= 2. * (torus_radius + tube_radius):
                return []       # user preshape can't straddle outer diameter
            if preshape > self.max_aperture:
                raise ValueError(
                    f"preshape {preshape:.3f}m > max_aperture {self.max_aperture:.3f}m"
                )
        self._validate_torus(torus_radius, tube_radius)

        if not name:
            name = f"{reference.title()} Torus Span Grasp"

        # Bw: yaw free [0, 2π] for full finger-orientation freedom; all else fixed
        Bw = np.zeros((6, 2))
        Bw[5, 1] = 2 * np.pi

        depths = np.linspace(clearance, self.finger_length - clearance, max(k, 1))
        templates = []
        for i, d in enumerate(depths):
            h_palm = self.finger_length - d
            dlabel = _depth_label(k, i)

            # Top: z_EE = [0,0,-1]; TSR origin at tube top (z = +tube_r)
            T_top = np.eye(4); T_top[2, 3] = tube_radius
            Tw_e_top = np.array([
                [-1.,  0.,  0.,  0.     ],
                [ 0.,  1.,  0.,  0.     ],
                [ 0.,  0., -1.,  h_palm ],
                [ 0.,  0.,  0.,  1.     ],
            ])
            t_desc = description or (
                f"{dlabel.capitalize()} torus span top on {reference}: "
                f"palm {h_palm*1000:.0f}mm above torus, preshape {preshape*1000:.0f}mm"
            )
            templates.append(TSRTemplate(
                T_ref_tsr=T_top, Bw=Bw, Tw_e=Tw_e_top,
                task="grasp", subject=subject, reference=reference,
                preshape=np.array([preshape]),
                name=f"{name} top — {dlabel}",
                description=t_desc,
            ))

            # Bottom: z_EE = [0,0,+1]; TSR origin at tube bottom (z = -tube_r)
            T_bot = np.eye(4); T_bot[2, 3] = -tube_radius
            Tw_e_bot = np.array([
                [ 1.,  0.,  0.,  0.     ],
                [ 0.,  1.,  0.,  0.     ],
                [ 0.,  0.,  1., -h_palm ],
                [ 0.,  0.,  0.,  1.     ],
            ])
            t_desc = description or (
                f"{dlabel.capitalize()} torus span bottom on {reference}: "
                f"palm {h_palm*1000:.0f}mm below torus, preshape {preshape*1000:.0f}mm"
            )
            templates.append(TSRTemplate(
                T_ref_tsr=T_bot, Bw=Bw, Tw_e=Tw_e_bot,
                task="grasp", subject=subject, reference=reference,
                preshape=np.array([preshape]),
                name=f"{name} bottom — {dlabel}",
                description=t_desc,
            ))
        return templates

    def renderer(self):
        """Return a SubjectRenderer using the parallel jaw wireframe.

        Requires the ``viz`` extra (pyvista). Lazy import so pyvista is not
        required just for template generation.
        """
        from tsr.viz import parallel_jaw_renderer
        return parallel_jaw_renderer(
            finger_length=self.finger_length,
            half_aperture=self.max_aperture / 2,
        )


# −90° z-rotation: corrects Robotiq wrist frame to canonical TSR EE frame.
# The Robotiq 2F-140 physical flange zero-pose is rotated −90° around z
# relative to the canonical frame (z=approach, y=opening, x=palm).
_R_z_neg90 = np.array([
    [ 0.,  1.,  0.,  0.],
    [-1.,  0.,  0.,  0.],
    [ 0.,  0.,  1.,  0.],
    [ 0.,  0.,  0.,  1.],
])


class Robotiq2F140(ParallelJawGripper):
    """Robotiq 2F-140 parallel gripper.

    Fixed hardware parameters: finger_length=55 mm, max_aperture=140 mm.

    Applies a −90° z-rotation correction to Tw_e so sampled poses are in
    the Robotiq physical wrist frame rather than the canonical TSR EE frame.

    Usage::

        gripper   = Robotiq2F140()
        templates = gripper.grasp_cylinder(cylinder_radius=0.04,
                                           cylinder_height=0.10,
                                           reference="mug")
        tsr  = templates[0].instantiate(mug_pose)
        pose = tsr.sample()   # pose in Robotiq wrist frame
    """

    FINGER_LENGTH = 0.055
    MAX_APERTURE  = 0.140

    def __init__(self):
        super().__init__(
            finger_length=self.FINGER_LENGTH,
            max_aperture=self.MAX_APERTURE,
        )

    def _apply_frame_correction(self, templates: List[TSRTemplate]) -> List[TSRTemplate]:
        return [dataclasses.replace(t, Tw_e=t.Tw_e @ _R_z_neg90) for t in templates]

    def grasp_cylinder_side(self, *args, **kwargs) -> List[TSRTemplate]:
        return self._apply_frame_correction(super().grasp_cylinder_side(*args, **kwargs))

    def grasp_cylinder_top(self, *args, **kwargs) -> List[TSRTemplate]:
        return self._apply_frame_correction(super().grasp_cylinder_top(*args, **kwargs))

    def grasp_cylinder_bottom(self, *args, **kwargs) -> List[TSRTemplate]:
        return self._apply_frame_correction(super().grasp_cylinder_bottom(*args, **kwargs))

    def grasp_box_top(self, *args, **kwargs) -> List[TSRTemplate]:
        return self._apply_frame_correction(super().grasp_box_top(*args, **kwargs))

    def grasp_box_bottom(self, *args, **kwargs) -> List[TSRTemplate]:
        return self._apply_frame_correction(super().grasp_box_bottom(*args, **kwargs))

    def grasp_box_face_x(self, *args, **kwargs) -> List[TSRTemplate]:
        return self._apply_frame_correction(super().grasp_box_face_x(*args, **kwargs))

    def grasp_box_face_y(self, *args, **kwargs) -> List[TSRTemplate]:
        return self._apply_frame_correction(super().grasp_box_face_y(*args, **kwargs))

    def grasp_sphere(self, *args, **kwargs) -> List[TSRTemplate]:
        return self._apply_frame_correction(super().grasp_sphere(*args, **kwargs))

    def grasp_torus_side(self, *args, **kwargs) -> List[TSRTemplate]:
        return self._apply_frame_correction(super().grasp_torus_side(*args, **kwargs))

    def grasp_torus_span(self, *args, **kwargs) -> List[TSRTemplate]:
        return self._apply_frame_correction(super().grasp_torus_span(*args, **kwargs))
