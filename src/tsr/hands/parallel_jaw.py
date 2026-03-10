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

    def _validate_box(self, box_x: float, box_y: float, box_z: float,
                      preshape: float) -> None:
        for dim, name in ((box_x, "box_x"), (box_y, "box_y"), (box_z, "box_z")):
            if dim <= 0:
                raise ValueError(f"{name} must be > 0")
        if preshape > self.max_aperture:
            raise ValueError(
                f"preshape {preshape:.3f}m > max_aperture {self.max_aperture:.3f}m"
            )

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
        """Top-down grasp templates for a box — k templates.

        Gripper approaches from above (z_EE = [0,0,-1]). TSR origin at
        z = box_z. Fingers slide in x and y within [−box_x/2+c, +box_x/2−c]
        and [−box_y/2+c, +box_y/2+c]; no rotational freedom.
        """
        if clearance is None:
            clearance = 0.1 * self.finger_length
        if preshape is None:
            preshape = self.max_aperture / 2.
        self._validate_box(box_x, box_y, box_z, preshape)

        hx, hy = box_x / 2. - clearance, box_y / 2. - clearance
        if hx <= 0 or hy <= 0:
            raise ValueError("box face too small for the given clearance")

        if not name:
            name = f"{reference.title()} Box Top Grasp"

        T_ref_tsr = np.eye(4)
        T_ref_tsr[2, 3] = box_z

        Bw = np.array([
            [-hx, hx],  # x: slide along face
            [-hy, hy],  # y: slide along face
            [0.,  0.],  # z: fixed at top face
            [0.,  0.],  # roll
            [0.,  0.],  # pitch
            [0.,  0.],  # yaw: no rotation
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
                f"palm {h_palm*1000:.0f}mm above top, preshape {preshape*1000:.0f}mm"
            )
            # z_EE = [0,0,-1] (approach down)
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
        """Bottom-up grasp templates for a box — k templates.

        Gripper approaches from below (z_EE = [0,0,+1]). TSR origin at
        z = 0 (bottom face). Fingers slide in x and y; no rotational freedom.
        """
        if clearance is None:
            clearance = 0.1 * self.finger_length
        if preshape is None:
            preshape = self.max_aperture / 2.
        self._validate_box(box_x, box_y, box_z, preshape)

        hx, hy = box_x / 2. - clearance, box_y / 2. - clearance
        if hx <= 0 or hy <= 0:
            raise ValueError("box face too small for the given clearance")

        if not name:
            name = f"{reference.title()} Box Bottom Grasp"

        del box_z  # bottom face is always at z=0; accepted for API symmetry
        T_ref_tsr = np.eye(4)  # origin at z=0

        Bw = np.array([
            [-hx, hx],  # x: slide along face
            [-hy, hy],  # y: slide along face
            [0.,  0.],  # z: fixed at bottom face
            [0.,  0.],  # roll
            [0.,  0.],  # pitch
            [0.,  0.],  # yaw: no rotation
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
        """Grasp templates for the ±x faces of a box — 2*k templates.

        k templates approach from +x (z_EE = [-1,0,0]) and k from -x
        (z_EE = [+1,0,0]). Fingers slide in y and z; no rotational freedom.

        Rotation matrices (det=+1, each is a valid SO(3)):
          +x face: z_EE=[-1,0,0], y_EE=[0,0,1]  →  x_EE = y×z = [0,1,0]×[-1,0,0] = [0,0,-1]
                   Wait — let's be careful: x_EE = y_EE × z_EE = [0,0,1]×[-1,0,0] = [0*0-1*0, 1*(-1)-0*0, 0*0-0*(-1)] = [0,-1,0]
          +x face: columns [x_EE | y_EE | z_EE] = [[0,-1,0]^T | [0,0,1]^T | [-1,0,0]^T]
          -x face: z_EE=[+1,0,0], y_EE=[0,0,1] → x_EE = [0,0,1]×[1,0,0] = [0*0-1*0, 1*1-0*0, 0*0-0*1] = [0,1,0]
          -x face: columns [[0,1,0]^T | [0,0,1]^T | [1,0,0]^T]
        """
        if clearance is None:
            clearance = 0.1 * self.finger_length
        if preshape is None:
            preshape = self.max_aperture / 2.
        self._validate_box(box_x, box_y, box_z, preshape)

        hy = box_y / 2. - clearance
        hz_lo = clearance
        hz_hi = box_z - clearance
        if hy <= 0:
            raise ValueError("box_y too small for the given clearance")
        if hz_hi <= hz_lo:
            raise ValueError("box_z too small for the given clearance")

        if not name:
            name = f"{reference.title()} Box X-Face Grasp"

        hz_mid  = (hz_lo + hz_hi) / 2.
        hz_half = (hz_hi - hz_lo) / 2.

        approach_max = min(self.finger_length, box_x / 2.) - clearance
        depths = np.linspace(clearance, approach_max, max(k, 1))

        # +x face: approach from +x, z_EE = [-1,0,0], y_EE = [0,0,1], x_EE = [0,-1,0]
        T_ref_tsr_pos = np.eye(4)
        T_ref_tsr_pos[0, 3] = box_x / 2.
        T_ref_tsr_pos[2, 3] = hz_mid

        # -x face: approach from -x, z_EE = [+1,0,0], y_EE = [0,0,1], x_EE = [0,+1,0]
        T_ref_tsr_neg = np.eye(4)
        T_ref_tsr_neg[0, 3] = -box_x / 2.
        T_ref_tsr_neg[2, 3] = hz_mid

        Bw = np.array([
            [0.,     0.    ],  # x: no radial freedom
            [-hy,    hy    ],  # y: slide along face
            [-hz_half, hz_half],  # z: height range
            [0.,     0.    ],  # roll
            [0.,     0.    ],  # pitch
            [0.,     0.    ],  # yaw: no rotation
        ])

        common_pos = dict(
            T_ref_tsr=T_ref_tsr_pos, Bw=Bw,
            task="grasp", subject=subject, reference=reference,
            preshape=np.array([preshape]),
        )
        common_neg = dict(
            T_ref_tsr=T_ref_tsr_neg, Bw=Bw,
            task="grasp", subject=subject, reference=reference,
            preshape=np.array([preshape]),
        )

        templates = []
        for i, d in enumerate(depths):
            standoff = box_x / 2. + self.finger_length - d
            dlabel = _depth_label(k, i)

            # +x face: R cols = [x_EE | y_EE | z_EE] = [[0,-1,0]^T | [0,0,1]^T | [-1,0,0]^T]
            Tw_e_pos = np.array([
                [ 0.,  0., -1.,  standoff],
                [-1.,  0.,  0.,  0.      ],
                [ 0.,  1.,  0.,  0.      ],
                [ 0.,  0.,  0.,  1.      ],
            ])
            # -x face: R cols = [[0,1,0]^T | [0,0,1]^T | [1,0,0]^T]
            Tw_e_neg = np.array([
                [ 0.,  0.,  1., -standoff],
                [ 1.,  0.,  0.,  0.      ],
                [ 0.,  1.,  0.,  0.      ],
                [ 0.,  0.,  0.,  1.      ],
            ])
            for Tw_e, common, face_label in (
                (Tw_e_pos, common_pos, "+x"),
                (Tw_e_neg, common_neg, "-x"),
            ):
                t_desc = description or (
                    f"{dlabel.capitalize()} {face_label}-face grasp on {reference}: "
                    f"standoff {standoff*1000:.0f}mm, preshape {preshape*1000:.0f}mm"
                )
                templates.append(TSRTemplate(
                    Tw_e=Tw_e,
                    name=f"{name} {face_label} — {dlabel}",
                    description=t_desc,
                    **common,
                ))
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
        """Grasp templates for the ±y faces of a box — 2*k templates.

        k templates approach from +y (z_EE = [0,-1,0]) and k from -y
        (z_EE = [0,+1,0]). Fingers slide in x and z; no rotational freedom.

        Rotation matrices (det=+1):
          +y face: z_EE=[0,-1,0], y_EE=[1,0,0] → x_EE = y×z = [1,0,0]×[0,-1,0] = [0,0,-1]
                   cols = [[0,0,-1]^T | [1,0,0]^T | [0,-1,0]^T]
          -y face: z_EE=[0,+1,0], y_EE=[1,0,0] → x_EE = [1,0,0]×[0,1,0] = [0,0,1]
                   cols = [[0,0,1]^T | [1,0,0]^T | [0,1,0]^T]
        """
        if clearance is None:
            clearance = 0.1 * self.finger_length
        if preshape is None:
            preshape = self.max_aperture / 2.
        self._validate_box(box_x, box_y, box_z, preshape)

        hx = box_x / 2. - clearance
        hz_lo = clearance
        hz_hi = box_z - clearance
        if hx <= 0:
            raise ValueError("box_x too small for the given clearance")
        if hz_hi <= hz_lo:
            raise ValueError("box_z too small for the given clearance")

        if not name:
            name = f"{reference.title()} Box Y-Face Grasp"

        hz_mid  = (hz_lo + hz_hi) / 2.
        hz_half = (hz_hi - hz_lo) / 2.

        approach_max = min(self.finger_length, box_y / 2.) - clearance
        depths = np.linspace(clearance, approach_max, max(k, 1))

        T_ref_tsr_pos = np.eye(4)
        T_ref_tsr_pos[1, 3] = box_y / 2.
        T_ref_tsr_pos[2, 3] = hz_mid

        T_ref_tsr_neg = np.eye(4)
        T_ref_tsr_neg[1, 3] = -box_y / 2.
        T_ref_tsr_neg[2, 3] = hz_mid

        Bw = np.array([
            [-hx,     hx    ],  # x: slide along face
            [0.,      0.    ],  # y: no radial freedom
            [-hz_half, hz_half],  # z: height range
            [0.,      0.    ],  # roll
            [0.,      0.    ],  # pitch
            [0.,      0.    ],  # yaw: no rotation
        ])

        common_pos = dict(
            T_ref_tsr=T_ref_tsr_pos, Bw=Bw,
            task="grasp", subject=subject, reference=reference,
            preshape=np.array([preshape]),
        )
        common_neg = dict(
            T_ref_tsr=T_ref_tsr_neg, Bw=Bw,
            task="grasp", subject=subject, reference=reference,
            preshape=np.array([preshape]),
        )

        templates = []
        for i, d in enumerate(depths):
            standoff = box_y / 2. + self.finger_length - d
            dlabel = _depth_label(k, i)

            # +y face: cols = [[0,0,-1]^T | [1,0,0]^T | [0,-1,0]^T]
            Tw_e_pos = np.array([
                [ 0.,  1.,  0.,  0.      ],
                [ 0.,  0., -1.,  standoff],
                [-1.,  0.,  0.,  0.      ],
                [ 0.,  0.,  0.,  1.      ],
            ])
            # -y face: cols = [[0,0,1]^T | [1,0,0]^T | [0,1,0]^T]
            Tw_e_neg = np.array([
                [ 0.,  1.,  0.,  0.      ],
                [ 0.,  0.,  1., -standoff],
                [ 1.,  0.,  0.,  0.      ],
                [ 0.,  0.,  0.,  1.      ],
            ])
            for Tw_e, common, face_label in (
                (Tw_e_pos, common_pos, "+y"),
                (Tw_e_neg, common_neg, "-y"),
            ):
                t_desc = description or (
                    f"{dlabel.capitalize()} {face_label}-face grasp on {reference}: "
                    f"standoff {standoff*1000:.0f}mm, preshape {preshape*1000:.0f}mm"
                )
                templates.append(TSRTemplate(
                    Tw_e=Tw_e,
                    name=f"{name} {face_label} — {dlabel}",
                    description=t_desc,
                    **common,
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
