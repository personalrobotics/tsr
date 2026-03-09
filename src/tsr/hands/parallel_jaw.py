"""Parallel jaw gripper hand models."""
from __future__ import annotations

import dataclasses
from typing import List, Optional, Tuple

import numpy as np

from tsr.template import TSRTemplate
from .base import GripperBase


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

    def grasp_cylinder(
        self,
        object_radius: float,
        height_range: Tuple[float, float],
        preshape: Optional[float] = None,
        k: int = 3,
        clearance: Optional[float] = None,
        angle_range: Tuple[float, float] = (0., 2 * np.pi),
        subject: str = "gripper",
        reference: str = "cylinder",
        name: str = "",
        description: str = "",
    ) -> List[TSRTemplate]:
        """Side grasp templates for a cylinder — 2*k templates total.

        Returns k depth levels × 2 roll orientations. Each template has a fixed
        radial offset baked into Tw_e, covering the full pre-grasp volume:

          depth 1/k : fingertips a clearance inside the surface (shallowest)
          depth k/k : palm a clearance from the surface (deepest)

        Two roll orientations per depth (180° apart around z_EE):
          roll=0 : palm normal = +cylinder_axis, fingers open in +tangential
          roll=π : palm normal = -cylinder_axis, fingers open in -tangential
        A symmetric hand produces identical poses; a non-symmetric hand needs both.

        The radial approach direction couples with yaw and cannot be encoded in
        Bw directly. Instead, k discrete depths are baked into Tw_e so the full
        pre-grasp volume is covered without post-processing.

        Args:
            preshape:   Pre-grasp jaw opening [m]. Default: 2*r + clearance
                        (minimum viable opening). Must exceed cylinder diameter.
            k:          Number of discrete approach depths (default 3).
            clearance:  Safety buffer [m] applied to: height ends, fingertip
                        start depth, and palm stop depth. Default: 10% of finger_length.

        Returns [] if preshape <= cylinder diameter (fingers can't span it).
        Raises ValueError for invalid geometry parameters.
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
        h0, h1 = height_range[0] + clearance, height_range[1] - clearance
        if h1 <= h0:
            raise ValueError("height_range too narrow for the given clearance")

        if not name:
            name = f"{reference.title()} Cylinder Side Grasp"
        if not description:
            description = (
                f"Side grasp on {reference} (r={object_radius*100:.0f}cm, "
                f"h=[{h0*100:.0f},{h1*100:.0f}]cm)"
            )

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

        approach_max = min(self.finger_length, object_radius) - clearance
        depths = np.linspace(clearance, approach_max, max(k, 1))

        _depth_labels = {1: ["mid"], 2: ["shallow", "deep"],
                         3: ["shallow", "mid", "deep"]}

        common = dict(
            T_ref_tsr=T_ref_tsr, Bw=Bw,
            task="grasp", subject=subject, reference=reference,
            preshape=np.array([preshape]),
        )
        templates = []
        for i, d in enumerate(depths):
            ro = object_radius + self.finger_length - d
            depth_label = (_depth_labels.get(k) or [f"depth {j+1}/{k}" for j in range(k)])[i]
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
                t_desc = (description or (
                    f"{depth_label.capitalize()} side grasp on {reference}: "
                    f"standoff {ro*1000:.0f}mm from axis, {roll_label}, "
                    f"preshape {preshape*1000:.0f}mm"
                ))
                templates.append(TSRTemplate(
                    Tw_e=Tw_e,
                    name=f"{name} — {depth_label}, {roll_label}",
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
        templates = gripper.grasp_cylinder(object_radius=0.04,
                                           height_range=(0.02, 0.10),
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

    def grasp_cylinder(self, *args, **kwargs) -> List[TSRTemplate]:
        templates = super().grasp_cylinder(*args, **kwargs)
        return [
            dataclasses.replace(t, Tw_e=t.Tw_e @ _R_z_neg90)
            for t in templates
        ]
