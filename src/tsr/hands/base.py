"""GripperBase: abstract base class for gripper hand models."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import numpy as np

from tsr.template import TSRTemplate


class GripperBase(ABC):
    """Abstract base class for gripper hand models.

    A hand model generates TSRTemplates from object geometry (grasp_* methods)
    and optionally provides a renderer for visualization.

    **Gripper frame convention** (canonical for this library):

        z = approach direction  (toward object surface)
        y = finger opening direction
        x = palm normal         (right-hand rule: x = y × z)

    Poses sampled from grasp TSRs are **pre-grasp configurations**: the hand
    is open at ``preshape``, positioned so that closing the fingers achieves
    stable contact with the object. The TSR guarantees this geometrically —
    it constrains where the hand must be before closing, but does not
    explicitly verify force closure.

    To convert sampled poses to another convention, apply a fixed rotation::

        # AnyGrasp / GraspNet uses x=approach:
        R = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
        anygrasp_pose = tsr_pose @ np.block([[R, np.zeros((3,1))],
                                             [np.zeros((1,3)), [[1]]]])
    """

    @abstractmethod
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
        """Generate TSRTemplates for side-grasping a cylinder.

        Returns pre-grasp configurations: hand open at preshape, positioned
        so closing the fingers contacts the cylinder surface.

        Args:
            object_radius:  Cylinder radius [m].
            height_range:   (z_min, z_max) of the graspable band [m].
            preshape:       Jaw opening [m]. Defaults to 2*r + clearance.
            k:              Number of discrete approach depths (default 3).
            clearance:      Safety buffer [m] at height ends and depth limits.
                            Defaults to 10% of finger_length.
            angle_range:    Yaw freedom (default full 360°).
            subject:        Label for the end-effector entity.
            reference:      Label for the reference object.
            name:           Template name prefix.
            description:    Template description.

        Returns:
            List of 2*k TSRTemplates (k depths × 2 roll orientations).
            Empty list if preshape cannot span the cylinder.
        """
        raise NotImplementedError

    def grasp_sphere(
        self,
        object_radius: float,
        preshape: Optional[float] = None,
        clearance: Optional[float] = None,
        subject: str = "gripper",
        reference: str = "sphere",
        name: str = "",
        description: str = "",
    ) -> List[TSRTemplate]:
        """Generate TSRTemplates for grasping a sphere.

        Not yet implemented — see https://github.com/personalrobotics/tsr/issues/26.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement grasp_sphere"
        )

    def grasp_torus(
        self,
        torus_radius: float,
        tube_radius: float,
        preshape: Optional[float] = None,
        clearance: Optional[float] = None,
        subject: str = "gripper",
        reference: str = "torus",
        name: str = "",
        description: str = "",
    ) -> List[TSRTemplate]:
        """Generate TSRTemplates for grasping a torus (e.g., rotating handle).

        Not yet implemented — see https://github.com/personalrobotics/tsr/issues/26.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement grasp_torus"
        )

    def renderer(self):
        """Return a SubjectRenderer for use with TSRVisualizer.

        Returns:
            Callable ``(pl: pv.Plotter, pose_4x4: np.ndarray, color: tuple) -> None``

        Raises:
            NotImplementedError: if this hand has no registered renderer.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement renderer()"
        )
