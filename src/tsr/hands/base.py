"""GripperBase: abstract base class for gripper hand models."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import numpy as np

from tsr.template import TSRTemplate

_SHARED_KWARGS = dict(
    preshape=None, k=3, clearance=None,
    angle_range=(0., 2 * np.pi), subject="gripper", reference="cylinder",
)


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

    **Object coordinate conventions:**

    All grasp_cylinder_* methods expect the reference object's frame to be
    placed with:

        Cylinder::

              ^ +z
              |
            --+-- z = cylinder_height  (top face)
            | | |
            | | |  ← axis along +z
            | | |
            --+-- z = 0               (bottom face, at origin)

        Box::

              ^ +z
              |
            +-+--------+  z = box_z  (top face)
            | |        |
            | +--------+  ← centered in x, y
            | |        |
            +-+--------+  z = 0      (bottom face, at origin)
              ·
            x ∈ [-box_x/2, box_x/2]
            y ∈ [-box_y/2, box_y/2]
            z ∈ [0,        box_z   ]

        Sphere::

                  * * *
                *       *
               *    O    *   ← center at origin (0, 0, 0)
                *       *
                  * * *
            radius = object_radius

        Torus::

                 ___
               /     \\
              |   O   |     <- center at origin, axis along +z
               \\     /
                 ‾‾‾
            torus_radius R = distance from center to tube center
            tube_radius  r = tube cross-section radius
            Tube center at azimuth θ: (R·cos θ, R·sin θ, 0)
            Outer radius: R + r,  Inner radius: R - r

        The reference object pose (T_world_object) transforms this frame into
        the world. E.g., a box sitting upright on a table at position p::

            T_world_box = np.eye(4)
            T_world_box[:3, 3] = p        # bottom-center of box at p
    """

    def grasp_cylinder(
        self,
        cylinder_radius: float,
        cylinder_height: float,
        preshape: Optional[float] = None,
        k: int = 3,
        clearance: Optional[float] = None,
        angle_range: Tuple[float, float] = (0., 2 * np.pi),
        subject: str = "gripper",
        reference: str = "cylinder",
    ) -> List[TSRTemplate]:
        """Generate TSRTemplates for all cylinder grasp modes.

        Combines side, top-down, and bottom-up approaches.
        Returns 2*k + k + k = 4*k templates (default k=3: 12 templates).

        Args:
            cylinder_radius: Cylinder radius [m].
            cylinder_height: Cylinder height [m]. Bottom at z=0, top at z=height.
            preshape:        Jaw opening [m]. Defaults to 2*r + clearance.
            k:               Number of discrete depths per mode (default 3).
            clearance:       Safety buffer [m]. Defaults to 10% of finger_length.
            angle_range:     Yaw freedom (default full 360°).
            subject:         Label for the end-effector entity.
            reference:       Label for the reference object.

        Returns:
            List of (2*k + k + k) TSRTemplates.
            Empty list if preshape cannot span the cylinder.
        """
        shared = dict(
            preshape=preshape, k=k, clearance=clearance,
            angle_range=angle_range, subject=subject, reference=reference,
        )
        return (
            self.grasp_cylinder_side(cylinder_radius, cylinder_height, **shared)
            + self.grasp_cylinder_top(cylinder_radius, cylinder_height, **shared)
            + self.grasp_cylinder_bottom(cylinder_radius, cylinder_height, **shared)
        )

    @abstractmethod
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
        """Generate TSRTemplates for side-grasping a cylinder — 2*k templates.

        Returns pre-grasp configurations: hand open at preshape, positioned
        so closing the fingers contacts the cylinder surface radially.

        Args:
            cylinder_radius: Cylinder radius [m].
            cylinder_height: Cylinder height [m]. Graspable band: [clearance, height-clearance].
            preshape:        Jaw opening [m]. Defaults to 2*r + clearance.
            k:               Number of discrete approach depths (default 3).
            clearance:       Safety buffer [m] at height ends and depth limits.
                             Defaults to 10% of finger_length.
            angle_range:     Yaw freedom (default full 360°).
            subject:         Label for the end-effector entity.
            reference:       Label for the reference object.
            name:            Template name prefix.
            description:     Template description.

        Returns:
            List of 2*k TSRTemplates (k depths × 2 roll orientations).
            Empty list if preshape cannot span the cylinder.
        """
        raise NotImplementedError

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
        """Generate TSRTemplates for grasping a cylinder from above — k templates.

        Returns pre-grasp configurations with z_EE = [0,0,-1] (approach
        downward). TSR origin at z = cylinder_height (top face).

        Args:
            cylinder_radius: Cylinder radius [m].
            cylinder_height: Cylinder height [m] (z of top face).
            preshape:        Jaw opening [m]. Must exceed cylinder diameter.
                             Defaults to 2*r + clearance.
            k:               Number of discrete approach depths (default 3).
            clearance:       Safety buffer [m]. Defaults to 10% of finger_length.
            angle_range:     Yaw freedom (default full 360°).

        Returns:
            List of k TSRTemplates. Empty list if preshape cannot span the cylinder.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement grasp_cylinder_top"
        )

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
        """Generate TSRTemplates for grasping a cylinder from below — k templates.

        Returns pre-grasp configurations with z_EE = [0,0,+1] (approach
        upward). TSR origin at z = 0 (bottom face).

        Args:
            cylinder_radius: Cylinder radius [m].
            cylinder_height: Cylinder height [m] (unused geometrically; kept for
                             API consistency with the other cylinder methods).
            preshape:        Jaw opening [m]. Must exceed cylinder diameter.
                             Defaults to 2*r + clearance.
            k:               Number of discrete approach depths (default 3).
            clearance:       Safety buffer [m]. Defaults to 10% of finger_length.
            angle_range:     Yaw freedom (default full 360°).

        Returns:
            List of k TSRTemplates. Empty list if preshape cannot span the cylinder.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement grasp_cylinder_bottom"
        )

    def grasp_box(
        self,
        box_x: float,
        box_y: float,
        box_z: float,
        preshape: Optional[float] = None,
        k: int = 3,
        clearance: Optional[float] = None,
        subject: str = "gripper",
        reference: str = "box",
    ) -> List[TSRTemplate]:
        """Generate TSRTemplates for all box grasp modes — 6*k templates.

        Combines top, bottom, +x/-x face, and +y/-y face approaches.
        Each face offers two finger orientations (one per face axis), filtered
        by max_aperture.  Maximum 2 orientations × 6 faces × k depths = 12*k.
        Actual count depends on box dimensions vs gripper aperture.

        Box coordinate convention::

            x ∈ [-box_x/2, +box_x/2]   (centered)
            y ∈ [-box_y/2, +box_y/2]   (centered)
            z ∈ [0,         box_z   ]   (bottom at z=0)

        Args:
            box_x:     Box width  [m] (along x-axis).
            box_y:     Box depth  [m] (along y-axis).
            box_z:     Box height [m] (along z-axis; top face at z=box_z).
            preshape:  Jaw opening [m]. Defaults to max_aperture / 2.
            k:         Number of discrete approach depths per face (default 3).
            clearance: Safety buffer [m]. Defaults to 10% of finger_length.
            subject:   Label for the end-effector entity.
            reference: Label for the reference object.

        Returns:
            List of 6*k TSRTemplates.
        """
        shared = dict(preshape=preshape, k=k, clearance=clearance,
                      subject=subject, reference=reference)
        return (
            self.grasp_box_face_x(box_x, box_y, box_z, **shared)
            + self.grasp_box_face_y(box_x, box_y, box_z, **shared)
            + self.grasp_box_top(box_x, box_y, box_z, **shared)
            + self.grasp_box_bottom(box_x, box_y, box_z, **shared)
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
        """Generate TSRTemplates for grasping a box from above — k templates.

        Gripper approaches from above (z_EE = [0,0,-1]). TSR origin at
        z = box_z (top face). Fingers slide freely in x and y within the face
        bounds; no rotational freedom.

        Args:
            box_x:     Box width  [m].
            box_y:     Box depth  [m].
            box_z:     Box height [m] (z of top face).
            preshape:  Jaw opening [m]. Defaults to max_aperture / 2.
            k:         Number of discrete approach depths (default 3).
            clearance: Safety buffer [m]. Defaults to 10% of finger_length.

        Returns:
            List of k TSRTemplates.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement grasp_box_top"
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
        """Generate TSRTemplates for grasping a box from below — k templates.

        Gripper approaches from below (z_EE = [0,0,+1]). TSR origin at
        z = 0 (bottom face). Fingers slide freely in x and y within the face
        bounds; no rotational freedom.

        Args:
            box_x:     Box width  [m].
            box_y:     Box depth  [m].
            box_z:     Box height [m] (unused geometrically; kept for API symmetry).
            preshape:  Jaw opening [m]. Defaults to max_aperture / 2.
            k:         Number of discrete approach depths (default 3).
            clearance: Safety buffer [m]. Defaults to 10% of finger_length.

        Returns:
            List of k TSRTemplates.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement grasp_box_bottom"
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
        """Generate TSRTemplates for grasping the ±x faces of a box — 2*k templates.

        k templates approach from +x (z_EE = [-1,0,0]) and k from -x
        (z_EE = [+1,0,0]). Fingers slide freely in y and z within the face
        bounds; no rotational freedom.

        Args:
            box_x:     Box width  [m] (standoff is box_x/2 + finger_length - depth).
            box_y:     Box depth  [m].
            box_z:     Box height [m].
            preshape:  Jaw opening [m]. Defaults to max_aperture / 2.
            k:         Number of discrete approach depths per face (default 3).
            clearance: Safety buffer [m]. Defaults to 10% of finger_length.

        Returns:
            List of 2*k TSRTemplates (k for +x face, k for -x face).
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement grasp_box_face_x"
        )

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
        """Generate TSRTemplates for grasping the ±y faces of a box — 2*k templates.

        k templates approach from +y (z_EE = [0,-1,0]) and k from -y
        (z_EE = [0,+1,0]). Fingers slide freely in x and z within the face
        bounds; no rotational freedom.

        Args:
            box_x:     Box width  [m].
            box_y:     Box depth  [m] (standoff is box_y/2 + finger_length - depth).
            box_z:     Box height [m].
            preshape:  Jaw opening [m]. Defaults to max_aperture / 2.
            k:         Number of discrete approach depths per face (default 3).
            clearance: Safety buffer [m]. Defaults to 10% of finger_length.

        Returns:
            List of 2*k TSRTemplates (k for +y face, k for -y face).
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement grasp_box_face_y"
        )

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
        """Equatorial grasp templates for a sphere — 2*k templates.

        Approach from any direction in the horizontal (xy) plane. TSR origin
        at the sphere center. Full yaw covers all equatorial approach directions.
        k discrete depths × 2 roll orientations.

        Sphere coordinate convention: center at origin, radius = object_radius.

        Args:
            object_radius: Sphere radius [m].
            preshape:      Jaw opening [m]. Defaults to 2*r + clearance.
            k:             Number of discrete approach depths (default 3).
            clearance:     Safety buffer [m]. Defaults to 10% of finger_length.
            angle_range:   Yaw freedom (default full 360°).
            subject:       Label for the end-effector entity.
            reference:     Label for the reference object.
            name:          Template name prefix.
            description:   Template description.

        Returns:
            List of k TSRTemplates. Empty list if preshape cannot span the sphere.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement grasp_sphere"
        )

    def grasp_torus(
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
    ) -> List[TSRTemplate]:
        """Generate TSRTemplates for all torus grasp modes.

        Combines radial side grasps (always) and span grasps (when the outer
        diameter fits within max_aperture).

        Torus coordinate convention: center at origin, axis along +z.
            torus_radius R: distance from center to tube center.
            tube_radius  r: cross-section radius of the tube.

        Side mode (2 * k * n_minor templates):
            Approach from n_minor discrete angles α ∈ [−π/2, +π/2] in the
            tube cross-section plane. α=0 is a pure equatorial radial approach;
            α=±π/2 approaches from below/above (matching span geometry).
            k depths × 2 hand flips per angle. Full yaw covers all azimuth
            positions around the ring.

        Span mode (up to 2*k templates, silently omitted if too large):
            Approach from above and below; fingers span the full outer diameter
            2*(R+r). Only included when 2*(R+r) + clearance ≤ max_aperture.
            k depths each, full yaw covers all finger orientations.

        Args:
            torus_radius:  Major radius R [m] — center to tube center.
            tube_radius:   Minor radius r [m] — tube cross-section.
            preshape:      Jaw opening for side grasps [m]. Defaults to
                           2*r + clearance.
            k:             Number of discrete depths per mode (default 3).
            n_minor:       Discrete approach angles around the tube
                           cross-section (default 5).
            clearance:     Safety buffer [m]. Defaults to 10% of finger_length.
            angle_range:   Yaw freedom for side grasps (default full 360°).
            subject:       Label for the end-effector entity.
            reference:     Label for the reference object.

        Returns:
            List of 2*k*n_minor + up to 2*k TSRTemplates.
        """
        shared = dict(preshape=preshape, k=k, clearance=clearance,
                      subject=subject, reference=reference)
        return (
            self.grasp_torus_side(torus_radius, tube_radius,
                                  n_minor=n_minor, angle_range=angle_range,
                                  **shared)
            + self.grasp_torus_span(torus_radius, tube_radius, **shared)
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

        The gripper approaches from n_minor discrete angles in the tube
        cross-section plane (α ∈ [−π/2, +π/2]), with full azimuthal yaw
        freedom around the torus ring. Two hand flip variants per (α, depth).

        Args:
            torus_radius:  Major radius R [m].
            tube_radius:   Minor radius r [m].
            preshape:      Jaw opening [m]. Defaults to 2*r + clearance.
            k:             Number of discrete approach depths (default 3).
            n_minor:       Discrete approach angles in the tube cross-section
                           (default 5: −π/2, −π/4, 0, +π/4, +π/2).
            clearance:     Safety buffer [m]. Defaults to 10% of finger_length.
            angle_range:   Yaw freedom (default full 360°).

        Returns:
            List of 2*k*n_minor TSRTemplates. Empty if preshape cannot span
            the tube or finger_length ≤ tube_radius.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement grasp_torus_side"
        )

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

        Approach from above (+k) and below (-k) with fingers spanning the full
        outer diameter 2*(R+r). Full yaw freedom covers all finger orientations.
        Silently returns [] if the outer diameter + clearance exceeds max_aperture.

        Args:
            torus_radius:  Major radius R [m].
            tube_radius:   Minor radius r [m].
            preshape:      Jaw opening [m]. Defaults to 2*(R+r) + clearance.
            k:             Number of discrete depths per direction (default 3).
            clearance:     Safety buffer [m]. Defaults to 10% of finger_length.

        Returns:
            List of up to 2*k TSRTemplates (k top + k bottom).
            Empty list if max_aperture cannot span the outer diameter.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement grasp_torus_span"
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
