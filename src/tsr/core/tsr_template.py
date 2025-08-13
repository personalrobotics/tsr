from __future__ import annotations

from dataclasses import dataclass
import numpy as np

# Use existing core TSR implementation without changes.
from .tsr import TSR as CoreTSR  # type: ignore[attr-defined]


@dataclass(frozen=True)
class TSRTemplate:
    """Neutral TSR template (pure geometry, scene-agnostic).

    A TSRTemplate defines a TSR in a reference-relative coordinate frame,
    allowing it to be instantiated at any reference pose in the world.
    This makes templates reusable across different scenes and object poses.

    Attributes:
        T_ref_tsr: 4×4 transform from REFERENCE frame to TSR frame.
                   This defines how the TSR frame is oriented relative to
                   the reference entity (e.g., object).
        Tw_e: 4×4 transform from TSR frame to SUBJECT frame at Bw = 0 (canonical).
              This defines the desired pose of the subject (e.g., end-effector)
              relative to the TSR frame when all bounds are at their nominal values.
        Bw: (6,2) bounds in TSR frame over [x,y,z,roll,pitch,yaw].
            Each row [i,:] defines the min/max bounds for dimension i.
            Translation bounds (rows 0-2) are in meters.
            Rotation bounds (rows 3-5) are in radians using RPY convention.

    Examples:
        >>> # Create a template for grasping a cylinder from the side
        >>> template = TSRTemplate(
        ...     T_ref_tsr=np.eye(4),  # TSR frame aligned with cylinder frame
        ...     Tw_e=np.array([
        ...         [0, 0, 1, -0.05],  # Approach from -z, 5cm offset
        ...         [1, 0, 0, 0],      # x-axis perpendicular to cylinder
        ...         [0, 1, 0, 0.05],   # y-axis along cylinder axis
        ...         [0, 0, 0, 1]
        ...     ]),
        ...     Bw=np.array([
        ...         [0, 0],           # x: fixed position
        ...         [0, 0],           # y: fixed position
        ...         [-0.01, 0.01],    # z: small tolerance
        ...         [0, 0],           # roll: fixed
        ...         [0, 0],           # pitch: fixed
        ...         [-np.pi, np.pi]   # yaw: full rotation
        ...     ])
        ... )
        >>> 
        >>> # Instantiate at a specific cylinder pose
        >>> cylinder_pose = np.array([
        ...     [1, 0, 0, 0.5],  # Cylinder at x=0.5
        ...     [0, 1, 0, 0.0],
        ...     [0, 0, 1, 0.3],
        ...     [0, 0, 0, 1]
        ... ])
        >>> tsr = template.instantiate(cylinder_pose)
        >>> pose = tsr.sample()  # Sample a grasp pose
    """

    T_ref_tsr: np.ndarray
    Tw_e: np.ndarray
    Bw: np.ndarray

    def instantiate(self, T_ref_world: np.ndarray) -> CoreTSR:
        """Bind this template to a concrete reference pose in world.

        This method creates a concrete TSR by combining the template's
        reference-relative definition with a specific reference pose in
        the world coordinate frame.

        Args:
            T_ref_world: 4×4 pose of the reference entity in world frame.
                        This is typically the pose of the object being
                        manipulated (e.g., mug, table, valve).

        Returns:
            CoreTSR whose T0_w = T_ref_world @ T_ref_tsr, Tw_e = Tw_e, Bw = Bw.
            The resulting TSR can be used for sampling, distance calculations,
            and other TSR operations.

        Examples:
            >>> # Create a template for placing objects on a table
            >>> place_template = TSRTemplate(
            ...     T_ref_tsr=np.eye(4),
            ...     Tw_e=np.array([
            ...         [1, 0, 0, 0],      # Object x-axis aligned with table
            ...         [0, 1, 0, 0],      # Object y-axis aligned with table
            ...         [0, 0, 1, 0.02],   # Object 2cm above table surface
            ...         [0, 0, 0, 1]
            ...     ]),
            ...     Bw=np.array([
            ...         [-0.1, 0.1],       # x: allow sliding on table
            ...         [-0.1, 0.1],       # y: allow sliding on table
            ...         [0, 0],            # z: fixed height
            ...         [0, 0],            # roll: keep level
            ...         [0, 0],            # pitch: keep level
            ...         [-np.pi/4, np.pi/4]  # yaw: allow some rotation
            ...     ])
            ... )
            >>> 
            >>> # Instantiate at table pose
            >>> table_pose = np.eye(4)  # Table at world origin
            >>> place_tsr = place_template.instantiate(table_pose)
            >>> placement_pose = place_tsr.sample()
        """
        T0_w = T_ref_world @ self.T_ref_tsr
        return CoreTSR(T0_w=T0_w, Tw_e=self.Tw_e, Bw=self.Bw)
