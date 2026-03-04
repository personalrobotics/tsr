"""Parallel jaw gripper TSR autogeneration example.

This module shows how to write a gripper class that autogenerates TSR templates
from object geometry. The pattern:
  - ParallelJawGripper knows its own geometry (finger_length, max_aperture)
  - It generates TSRTemplates from object shape parameters

Gripper frame convention (canonical for this library):
    z = approach direction (toward object surface)
    y = finger opening direction
    x = palm normal  (right-hand rule: x = y × z)

AnyGrasp / GraspNet convention:
    x = approach direction
    y = finger opening direction
    z = palm normal

To convert AnyGrasp output to this library's convention, apply R_y(90°):
    R_convert = [[0, 0, -1],
                 [0, 1,  0],
                 [1, 0,  0]]

Usage:
    uv run python examples/parallel_jaw_grasp.py
"""

import numpy as np
from typing import Tuple, Optional
from tsr.template import TSRTemplate


class ParallelJawGripper:
    """Generates TSR templates for a parallel jaw gripper.

    Frame convention:
        z = approach direction (toward object surface)
        y = finger opening direction
        x = palm normal (right-hand rule: x = y × z)

    For a cylinder side grasp, at yaw=0 in the object's TSR frame:
        - EE x-axis aligns with the cylinder axis (z of TSR frame)
        - EE y-axis is tangential (y of TSR frame)
        - EE z-axis points radially inward toward the cylinder center (-x of TSR frame)
        - EE origin is at (radius + finger_length) radially outward from cylinder center

    This gives the Tw_e:
        [[0, 0, -1, radius + finger_length],
         [0, 1,  0, 0                    ],
         [1, 0,  0, 0                    ],
         [0, 0,  0, 1                    ]]

    Args:
        finger_length: Distance from palm to fingertip in meters.
        max_aperture: Maximum jaw opening in meters.
    """

    def __init__(self, finger_length: float, max_aperture: float):
        self.finger_length = finger_length
        self.max_aperture = max_aperture

    def grasp_cylinder(
        self,
        object_radius: float,
        height_range: Tuple[float, float],
        angle_range: Tuple[float, float] = (0., 2 * np.pi),
        subject: str = "gripper",
        reference: str = "object",
        name: str = "",
        description: str = "",
    ) -> TSRTemplate:
        """Generate a side grasp template for a cylinder.

        The TSR frame coincides with the cylinder frame (z = cylinder axis).
        The gripper approaches radially from outside the cylinder, can land at
        any height in height_range, and can rotate freely in angle_range around
        the cylinder axis.

        Args:
            object_radius: Cylinder radius in meters.
            height_range: (z_min, z_max) grasp height range relative to cylinder
                base in meters.
            angle_range: (angle_min, angle_max) in radians around cylinder axis.
            subject: Subject entity string (e.g. "gripper", "robotiq_2f140").
            reference: Reference entity string (e.g. "mug", "can").
            name: Template name (auto-generated if empty).
            description: Template description (auto-generated if empty).

        Returns:
            TSRTemplate for cylinder side grasp.

        Raises:
            ValueError: If object diameter >= max_aperture or parameters invalid.
        """
        if object_radius <= 0:
            raise ValueError("object_radius must be > 0")
        if 2. * object_radius >= self.max_aperture:
            raise ValueError(
                f"object diameter {2*object_radius:.3f}m >= max_aperture {self.max_aperture:.3f}m"
            )
        h0, h1 = height_range
        if h1 <= h0:
            raise ValueError("height_range[1] must be > height_range[0]")

        if not name:
            name = f"{reference.title()} Cylinder Side Grasp"
        if not description:
            description = (
                f"Side grasp on {reference} (r={object_radius*100:.0f}cm, "
                f"h=[{h0*100:.0f},{h1*100:.0f}]cm)"
            )

        # T_ref_tsr shifts TSR frame to height midpoint so Bw[2] is symmetric.
        z_mid = (h0 + h1) / 2.
        z_half = (h1 - h0) / 2.
        T_ref_tsr = np.eye(4)
        T_ref_tsr[2, 3] = z_mid

        # Tw_e: EE is at (radius + finger_length) radially outward from cylinder
        # center at yaw=0. EE z-axis points inward (toward cylinder = approach).
        #
        # Columns of rotation part (EE frame axes in TSR frame):
        #   col 0 (x̂_EE) = [0,0,1] = ẑ_TSR  (cylinder axis = palm normal)
        #   col 1 (ŷ_EE) = [0,1,0] = ŷ_TSR  (tangential = finger opening)
        #   col 2 (ẑ_EE) = [-1,0,0] = -x̂_TSR (radially inward = approach)
        radial_offset = object_radius + self.finger_length
        Tw_e = np.array([
            [0., 0., -1., radial_offset],
            [0., 1.,  0., 0.           ],
            [1., 0.,  0., 0.           ],
            [0., 0.,  0., 1.           ],
        ])

        Bw = np.array([
            [0.,            0.          ],   # x: no radial freedom
            [0.,            0.          ],   # y: no tangential freedom
            [-z_half,       z_half      ],   # z: height range (symmetric)
            [0.,            0.          ],   # roll: fixed
            [0.,            0.          ],   # pitch: fixed
            [angle_range[0], angle_range[1]], # yaw: angular freedom
        ])

        return TSRTemplate(
            T_ref_tsr=T_ref_tsr,
            Tw_e=Tw_e,
            Bw=Bw,
            task="grasp",
            subject=subject,
            reference=reference,
            name=name,
            description=description,
            preshape=np.array([2. * object_radius]),
        )

    def grasp_box_face(
        self,
        face_width: float,
        face_height: float,
        face_depth: float,
        subject: str = "gripper",
        reference: str = "object",
        name: str = "",
        description: str = "",
    ) -> TSRTemplate:
        """Generate a grasp template for the front face of a box.

        The TSR frame is at the face center (z = outward normal of face).
        The gripper approaches along the face normal.

        Args:
            face_width: Width of the face (finger opening direction) in meters.
            face_height: Height of the face in meters.
            face_depth: Unused (included for clarity of caller intent).
            subject: Subject entity string.
            reference: Reference entity string.
            name: Template name.
            description: Template description.

        Returns:
            TSRTemplate for box face grasp.
        """
        if face_width <= 0 or face_height <= 0:
            raise ValueError("face dimensions must be > 0")
        if face_width >= self.max_aperture:
            raise ValueError(
                f"face_width {face_width:.3f}m >= max_aperture {self.max_aperture:.3f}m"
            )

        if not name:
            name = f"{reference.title()} Box Face Grasp"
        if not description:
            description = f"Front face grasp on {reference}"

        # TSR frame: at face center, z = outward face normal.
        # EE approaches along -z (face normal inward = approach).
        # EE z-axis = -z_TSR (inward), y-axis = y_TSR (height), x-axis = x_TSR (width)
        #
        # Columns of rotation:
        #   col 0 (x̂_EE) = [1, 0, 0] = x̂_TSR  (width direction = palm normal)
        #   col 1 (ŷ_EE) = [0, 1, 0] = ŷ_TSR  (height direction = finger opening)
        #   col 2 (ẑ_EE) = [0, 0, -1] = -ẑ_TSR (inward = approach)
        T_ref_tsr = np.eye(4)
        Tw_e = np.array([
            [1., 0.,  0., 0.                  ],
            [0., 1.,  0., 0.                  ],
            [0., 0., -1., self.finger_length   ],
            [0., 0.,  0., 1.                  ],
        ])

        Bw = np.array([
            [-face_width/2.,   face_width/2. ],   # x: slide along width
            [-face_height/2.,  face_height/2.],   # y: slide along height
            [0.,               0.            ],   # z: fixed (at face)
            [0.,               0.            ],   # roll: fixed
            [0.,               0.            ],   # pitch: fixed
            [0.,               0.            ],   # yaw: fixed
        ])

        return TSRTemplate(
            T_ref_tsr=T_ref_tsr,
            Tw_e=Tw_e,
            Bw=Bw,
            task="grasp",
            subject=subject,
            reference=reference,
            name=name,
            description=description,
            preshape=np.array([face_width]),
        )

    def grasp_plane(
        self,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        subject: str = "gripper",
        reference: str = "surface",
        name: str = "",
        description: str = "",
    ) -> TSRTemplate:
        """Generate a top-down grasp template for a planar surface.

        The gripper approaches from above (along -z of surface frame) and can
        land anywhere within the x/y range on the surface, at any yaw angle.

        Args:
            x_range: (x_min, x_max) reachable x range on surface in meters.
            y_range: (y_min, y_max) reachable y range on surface in meters.
            subject: Subject entity string.
            reference: Reference entity string.
            name: Template name.
            description: Template description.

        Returns:
            TSRTemplate for planar top-down grasp.
        """
        if not name:
            name = f"{reference.title()} Top Grasp"
        if not description:
            description = f"Top-down grasp on {reference} surface"

        T_ref_tsr = np.eye(4)

        # EE z-axis = -z_TSR (downward = approach toward surface).
        # EE y-axis = y_TSR (finger opening direction in surface plane).
        # EE x-axis = -x_TSR (palm normal).
        #
        # Columns: col0=[-1,0,0], col1=[0,1,0], col2=[0,0,-1]
        Tw_e = np.array([
            [-1.,  0.,  0., 0.                ],
            [ 0.,  1.,  0., 0.                ],
            [ 0.,  0., -1., self.finger_length ],
            [ 0.,  0.,  0., 1.                ],
        ])

        Bw = np.array([
            [x_range[0], x_range[1]],  # x: range on surface
            [y_range[0], y_range[1]],  # y: range on surface
            [0.,         0.         ],  # z: fixed
            [0.,         0.         ],  # roll: fixed
            [0.,         0.         ],  # pitch: fixed
            [-np.pi,     np.pi      ],  # yaw: any approach angle
        ])

        return TSRTemplate(
            T_ref_tsr=T_ref_tsr,
            Tw_e=Tw_e,
            Bw=Bw,
            task="grasp",
            subject=subject,
            reference=reference,
            name=name,
            description=description,
        )


def main():
    """Demonstrate ParallelJawGripper for a Robotiq 2F-140."""
    # Robotiq 2F-140: 140mm max aperture, ~100mm finger length
    gripper = ParallelJawGripper(finger_length=0.10, max_aperture=0.14)

    print("ParallelJawGripper (Robotiq 2F-140 approximation)")
    print(f"  finger_length = {gripper.finger_length*1000:.0f}mm")
    print(f"  max_aperture  = {gripper.max_aperture*1000:.0f}mm")
    print()

    # --- Can grasp ---
    can_radius = 0.033   # 330ml beverage can
    can_height = 0.115
    can_grasp = gripper.grasp_cylinder(
        object_radius=can_radius,
        height_range=(0.03, can_height - 0.03),
        subject="robotiq_2f140",
        reference="can",
    )
    print(f"Can side grasp: {can_grasp.name}")
    print(f"  task={can_grasp.task}  subject={can_grasp.subject}  reference={can_grasp.reference}")
    print(f"  Tw_e:\n{np.array2string(can_grasp.Tw_e, precision=4, suppress_small=True)}")
    print(f"  Bw:\n{np.array2string(can_grasp.Bw, precision=4, suppress_small=True)}")
    print(f"  preshape (aperture): {can_grasp.preshape[0]*1000:.0f}mm")
    print()

    # --- Box grasp ---
    box_grasp = gripper.grasp_box_face(
        face_width=0.08,
        face_height=0.12,
        face_depth=0.20,
        subject="robotiq_2f140",
        reference="cereal_box",
    )
    print(f"Box front grasp: {box_grasp.name}")
    print(f"  task={box_grasp.task}  preshape: {box_grasp.preshape[0]*1000:.0f}mm")
    print()

    # --- Instantiate at a concrete object pose and sample ---
    object_pose = np.array([
        [1., 0., 0., 0.5],
        [0., 1., 0., 0.2],
        [0., 0., 1., 0.8],
        [0., 0., 0., 1. ],
    ])
    tsr = can_grasp.instantiate(object_pose)
    sample = tsr.sample()
    print(f"Sampled grasp pose (can at [0.5, 0.2, 0.8]):")
    print(f"{np.array2string(sample, precision=4, suppress_small=True)}")

    R = sample[:3, :3]
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-6), "Not orthogonal"
    assert np.allclose(np.linalg.det(R), 1., atol=1e-6), "Not special orthogonal"
    print("\nValid SE(3) transform confirmed.")


if __name__ == "__main__":
    main()
