#!/usr/bin/env python
"""
TSR Templates Example: Reusable, scene-agnostic TSR definitions.

This example demonstrates TSR templates for reusable TSR definitions:
- Creating templates for different object types
- Instantiating templates at specific poses
- Reusing templates across different scenes
- Examples: Cylindrical grasp and surface placement
"""

import numpy as np
from numpy import pi

from tsr import TSRTemplate


def main():
    """Demonstrate TSR templates for reusable definitions."""
    print("=== TSR Template Example ===")
    
    # Create a template for grasping cylindrical objects
    cylinder_grasp_template = TSRTemplate(
        T_ref_tsr=np.eye(4),  # TSR frame aligned with cylinder frame
        Tw_e=np.array([
            [0, 0, 1, -0.05],  # Approach from -z, 5cm offset
            [1, 0, 0, 0],      # x-axis perpendicular to cylinder
            [0, 1, 0, 0.05],   # y-axis along cylinder axis
            [0, 0, 0, 1]
        ]),
        Bw=np.array([
            [0, 0],           # x: fixed position
            [0, 0],           # y: fixed position
            [-0.01, 0.01],    # z: small tolerance
            [0, 0],           # roll: fixed
            [0, 0],           # pitch: fixed
            [-pi, pi]         # yaw: full rotation
        ])
    )
    
    # Create a template for placing objects on surfaces
    surface_place_template = TSRTemplate(
        T_ref_tsr=np.eye(4),  # TSR frame aligned with surface frame
        Tw_e=np.array([
            [1, 0, 0, 0],      # Object x-axis aligned with surface
            [0, 1, 0, 0],      # Object y-axis aligned with surface
            [0, 0, 1, 0.02],   # Object 2cm above surface
            [0, 0, 0, 1]
        ]),
        Bw=np.array([
            [-0.1, 0.1],       # x: allow sliding on surface
            [-0.1, 0.1],       # y: allow sliding on surface
            [0, 0],            # z: fixed height
            [0, 0],            # roll: keep level
            [0, 0],            # pitch: keep level
            [-pi/4, pi/4]      # yaw: allow some rotation
        ])
    )
    
    # Instantiate templates at specific poses
    mug_pose = np.array([
        [1, 0, 0, 0.5],  # Mug at x=0.5
        [0, 1, 0, 0.0],
        [0, 0, 1, 0.3],
        [0, 0, 0, 1]
    ])
    
    table_pose = np.array([
        [1, 0, 0, 0.0],  # Table at origin
        [0, 1, 0, 0.0],
        [0, 0, 1, 0.0],
        [0, 0, 0, 1]
    ])
    
    # Create concrete TSRs
    mug_grasp_tsr = cylinder_grasp_template.instantiate(mug_pose)
    table_place_tsr = surface_place_template.instantiate(table_pose)
    
    # Sample poses
    grasp_pose = mug_grasp_tsr.sample()
    place_pose = table_place_tsr.sample()
    
    print(f"Mug grasp pose:\n{grasp_pose}")
    print(f"Table place pose:\n{place_pose}")
    
    # Demonstrate reusability: instantiate at different poses
    bottle_pose = np.array([
        [1, 0, 0, 0.8],  # Bottle at x=0.8
        [0, 1, 0, 0.2],
        [0, 0, 1, 0.4],
        [0, 0, 0, 1]
    ])
    
    bottle_grasp_tsr = cylinder_grasp_template.instantiate(bottle_pose)
    bottle_grasp_pose = bottle_grasp_tsr.sample()
    
    print(f"Bottle grasp pose:\n{bottle_grasp_pose}")
    
    print()


if __name__ == "__main__":
    main()
