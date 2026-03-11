#!/usr/bin/env python
"""
Basic TSR Example: Core TSR creation and usage.

This example demonstrates fundamental TSR operations:
- Creating a TSR with bounds
- Sampling poses from the TSR
- Checking if poses are within the TSR
- Computing distances to the TSR
"""

import numpy as np
from numpy import pi

from tsr import TSR


def main():
    print("Basic TSR Example")
    print("=" * 50)

    # A TSR is defined by three components:
    # - T0_w: Transform from world to TSR frame (where is the object?)
    # - Tw_e: Transform from TSR frame to end-effector (gripper orientation)
    # - Bw: 6x2 bounds matrix [x, y, z, roll, pitch, yaw]

    # Example: Grasp a mug from the side
    # Place the TSR frame at the mug's position
    T0_w = np.eye(4)
    T0_w[0:3, 3] = [0.5, 0.0, 0.3]  # Mug at x=0.5, z=0.3

    # Gripper approaches from the side (-x direction), offset by 5cm
    Tw_e = np.array([
        [0, 0, 1, -0.05],   # gripper z-axis points toward mug
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    ])

    # Bounds: allow any rotation around the mug (yaw), small z tolerance
    Bw = np.zeros((6, 2))
    Bw[2, :] = [-0.02, 0.02]      # z: +/- 2cm
    Bw[5, :] = [-pi, pi]          # yaw: full rotation around mug

    # Create the TSR
    tsr = TSR(T0_w=T0_w, Tw_e=Tw_e, Bw=Bw)

    # Sample valid grasp poses
    print("\n1. Sampling poses")
    print("-" * 30)
    for i in range(3):
        pose = tsr.sample()
        pos = pose[0:3, 3]
        print(f"   Sample {i+1}: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")

    # Check if a pose is valid
    print("\n2. Containment check")
    print("-" * 30)
    sampled_pose = tsr.sample()
    print(f"   Sampled pose valid: {tsr.contains(sampled_pose)}")

    random_pose = np.eye(4)
    random_pose[0:3, 3] = [1.0, 1.0, 1.0]  # Far away
    print(f"   Random pose valid: {tsr.contains(random_pose)}")

    # Compute distance to TSR
    print("\n3. Distance computation")
    print("-" * 30)
    distance, closest = tsr.distance(random_pose)
    print(f"   Distance to random pose: {distance:.3f}")

    distance, closest = tsr.distance(sampled_pose)
    print(f"   Distance to sampled pose: {distance:.3f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
