#!/usr/bin/env python
"""
Basic TSR Example: Core TSR creation and usage.

This example demonstrates the fundamental TSR operations:
- Creating a TSR for grasping a glass
- Sampling poses from the TSR
- Checking if poses are within the TSR
- Computing distances to the TSR
"""

import numpy as np
from numpy import pi

from tsr import TSR


def main():
    """Demonstrate basic TSR creation and usage."""
    print("=== Basic TSR Example ===")
    
    # Create a simple TSR for grasping a glass
    T0_w = np.eye(4)  # Glass frame at world origin
    T0_w[0:3, 3] = [0.5, 0.0, 0.3]  # Glass at x=0.5, y=0, z=0.3
    
    # Desired end-effector pose relative to glass
    Tw_e = np.array([
        [0, 0, 1, -0.20],  # Approach from -z, 20cm offset
        [1, 0, 0, 0],      # x-axis perpendicular to glass
        [0, 1, 0, 0.08],   # y-axis along glass height
        [0, 0, 0, 1]
    ])
    
    # Bounds on TSR coordinates
    Bw = np.zeros((6, 2))
    Bw[2, :] = [0.0, 0.02]    # Allow small vertical movement
    Bw[5, :] = [-pi, pi]      # Allow any orientation about z-axis
    
    # Create TSR
    grasp_tsr = TSR(T0_w=T0_w, Tw_e=Tw_e, Bw=Bw)
    
    # Sample a grasp pose
    grasp_pose = grasp_tsr.sample()
    print(f"Sampled grasp pose:\n{grasp_pose}")
    
    # Check if a pose is within the TSR
    current_pose = np.eye(4)
    is_valid = grasp_tsr.contains(current_pose)
    print(f"Current pose is valid: {is_valid}")
    
    # Compute distance to TSR
    distance, closest_point = grasp_tsr.distance(current_pose)
    print(f"Distance to TSR: {distance:.3f}")
    
    print()


if __name__ == "__main__":
    main()
