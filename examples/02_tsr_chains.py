#!/usr/bin/env python
"""
TSR Chains Example: Complex constraints with multiple TSRs.

This example demonstrates TSR chains for complex manipulation tasks:
- Creating multiple TSRs for different constraints
- Composing them into a chain
- Sampling poses from the chain
- Example: Opening a refrigerator door
"""

import numpy as np
from numpy import pi

from tsr import TSR, TSRChain


def main():
    """Demonstrate TSR chains for complex constraints."""
    print("=== TSR Chain Example ===")
    
    # Example: Opening a refrigerator door
    # First TSR: handle constraint relative to hinge
    hinge_pose = np.eye(4)
    hinge_pose[0:3, 3] = [0.0, 0.0, 0.8]  # Hinge at z=0.8
    
    handle_offset = np.array([
        [1, 0, 0, 0.6],  # Handle 60cm from hinge
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    
    handle_bounds = np.zeros((6, 2))
    handle_bounds[5, :] = [0, pi/2]  # Door opens 90 degrees
    
    hinge_tsr = TSR(T0_w=hinge_pose, Tw_e=handle_offset, Bw=handle_bounds)
    
    # Second TSR: end-effector constraint relative to handle
    ee_in_handle = np.array([
        [0, 0, 1, -0.05],  # Approach handle from -z
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    ])
    
    ee_bounds = np.zeros((6, 2))
    ee_bounds[2, :] = [-0.01, 0.01]  # Small tolerance in approach
    ee_bounds[5, :] = [-pi/6, pi/6]  # Some rotation tolerance
    
    ee_tsr = TSR(T0_w=np.eye(4), Tw_e=ee_in_handle, Bw=ee_bounds)
    
    # Compose into a chain
    door_chain = TSRChain(TSRs=[hinge_tsr, ee_tsr])
    
    # Sample a pose from the chain
    door_pose = door_chain.sample()
    print(f"Door opening pose:\n{door_pose}")
    
    # Check if a pose satisfies the chain
    test_pose = np.eye(4)
    is_valid = door_chain.contains(test_pose)
    print(f"Test pose satisfies chain: {is_valid}")
    
    print()


if __name__ == "__main__":
    main()
