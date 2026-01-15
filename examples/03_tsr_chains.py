#!/usr/bin/env python
"""
TSR Chains Example: Composing multiple TSRs for articulated constraints.

This example demonstrates:
- Chaining TSRs where each frame is relative to the previous
- Example: Opening a door (handle attached to door, door hinged to wall)
"""

import numpy as np
from numpy import pi

from tsr import TSR, TSRChain


def main():
    print("TSR Chains Example")
    print("=" * 50)

    # -------------------------------------------------------------------------
    # Example: Opening a refrigerator door
    #
    # The constraint hierarchy:
    #   World -> Hinge (fixed) -> Door (rotates) -> Handle -> Gripper
    #
    # We chain two TSRs:
    #   1. Door rotation around hinge (yaw freedom)
    #   2. Gripper pose relative to handle (small tolerance)
    # -------------------------------------------------------------------------

    print("\nDoor opening constraint chain")
    print("-" * 30)

    # TSR 1: Door hinge constraint
    # The hinge is fixed in the world, door can rotate around it
    hinge_pose = np.eye(4)
    hinge_pose[0:3, 3] = [0.0, 0.0, 0.8]  # Hinge at z=0.8m

    # Handle is 60cm from hinge along the door
    handle_offset = np.array([
        [1, 0, 0, 0.6],   # Handle 60cm from hinge
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    # Door can open from 0 to 90 degrees
    hinge_bounds = np.zeros((6, 2))
    hinge_bounds[5, :] = [0, pi/2]  # yaw: 0 to 90 degrees

    hinge_tsr = TSR(T0_w=hinge_pose, Tw_e=handle_offset, Bw=hinge_bounds)

    # TSR 2: Gripper constraint relative to handle
    # Gripper approaches handle from the front
    gripper_offset = np.array([
        [0, 0, 1, -0.05],  # Approach from -z, 5cm offset
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    ])

    # Small tolerance in gripper position
    gripper_bounds = np.zeros((6, 2))
    gripper_bounds[2, :] = [-0.01, 0.01]   # z: +/- 1cm
    gripper_bounds[5, :] = [-pi/6, pi/6]   # yaw: +/- 30 degrees

    gripper_tsr = TSR(T0_w=np.eye(4), Tw_e=gripper_offset, Bw=gripper_bounds)

    # Chain the TSRs together
    door_chain = TSRChain(TSRs=[hinge_tsr, gripper_tsr])

    print(f"   Chain has {len(door_chain.TSRs)} TSRs")

    # Sample poses along the door opening trajectory
    print("\n   Sampled gripper poses while opening door:")
    for i in range(4):
        pose = door_chain.sample()
        pos = pose[0:3, 3]
        print(f"      {i+1}: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")

    # Check if a pose satisfies the chain
    print("\n   Containment check:")
    test_pose = door_chain.sample()
    is_valid = door_chain.contains(test_pose)
    print(f"      Sampled pose valid: {is_valid}")

    # Compute distance
    distance, closest = door_chain.distance(test_pose)
    print(f"      Distance to chain: {distance:.4f}")

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------
    print("\n   Serialization:")
    chain_dict = door_chain.to_dict()
    print(f"      Serialized to dict with {len(chain_dict['tsrs'])} TSRs")

    reconstructed = TSRChain.from_dict(chain_dict)
    print(f"      Reconstructed chain has {len(reconstructed.TSRs)} TSRs")

    print("\nDone!")


if __name__ == "__main__":
    main()
