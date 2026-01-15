#!/usr/bin/env python
"""
Templates Example: Using human-readable YAML templates.

This example demonstrates:
- Loading templates from YAML files
- Understanding the 9 geometric primitives
- Creating TSRs from templates at object poses
- Sampling valid grasp/placement poses
"""

import numpy as np
from pathlib import Path

from tsr.core.tsr_primitive import load_template_file
from tsr.core.tsr import TSR


def get_templates_dir() -> Path:
    """Get the templates directory relative to this script."""
    return Path(__file__).parent.parent / "templates"


def main():
    print("Templates Example")
    print("=" * 50)

    templates_dir = get_templates_dir()

    # -------------------------------------------------------------------------
    # 1. Load a grasp template (cylinder primitive)
    # -------------------------------------------------------------------------
    print("\n1. Cylinder grasp template")
    print("-" * 30)

    template = load_template_file(str(templates_dir / "grasps/mug_side_grasp.yaml"))
    print(f"   Name: {template.name}")
    print(f"   Description: {template.description}")
    print(f"   Task: {template.task}")
    print(f"   Subject: {template.subject} -> Reference: {template.reference}")

    # Create TSR at a specific mug pose
    mug_pose = np.eye(4)
    mug_pose[0:3, 3] = [0.5, 0.3, 0.1]  # Mug on table

    tsr = TSR(T0_w=mug_pose, Tw_e=template.Tw_e, Bw=template.Bw)

    # Sample grasp poses
    print(f"   Mug at: [{mug_pose[0,3]:.2f}, {mug_pose[1,3]:.2f}, {mug_pose[2,3]:.2f}]")
    print(f"   Valid grasp poses:")
    for i in range(3):
        pose = tsr.sample()
        print(f"      {i+1}: [{pose[0,3]:.3f}, {pose[1,3]:.3f}, {pose[2,3]:.3f}]")

    # -------------------------------------------------------------------------
    # 2. Load a ring grasp template (bowl rim)
    # -------------------------------------------------------------------------
    print("\n2. Ring grasp template")
    print("-" * 30)

    template = load_template_file(str(templates_dir / "grasps/bowl_rim_grasp.yaml"))
    print(f"   Name: {template.name}")
    print(f"   Primitive: ring (grasp rim from any angle)")

    bowl_pose = np.eye(4)
    bowl_pose[0:3, 3] = [0.4, 0.2, 0.05]

    tsr = TSR(T0_w=bowl_pose, Tw_e=template.Tw_e, Bw=template.Bw)
    pose = tsr.sample()
    print(f"   Bowl at: [{bowl_pose[0,3]:.2f}, {bowl_pose[1,3]:.2f}, {bowl_pose[2,3]:.2f}]")
    print(f"   Grasp pose: [{pose[0,3]:.3f}, {pose[1,3]:.3f}, {pose[2,3]:.3f}]")

    # -------------------------------------------------------------------------
    # 3. Load a line grasp template (knife handle)
    # -------------------------------------------------------------------------
    print("\n3. Line grasp template")
    print("-" * 30)

    template = load_template_file(str(templates_dir / "grasps/knife_handle_grasp.yaml"))
    print(f"   Name: {template.name}")
    print(f"   Primitive: line (slide along handle)")

    knife_pose = np.eye(4)
    knife_pose[0:3, 3] = [0.3, 0.1, 0.02]

    tsr = TSR(T0_w=knife_pose, Tw_e=template.Tw_e, Bw=template.Bw)
    pose = tsr.sample()
    print(f"   Knife at: [{knife_pose[0,3]:.2f}, {knife_pose[1,3]:.2f}, {knife_pose[2,3]:.2f}]")
    print(f"   Grasp pose: [{pose[0,3]:.3f}, {pose[1,3]:.3f}, {pose[2,3]:.3f}]")

    # -------------------------------------------------------------------------
    # 4. Load a task template (valve turning)
    # -------------------------------------------------------------------------
    print("\n4. Task template (valve turning)")
    print("-" * 30)

    template = load_template_file(str(templates_dir / "tasks/turn_valve.yaml"))
    print(f"   Name: {template.name}")
    print(f"   Primitive: ring (grip valve wheel rim)")

    valve_pose = np.eye(4)
    valve_pose[0:3, 3] = [0.6, 0.0, 0.8]

    tsr = TSR(T0_w=valve_pose, Tw_e=template.Tw_e, Bw=template.Bw)
    pose = tsr.sample()
    print(f"   Valve at: [{valve_pose[0,3]:.2f}, {valve_pose[1,3]:.2f}, {valve_pose[2,3]:.2f}]")
    print(f"   Grip pose: [{pose[0,3]:.3f}, {pose[1,3]:.3f}, {pose[2,3]:.3f}]")

    # -------------------------------------------------------------------------
    # Summary of the 9 primitives
    # -------------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("The 9 Geometric Primitives:")
    print("-" * 50)
    primitives = [
        ("point", "Fixed location", "Precise grasp point"),
        ("line", "Along one axis", "Knife handle, drawer"),
        ("plane", "Flat 2D region", "Table placement"),
        ("box", "3D volume", "Tolerance region"),
        ("ring", "Circle around axis", "Valve wheel, bowl rim"),
        ("disk", "Filled circle", "Coaster placement"),
        ("cylinder", "Cylinder surface", "Mug side grasp"),
        ("shell", "Thick cylinder", "Jar lid rim"),
        ("sphere", "Spherical surface", "Handover zone"),
    ]
    for name, desc, use_case in primitives:
        print(f"   {name:10} - {desc:20} ({use_case})")

    print("\nDone!")


if __name__ == "__main__":
    main()
