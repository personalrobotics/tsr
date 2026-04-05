#!/usr/bin/env python
"""
Templates Example: Loading and using YAML templates for manipulation narratives.

This example demonstrates:
- Loading templates from the package library
- Binding templates to object poses at runtime
- Sampling valid gripper poses for each step of a task
"""

import numpy as np

from tsr import load_package_template


def print_section(title):
    print(f"\n{title}")
    print("-" * len(title))


def show_template(label, category, filename, object_pose, object_label):
    t = load_package_template(category, filename)
    tsr = t.instantiate(object_pose)
    pose = tsr.sample()
    pos = pose[:3, 3]
    print(f"  {label}")
    print(f"    {t.name}")
    print(f"    {object_label}: {object_pose[:3, 3].tolist()}")
    print(f"    sample: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")


def main():
    print("Templates Example")
    print("=" * 50)

    # -------------------------------------------------------------------------
    # Narrative 1: Tool Use — Screwdriver
    # Grasp screwdriver -> drive screw -> drop in toolchest
    # -------------------------------------------------------------------------
    print_section("Narrative 1: Tool Use -- Screwdriver")

    screwdriver_pose = np.eye(4)
    screwdriver_pose[:3, 3] = [0.4, 0.1, 0.02]

    screw_pose = np.eye(4)
    screw_pose[:3, 3] = [0.5, 0.3, 0.0]

    toolchest_pose = np.eye(4)
    toolchest_pose[:3, 3] = [0.2, -0.4, 0.5]

    show_template("1. Grasp", "grasps", "screwdriver_grasp.yaml", screwdriver_pose, "screwdriver")
    show_template("2. Drive screw", "tasks", "drive_screw.yaml", screw_pose, "screw")
    show_template(
        "3. Drop in toolchest",
        "places",
        "toolchest_drop.yaml",
        toolchest_pose,
        "toolchest",
    )

    # -------------------------------------------------------------------------
    # Narrative 2: Everyday Manipulation — Mug of Water
    # Grasp mug -> transport upright -> pour into sink -> place on table
    # -------------------------------------------------------------------------
    print_section("Narrative 2: Everyday Manipulation -- Mug of Water")

    mug_pose = np.eye(4)
    mug_pose[:3, 3] = [0.5, 0.0, 0.0]

    sink_pose = np.eye(4)
    sink_pose[:3, 3] = [0.8, 0.0, 0.9]

    table_pose = np.eye(4)
    table_pose[:3, 3] = [0.6, 0.3, 0.75]

    show_template("1. Grasp mug", "grasps", "mug_handle_grasp.yaml", mug_pose, "mug")
    show_template("2. Transport upright", "tasks", "mug_transport_upright.yaml", mug_pose, "mug")
    show_template("3. Pour into sink", "tasks", "mug_pour_into_sink.yaml", sink_pose, "sink")
    show_template("4. Place on table", "places", "mug_on_table.yaml", table_pose, "table")

    print("\nDone!")


if __name__ == "__main__":
    main()
