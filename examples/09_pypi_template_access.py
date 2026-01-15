#!/usr/bin/env python
"""
Template Access Example: How to use templates from the package.

This example demonstrates how users can access TSR templates included
in the package.
"""

import numpy as np
from pathlib import Path

from tsr.core.tsr_primitive import load_template_file
from tsr.core.tsr import TSR


def get_templates_dir() -> Path:
    """Get the path to the templates directory."""
    # Templates are in the package root
    import tsr
    package_dir = Path(tsr.__file__).parent.parent.parent
    return package_dir / "templates"


def list_templates(category: str = None) -> list:
    """List available templates, optionally filtered by category."""
    templates_dir = get_templates_dir()
    if category:
        search_dir = templates_dir / category
        if search_dir.exists():
            return list(search_dir.glob("*.yaml"))
        return []
    else:
        return list(templates_dir.glob("**/*.yaml"))


def demonstrate_template_discovery():
    """Demonstrate discovering available templates."""
    print("\n1. Template Discovery")
    print("=" * 50)

    templates_dir = get_templates_dir()
    print(f"Templates directory: {templates_dir}")

    # List templates by category
    for category in ["grasps", "places", "tasks"]:
        templates = list_templates(category)
        print(f"\n{category}/ ({len(templates)} templates):")
        for t in templates:
            print(f"   - {t.name}")


def demonstrate_template_loading():
    """Demonstrate loading templates."""
    print("\n2. Template Loading")
    print("=" * 50)

    templates_dir = get_templates_dir()

    # Load a grasp template
    grasp_template = load_template_file(str(templates_dir / "grasps/mug_side_grasp.yaml"))
    print(f"Loaded: {grasp_template.name}")
    print(f"  Description: {grasp_template.description}")
    print(f"  Task: {grasp_template.task}")
    print(f"  Subject: {grasp_template.subject}")
    print(f"  Reference: {grasp_template.reference}")

    # Load a placement template
    place_template = load_template_file(str(templates_dir / "places/mug_on_table.yaml"))
    print(f"\nLoaded: {place_template.name}")
    print(f"  Description: {place_template.description}")
    print(f"  Reference frame: {place_template.reference_frame}")


def demonstrate_template_usage():
    """Demonstrate using templates to create TSRs."""
    print("\n3. Template Usage")
    print("=" * 50)

    templates_dir = get_templates_dir()

    # Load template
    template = load_template_file(str(templates_dir / "grasps/mug_side_grasp.yaml"))

    # Create TSR at object pose
    mug_pose = np.array([
        [1, 0, 0, 0.5],   # Mug at x=0.5m
        [0, 1, 0, 0.3],   # y=0.3m
        [0, 0, 1, 0.1],   # z=0.1m
        [0, 0, 0, 1]
    ])

    tsr = TSR(T0_w=mug_pose, Tw_e=template.Tw_e, Bw=template.Bw)

    # Sample poses
    print(f"Using template: {template.name}")
    print(f"Mug pose: [{mug_pose[0,3]:.2f}, {mug_pose[1,3]:.2f}, {mug_pose[2,3]:.2f}]")
    print(f"Sampled grasp poses:")
    for i in range(3):
        pose = tsr.sample()
        print(f"  {i+1}: [{pose[0,3]:.3f}, {pose[1,3]:.3f}, {pose[2,3]:.3f}]")


def demonstrate_placement_with_reference_frame():
    """Demonstrate using reference_frame for placements."""
    print("\n4. Placement with Reference Frame")
    print("=" * 50)

    templates_dir = get_templates_dir()

    # Load placement template
    template = load_template_file(str(templates_dir / "places/mug_on_table.yaml"))
    print(f"Template: {template.name}")
    print(f"Reference frame: {template.reference_frame}")

    # The template expects the mug's bottom frame
    # If perception gives COM, transform accordingly
    mug_com_pose = np.eye(4)
    mug_com_pose[0:3, 3] = [0.5, 0.3, 0.1]  # COM position

    # Transform from COM to bottom (assuming mug is 10cm tall)
    T_com_to_bottom = np.eye(4)
    T_com_to_bottom[2, 3] = -0.05  # Bottom is 5cm below COM

    mug_bottom_pose = mug_com_pose @ T_com_to_bottom

    print(f"Mug COM: [{mug_com_pose[0,3]:.2f}, {mug_com_pose[1,3]:.2f}, {mug_com_pose[2,3]:.2f}]")
    print(f"Mug bottom: [{mug_bottom_pose[0,3]:.2f}, {mug_bottom_pose[1,3]:.2f}, {mug_bottom_pose[2,3]:.2f}]")

    # Create TSR with bottom frame (as template expects)
    tsr = TSR(T0_w=mug_bottom_pose, Tw_e=template.Tw_e, Bw=template.Bw)

    # Sample valid placement pose
    pose = tsr.sample()
    print(f"Valid placement: [{pose[0,3]:.3f}, {pose[1,3]:.3f}, {pose[2,3]:.3f}]")


def main():
    """Run template access examples."""
    print("Template Access Example")
    print("=" * 60)

    demonstrate_template_discovery()
    demonstrate_template_loading()
    demonstrate_template_usage()
    demonstrate_placement_with_reference_frame()

    print("\n" + "=" * 60)
    print("Template access example completed!")


if __name__ == "__main__":
    main()
