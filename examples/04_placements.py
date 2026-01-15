#!/usr/bin/env python
"""
Placements Example: Using reference_frame for placement constraints.

This example demonstrates:
- Loading placement templates with reference_frame
- Understanding the difference between object origin and contact surface
- Transforming poses to match template expectations
"""

import numpy as np
from pathlib import Path

from tsr.core.tsr_primitive import load_template_file
from tsr.core.tsr import TSR


def get_templates_dir() -> Path:
    """Get the templates directory relative to this script."""
    return Path(__file__).parent.parent / "templates"


def main():
    print("Placements Example")
    print("=" * 50)

    templates_dir = get_templates_dir()

    # -------------------------------------------------------------------------
    # The reference_frame problem
    # -------------------------------------------------------------------------
    print("\nThe reference_frame problem")
    print("-" * 30)
    print("""
   When placing a mug on a table, the constraint is about where the
   mug's BOTTOM touches the table, not where the mug's CENTER is.

   If your perception system gives you the mug's center-of-mass (COM),
   you need to transform it to the bottom frame before using a
   placement template.

   The reference_frame field tells you which frame the template expects.
""")

    # -------------------------------------------------------------------------
    # Load a placement template
    # -------------------------------------------------------------------------
    print("1. Load placement template")
    print("-" * 30)

    template = load_template_file(str(templates_dir / "places/mug_on_table.yaml"))
    print(f"   Name: {template.name}")
    print(f"   Reference frame: {template.reference_frame}")
    print(f"   (This template expects the mug's bottom frame)")

    # -------------------------------------------------------------------------
    # Scenario: Perception gives COM, template needs bottom
    # -------------------------------------------------------------------------
    print("\n2. Transform from COM to bottom")
    print("-" * 30)

    # Perception gives us the mug's COM pose
    mug_com_pose = np.eye(4)
    mug_com_pose[0:3, 3] = [0.5, 0.3, 0.15]  # COM at z=0.15m
    print(f"   Mug COM: [{mug_com_pose[0,3]:.2f}, {mug_com_pose[1,3]:.2f}, {mug_com_pose[2,3]:.2f}]")

    # The mug is 10cm tall, so bottom is 5cm below COM
    mug_height = 0.10
    T_com_to_bottom = np.eye(4)
    T_com_to_bottom[2, 3] = -mug_height / 2  # Bottom is below COM

    # Transform to bottom frame
    mug_bottom_pose = mug_com_pose @ T_com_to_bottom
    print(f"   Mug bottom: [{mug_bottom_pose[0,3]:.2f}, {mug_bottom_pose[1,3]:.2f}, {mug_bottom_pose[2,3]:.2f}]")

    # -------------------------------------------------------------------------
    # Create TSR and sample placements
    # -------------------------------------------------------------------------
    print("\n3. Sample valid placement poses")
    print("-" * 30)

    # Use the BOTTOM frame (as the template expects)
    tsr = TSR(T0_w=mug_bottom_pose, Tw_e=template.Tw_e, Bw=template.Bw)

    print("   Valid placement positions (bottom frame):")
    for i in range(3):
        pose = tsr.sample()
        pos = pose[0:3, 3]
        print(f"      {i+1}: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")

    # -------------------------------------------------------------------------
    # Different placement templates
    # -------------------------------------------------------------------------
    print("\n4. Other placement templates")
    print("-" * 30)

    # Cup on coaster (also needs bottom frame)
    template = load_template_file(str(templates_dir / "places/cup_on_coaster.yaml"))
    print(f"   {template.name}")
    print(f"      reference_frame: {template.reference_frame}")
    print(f"      (disk primitive - must be centered)")

    # Bottle in rack (needs side frame since bottle lies down)
    template = load_template_file(str(templates_dir / "places/bottle_in_rack.yaml"))
    print(f"   {template.name}")
    print(f"      reference_frame: {template.reference_frame}")
    print(f"      (line primitive - bottle on its side)")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("Summary:")
    print("-" * 50)
    print("""
   - Placement templates specify reference_frame to indicate which
     part of the object contacts the surface

   - No reference_frame = object origin (default)

   - reference_frame: bottom = bottom contact surface

   - reference_frame: side = side contact surface

   - Transform your object pose to match before creating the TSR
""")

    print("Done!")


if __name__ == "__main__":
    main()
