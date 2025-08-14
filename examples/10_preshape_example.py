#!/usr/bin/env python
"""
Preshape Example: Demonstrating gripper configuration in TSR templates.

This example shows how to use the optional preshape field in TSR templates
to specify gripper configurations (DOF values) that should be achieved
before or during TSR execution.
"""

import numpy as np

from tsr import (
    EntityClass, TaskCategory, TaskType,
    generate_cylinder_grasp_template,
    generate_box_grasp_template,
    generate_mug_grasp_template,
    TSRTemplate,
    save_template
)


def demonstrate_parallel_jaw_preshape():
    """Demonstrate preshape for parallel jaw grippers."""
    print("=== Parallel Jaw Gripper Preshape ===")
    
    # Parallel jaw gripper with 8cm aperture for mug side grasp
    mug_grasp = generate_mug_grasp_template(
        variant="side",
        preshape=np.array([0.08])  # 8cm aperture
    )
    
    print(f"Template: {mug_grasp.name}")
    print(f"Preshape: {mug_grasp.preshape} (aperture in meters)")
    print(f"Description: {mug_grasp.description}")
    print()
    
    # Parallel jaw gripper with 12cm aperture for larger object
    large_grasp = generate_cylinder_grasp_template(
        subject_entity=EntityClass.GENERIC_GRIPPER,
        reference_entity=EntityClass.MUG,
        variant="side",
        cylinder_radius=0.06,
        cylinder_height=0.20,
        preshape=np.array([0.12])  # 12cm aperture for larger mug
    )
    
    print(f"Template: {large_grasp.name}")
    print(f"Preshape: {large_grasp.preshape} (aperture in meters)")
    print()


def demonstrate_multi_finger_preshape():
    """Demonstrate preshape for multi-finger hands."""
    print("=== Multi-Finger Hand Preshape ===")
    
    # 6-DOF hand configuration for precision grasp
    precision_grasp = generate_box_grasp_template(
        subject_entity=EntityClass.GENERIC_GRIPPER,
        reference_entity=EntityClass.BOX,
        variant="side_x",
        box_length=0.15,
        box_width=0.10,
        box_height=0.08,
        preshape=np.array([0.0, 0.5, 0.5, 0.0, 0.5, 0.5])  # 6-DOF hand configuration
    )
    
    print(f"Template: {precision_grasp.name}")
    print(f"Preshape: {precision_grasp.preshape} (6-DOF hand configuration)")
    print(f"Description: {precision_grasp.description}")
    print()
    
    # 3-finger hand configuration for power grasp
    power_grasp = generate_cylinder_grasp_template(
        subject_entity=EntityClass.GENERIC_GRIPPER,
        reference_entity=EntityClass.MUG,
        variant="side",
        cylinder_radius=0.05,
        cylinder_height=0.15,
        preshape=np.array([0.8, 0.8, 0.8])  # 3-finger power grasp
    )
    
    print(f"Template: {power_grasp.name}")
    print(f"Preshape: {power_grasp.preshape} (3-finger power grasp)")
    print()


def demonstrate_no_preshape():
    """Demonstrate templates without preshape (default behavior)."""
    print("=== No Preshape (Default) ===")
    
    # Template without preshape - gripper configuration not specified
    place_template = TSRTemplate(
        T_ref_tsr=np.eye(4),
        Tw_e=np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0.02],
            [0, 0, 0, 1]
        ]),
        Bw=np.array([
            [-0.1, 0.1],
            [-0.1, 0.1],
            [0, 0],
            [0, 0],
            [0, 0],
            [-np.pi/4, np.pi/4]
        ]),
        subject_entity=EntityClass.MUG,
        reference_entity=EntityClass.TABLE,
        task_category=TaskCategory.PLACE,
        variant="on",
        name="Table Placement",
        description="Place object on table surface"
        # No preshape specified - will be None
    )
    
    print(f"Template: {place_template.name}")
    print(f"Preshape: {place_template.preshape} (None - no gripper configuration specified)")
    print()


def demonstrate_preshape_serialization():
    """Demonstrate that preshape is properly serialized."""
    print("=== Preshape Serialization ===")
    
    # Create template with preshape
    template = generate_mug_grasp_template(
        variant="side",
        preshape=np.array([0.08])
    )
    
    # Serialize to dict
    template_dict = template.to_dict()
    print(f"Serialized preshape: {template_dict.get('preshape')}")
    
    # Deserialize back to template
    reconstructed = TSRTemplate.from_dict(template_dict)
    print(f"Reconstructed preshape: {reconstructed.preshape}")
    print(f"Preshape arrays equal: {np.array_equal(template.preshape, reconstructed.preshape)}")
    print()


def demonstrate_preshape_in_library():
    """Demonstrate using preshape in the relational library."""
    print("=== Preshape in Relational Library ===")
    
    from tsr import TSRLibraryRelational
    
    library = TSRLibraryRelational()
    
    # Register templates with different preshapes
    template1 = generate_mug_grasp_template(variant="side", preshape=np.array([0.08]))
    library.register_template(
        template1.subject_entity,
        template1.reference_entity,
        TaskType(template1.task_category, template1.variant),
        template1,
        "Small aperture grasp"
    )
    
    template2 = generate_mug_grasp_template(variant="side", preshape=np.array([0.12]))
    library.register_template(
        template2.subject_entity,
        template2.reference_entity,
        TaskType(template2.task_category, template2.variant),
        template2,
        "Large aperture grasp"
    )
    
    # Query templates
    templates = library.query_templates(
        subject=EntityClass.GENERIC_GRIPPER,
        reference=EntityClass.MUG,
        task=TaskType(TaskCategory.GRASP, "side")
    )
    
    print(f"Found {len(templates)} templates:")
    for i, template in enumerate(templates):
        print(f"  {i+1}. {template.name} - Preshape: {template.preshape}")
    print()


def main():
    """Run all preshape demonstrations."""
    print("🤖 TSR Template Preshape Examples")
    print("=" * 50)
    print()
    
    demonstrate_parallel_jaw_preshape()
    demonstrate_multi_finger_preshape()
    demonstrate_no_preshape()
    demonstrate_preshape_serialization()
    demonstrate_preshape_in_library()
    
    print("✅ All preshape examples completed!")


if __name__ == "__main__":
    main()
