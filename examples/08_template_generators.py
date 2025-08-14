#!/usr/bin/env python
"""
Template Generators Example: Generate TSR templates for primitive objects.

This example demonstrates how to use the template generators to create
TSR templates for common primitive objects and tasks.
"""

import numpy as np

from tsr import (
    EntityClass, TaskCategory, TaskType,
    generate_cylinder_grasp_template,
    generate_box_grasp_template,
    generate_place_template,
    generate_transport_template,
    generate_mug_grasp_template,
    generate_box_place_template,
    TSRLibraryRelational,
    save_template
)


def demonstrate_cylinder_grasps():
    """Demonstrate generating cylinder grasp templates."""
    print("\n🔵 Cylinder Grasp Templates")
    print("=" * 50)
    
    # Generate different cylinder grasp variants
    side_grasp = generate_cylinder_grasp_template(
        subject_entity=EntityClass.GENERIC_GRIPPER,
        reference_entity=EntityClass.MUG,
        variant="side",
        cylinder_radius=0.04,
        cylinder_height=0.12,
        approach_distance=0.05
    )
    
    top_grasp = generate_cylinder_grasp_template(
        subject_entity=EntityClass.GENERIC_GRIPPER,
        reference_entity=EntityClass.MUG,
        variant="top",
        cylinder_radius=0.04,
        cylinder_height=0.12,
        approach_distance=0.03
    )
    
    print(f"✅ Generated {side_grasp.name}")
    print(f"   Description: {side_grasp.description}")
    print(f"   Variant: {side_grasp.variant}")
    
    print(f"✅ Generated {top_grasp.name}")
    print(f"   Description: {top_grasp.description}")
    print(f"   Variant: {top_grasp.variant}")
    
    return [side_grasp, top_grasp]


def demonstrate_box_grasps():
    """Demonstrate generating box grasp templates."""
    print("\n📦 Box Grasp Templates")
    print("=" * 50)
    
    # Generate different box grasp variants
    side_x_grasp = generate_box_grasp_template(
        subject_entity=EntityClass.GENERIC_GRIPPER,
        reference_entity=EntityClass.BOX,
        variant="side_x",
        box_length=0.15,
        box_width=0.10,
        box_height=0.08,
        approach_distance=0.05
    )
    
    top_grasp = generate_box_grasp_template(
        subject_entity=EntityClass.GENERIC_GRIPPER,
        reference_entity=EntityClass.BOX,
        variant="top",
        box_length=0.15,
        box_width=0.10,
        box_height=0.08,
        approach_distance=0.03
    )
    
    print(f"✅ Generated {side_x_grasp.name}")
    print(f"   Description: {side_x_grasp.description}")
    print(f"   Variant: {side_x_grasp.variant}")
    
    print(f"✅ Generated {top_grasp.name}")
    print(f"   Description: {top_grasp.description}")
    print(f"   Variant: {top_grasp.variant}")
    
    return [side_x_grasp, top_grasp]


def demonstrate_placement_templates():
    """Demonstrate generating placement templates."""
    print("\n📍 Placement Templates")
    print("=" * 50)
    
    # Generate placement templates
    mug_place = generate_place_template(
        subject_entity=EntityClass.MUG,
        reference_entity=EntityClass.TABLE,
        variant="on",
        surface_height=0.0,
        placement_tolerance=0.1,
        orientation_tolerance=0.2
    )
    
    box_place = generate_place_template(
        subject_entity=EntityClass.BOX,
        reference_entity=EntityClass.SHELF,
        variant="on",
        surface_height=0.5,
        placement_tolerance=0.05,
        orientation_tolerance=0.1
    )
    
    print(f"✅ Generated {mug_place.name}")
    print(f"   Description: {mug_place.description}")
    print(f"   Variant: {mug_place.variant}")
    
    print(f"✅ Generated {box_place.name}")
    print(f"   Description: {box_place.description}")
    print(f"   Variant: {box_place.variant}")
    
    return [mug_place, box_place]


def demonstrate_transport_templates():
    """Demonstrate generating transport templates."""
    print("\n🚚 Transport Templates")
    print("=" * 50)
    
    # Generate transport templates
    upright_transport = generate_transport_template(
        subject_entity=EntityClass.MUG,
        reference_entity=EntityClass.GENERIC_GRIPPER,
        variant="upright",
        roll_epsilon=0.1,
        pitch_epsilon=0.1,
        yaw_epsilon=0.2
    )
    
    print(f"✅ Generated {upright_transport.name}")
    print(f"   Description: {upright_transport.description}")
    print(f"   Variant: {upright_transport.variant}")
    
    return [upright_transport]


def demonstrate_convenience_functions():
    """Demonstrate convenience functions."""
    print("\n🎯 Convenience Functions")
    print("=" * 50)
    
    # Use convenience functions with default parameters
    mug_grasp = generate_mug_grasp_template()
    box_place = generate_box_place_template()
    
    print(f"✅ Generated {mug_grasp.name}")
    print(f"   Description: {mug_grasp.description}")
    print(f"   Default radius: {0.04}m, height: {0.12}m")
    
    print(f"✅ Generated {box_place.name}")
    print(f"   Description: {box_place.description}")
    print(f"   Default placement on table")
    
    return [mug_grasp, box_place]


def demonstrate_library_integration():
    """Demonstrate integrating generators with the relational library."""
    print("\n📚 Library Integration")
    print("=" * 50)
    
    # Create library and register generated templates
    library = TSRLibraryRelational()
    
    # Generate and register templates
    mug_grasp = generate_mug_grasp_template()
    mug_place = generate_place_template(
        subject_entity=EntityClass.MUG,
        reference_entity=EntityClass.TABLE,
        variant="on"
    )
    
    # Register in library
    library.register_template(
        subject=EntityClass.GENERIC_GRIPPER,
        reference=EntityClass.MUG,
        task=TaskType(TaskCategory.GRASP, "side"),
        template=mug_grasp,
        description="Side grasp for mug"
    )
    
    library.register_template(
        subject=EntityClass.MUG,
        reference=EntityClass.TABLE,
        task=TaskType(TaskCategory.PLACE, "on"),
        template=mug_place,
        description="Place mug on table"
    )
    
    # Query available templates
    available = library.list_available_templates()
    print(f"✅ Registered {len(available)} templates in library:")
    for subject, reference, task, description in available:
        print(f"   {subject.value} -> {reference.value} ({task}): {description}")
    
    return library


def demonstrate_template_usage():
    """Demonstrate using generated templates."""
    print("\n🎮 Template Usage")
    print("=" * 50)
    
    # Generate a template
    template = generate_mug_grasp_template()
    
    # Simulate object pose (mug at x=0.5, y=0.3, z=0.1)
    mug_pose = np.array([
        [1, 0, 0, 0.5],   # Mug at x=0.5m
        [0, 1, 0, 0.3],   # y=0.3m
        [0, 0, 1, 0.1],   # z=0.1m (on table)
        [0, 0, 0, 1]
    ])
    
    # Instantiate template at mug pose
    tsr = template.instantiate(mug_pose)
    
    # Sample valid poses
    poses = [tsr.sample() for _ in range(3)]
    
    print(f"✅ Generated {template.name}")
    print(f"   Instantiated at mug pose: [{mug_pose[0,3]:.3f}, {mug_pose[1,3]:.3f}, {mug_pose[2,3]:.3f}]")
    print(f"   Sampled poses:")
    for i, pose in enumerate(poses):
        print(f"     {i+1}: [{pose[0,3]:.3f}, {pose[1,3]:.3f}, {pose[2,3]:.3f}]")


def main():
    """Demonstrate all template generator functionality."""
    print("TSR Template Generators Example")
    print("=" * 60)
    
    # Demonstrate different generator types
    cylinder_templates = demonstrate_cylinder_grasps()
    box_templates = demonstrate_box_grasps()
    placement_templates = demonstrate_placement_templates()
    transport_templates = demonstrate_transport_templates()
    convenience_templates = demonstrate_convenience_functions()
    
    # Demonstrate library integration
    library = demonstrate_library_integration()
    
    # Demonstrate template usage
    demonstrate_template_usage()
    
    # Summary
    all_templates = (cylinder_templates + box_templates + 
                    placement_templates + transport_templates + 
                    convenience_templates)
    
    print(f"\n🎯 Summary")
    print("=" * 50)
    print(f"✅ Generated {len(all_templates)} TSR templates")
    print(f"✅ All templates are simulator-agnostic")
    print(f"✅ All templates include semantic context")
    print(f"✅ All templates are compatible with the relational library")
    print(f"✅ All templates support YAML serialization")
    
    print(f"\n📋 Template Types Generated:")
    print(f"   - Cylinder grasps: {len(cylinder_templates)}")
    print(f"   - Box grasps: {len(box_templates)}")
    print(f"   - Placement: {len(placement_templates)}")
    print(f"   - Transport: {len(transport_templates)}")
    print(f"   - Convenience: {len(convenience_templates)}")


if __name__ == "__main__":
    main()
