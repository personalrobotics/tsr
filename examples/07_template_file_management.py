#!/usr/bin/env python
"""
Template File Management Example: Multiple files approach.

This example demonstrates the recommended approach of using one YAML file
per TSR template for better version control, collaboration, and maintainability.
"""

import numpy as np
import tempfile
from pathlib import Path

from tsr import (
    TSRTemplate, EntityClass, TaskCategory, TaskType,
    TemplateIO, save_template, load_template
)


def create_sample_templates():
    """Create sample TSR templates for demonstration."""
    
    # Mug side grasp template
    mug_side_grasp = TSRTemplate(
        T_ref_tsr=np.eye(4),
        Tw_e=np.array([
            [0, 0, 1, -0.05],  # Approach from -z, 5cm offset
            [1, 0, 0, 0],      # x-axis perpendicular to mug
            [0, 1, 0, 0.05],   # y-axis along mug axis
            [0, 0, 0, 1]
        ]),
        Bw=np.array([
            [0, 0],           # x: fixed position
            [0, 0],           # y: fixed position
            [-0.01, 0.01],    # z: small tolerance
            [0, 0],           # roll: fixed
            [0, 0],           # pitch: fixed
            [-np.pi, np.pi]   # yaw: full rotation
        ]),
        subject_entity=EntityClass.GENERIC_GRIPPER,
        reference_entity=EntityClass.MUG,
        task_category=TaskCategory.GRASP,
        variant="side",
        name="Mug Side Grasp",
        description="Grasp mug from the side with 5cm approach distance"
    )
    
    # Mug top grasp template
    mug_top_grasp = TSRTemplate(
        T_ref_tsr=np.eye(4),
        Tw_e=np.array([
            [0, 0, 1, 0],      # Approach from -z, no offset
            [1, 0, 0, 0],      # x-axis perpendicular to mug
            [0, 1, 0, 0],      # y-axis along mug axis
            [0, 0, 0, 1]
        ]),
        Bw=np.array([
            [-0.01, 0.01],    # x: small tolerance
            [-0.01, 0.01],    # y: small tolerance
            [0, 0],           # z: fixed position
            [0, 0],           # roll: fixed
            [0, 0],           # pitch: fixed
            [-np.pi, np.pi]   # yaw: full rotation
        ]),
        subject_entity=EntityClass.GENERIC_GRIPPER,
        reference_entity=EntityClass.MUG,
        task_category=TaskCategory.GRASP,
        variant="top",
        name="Mug Top Grasp",
        description="Grasp mug from the top with vertical approach"
    )
    
    # Mug place on table template
    mug_place_table = TSRTemplate(
        T_ref_tsr=np.eye(4),
        Tw_e=np.array([
            [1, 0, 0, 0],      # Mug x-axis aligned with table
            [0, 1, 0, 0],      # Mug y-axis aligned with table
            [0, 0, 1, 0.02],   # Mug 2cm above table surface
            [0, 0, 0, 1]
        ]),
        Bw=np.array([
            [-0.1, 0.1],       # x: allow sliding on table
            [-0.1, 0.1],       # y: allow sliding on table
            [0, 0],            # z: fixed height
            [0, 0],            # roll: keep level
            [0, 0],            # pitch: keep level
            [-np.pi/4, np.pi/4]  # yaw: allow some rotation
        ]),
        subject_entity=EntityClass.MUG,
        reference_entity=EntityClass.TABLE,
        task_category=TaskCategory.PLACE,
        variant="on",
        name="Mug Table Placement",
        description="Place mug on table surface with 2cm clearance"
    )
    
    # Box side grasp template
    box_side_grasp = TSRTemplate(
        T_ref_tsr=np.eye(4),
        Tw_e=np.array([
            [0, 0, 1, -0.08],  # Approach from -z, 8cm offset
            [1, 0, 0, 0],      # x-axis perpendicular to box
            [0, 1, 0, 0.08],   # y-axis along box axis
            [0, 0, 0, 1]
        ]),
        Bw=np.array([
            [0, 0],           # x: fixed position
            [0, 0],           # y: fixed position
            [-0.02, 0.02],    # z: small tolerance
            [0, 0],           # roll: fixed
            [0, 0],           # pitch: fixed
            [-np.pi, np.pi]   # yaw: full rotation
        ]),
        subject_entity=EntityClass.GENERIC_GRIPPER,
        reference_entity=EntityClass.BOX,
        task_category=TaskCategory.GRASP,
        variant="side",
        name="Box Side Grasp",
        description="Grasp box from the side with 8cm approach distance"
    )
    
    return [mug_side_grasp, mug_top_grasp, mug_place_table, box_side_grasp]


def demonstrate_file_organization(templates, temp_dir):
    """Demonstrate organized file structure for templates."""
    print("\n📁 Template File Organization")
    print("=" * 50)
    
    # Create organized directory structure
    grasps_dir = temp_dir / "grasps"
    places_dir = temp_dir / "places"
    grasps_dir.mkdir(parents=True, exist_ok=True)
    places_dir.mkdir(parents=True, exist_ok=True)
    
    # Save templates with descriptive filenames
    save_template(templates[0], grasps_dir / "mug_side_grasp.yaml")
    save_template(templates[1], grasps_dir / "mug_top_grasp.yaml")
    save_template(templates[2], places_dir / "mug_place_table.yaml")
    save_template(templates[3], grasps_dir / "box_side_grasp.yaml")
    
    print(f"✅ Saved templates to organized structure:")
    print(f"   {temp_dir}/")
    print(f"   ├── grasps/")
    print(f"   │   ├── mug_side_grasp.yaml")
    print(f"   │   ├── mug_top_grasp.yaml")
    print(f"   │   └── box_side_grasp.yaml")
    print(f"   └── places/")
    print(f"       └── mug_place_table.yaml")
    
    return grasps_dir, places_dir


def demonstrate_individual_loading(grasps_dir, places_dir):
    """Demonstrate loading individual template files."""
    print("\n📂 Loading Individual Templates")
    print("=" * 50)
    
    # Load specific templates as needed
    mug_side = load_template(grasps_dir / "mug_side_grasp.yaml")
    mug_top = load_template(grasps_dir / "mug_top_grasp.yaml")
    box_side = load_template(grasps_dir / "box_side_grasp.yaml")
    mug_place = load_template(places_dir / "mug_place_table.yaml")
    
    print(f"✅ Loaded individual templates:")
    print(f"   {mug_side.name}: {mug_side.description}")
    print(f"   {mug_top.name}: {mug_top.description}")
    print(f"   {box_side.name}: {box_side.description}")
    print(f"   {mug_place.name}: {mug_place.description}")
    
    return [mug_side, mug_top, box_side, mug_place]


def demonstrate_bulk_loading(grasps_dir, places_dir):
    """Demonstrate loading all templates from directories."""
    print("\n📚 Bulk Loading from Directories")
    print("=" * 50)
    
    # Load all templates from each directory
    all_grasps = TemplateIO.load_templates_from_directory(grasps_dir)
    all_places = TemplateIO.load_templates_from_directory(places_dir)
    
    print(f"✅ Loaded {len(all_grasps)} grasp templates:")
    for template in all_grasps:
        print(f"   - {template.name} ({template.subject_entity.value} -> {template.reference_entity.value})")
    
    print(f"✅ Loaded {len(all_places)} place templates:")
    for template in all_places:
        print(f"   - {template.name} ({template.subject_entity.value} -> {template.reference_entity.value})")
    
    return all_grasps + all_places


def demonstrate_template_info(grasps_dir, places_dir):
    """Demonstrate getting template metadata without loading."""
    print("\nℹ️ Template Information (Without Loading)")
    print("=" * 50)
    
    # Get info about templates without loading them completely
    mug_side_info = TemplateIO.get_template_info(grasps_dir / "mug_side_grasp.yaml")
    mug_place_info = TemplateIO.get_template_info(places_dir / "mug_place_table.yaml")
    
    print(f"✅ Template metadata:")
    print(f"   {mug_side_info['name']}:")
    print(f"     Subject: {mug_side_info['subject_entity']}")
    print(f"     Reference: {mug_side_info['reference_entity']}")
    print(f"     Task: {mug_side_info['task_category']}/{mug_side_info['variant']}")
    print(f"     Description: {mug_side_info['description']}")
    
    print(f"   {mug_place_info['name']}:")
    print(f"     Subject: {mug_place_info['subject_entity']}")
    print(f"     Reference: {mug_place_info['reference_entity']}")
    print(f"     Task: {mug_place_info['task_category']}/{mug_place_info['variant']}")
    print(f"     Description: {mug_place_info['description']}")


def demonstrate_template_usage(loaded_templates):
    """Demonstrate using the loaded templates."""
    print("\n🎯 Using Loaded Templates")
    print("=" * 50)
    
    # Simulate object poses
    mug_pose = np.array([
        [1, 0, 0, 0.5],   # Mug at x=0.5m
        [0, 1, 0, 0.3],   # y=0.3m
        [0, 0, 1, 0.1],   # z=0.1m (on table)
        [0, 0, 0, 1]
    ])
    
    table_pose = np.array([
        [1, 0, 0, 0],     # Table at origin
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    
    # Instantiate templates at specific poses
    for template in loaded_templates:
        if template.task_category == TaskCategory.GRASP:
            if template.reference_entity == EntityClass.MUG:
                tsr = template.instantiate(mug_pose)
                print(f"✅ Instantiated {template.name} at mug pose")
                pose = tsr.sample()
                print(f"   Sampled pose: [{pose[0,3]:.3f}, {pose[1,3]:.3f}, {pose[2,3]:.3f}]")
        
        elif template.task_category == TaskCategory.PLACE:
            if template.reference_entity == EntityClass.TABLE:
                tsr = template.instantiate(table_pose)
                print(f"✅ Instantiated {template.name} at table pose")
                pose = tsr.sample()
                print(f"   Sampled pose: [{pose[0,3]:.3f}, {pose[1,3]:.3f}, {pose[2,3]:.3f}]")





def main():
    """Demonstrate the multiple files approach for template management."""
    print("TSR Template File Management: Multiple Files Approach")
    print("=" * 60)
    
    # Create sample templates
    templates = create_sample_templates()
    print(f"✅ Created {len(templates)} sample templates")
    
    # Create temporary directory for demonstration
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Demonstrate organized file structure
        grasps_dir, places_dir = demonstrate_file_organization(templates, temp_path)
        
        # Demonstrate individual loading
        loaded_templates = demonstrate_individual_loading(grasps_dir, places_dir)
        
        # Demonstrate bulk loading
        all_templates = demonstrate_bulk_loading(grasps_dir, places_dir)
        
        # Demonstrate template info
        demonstrate_template_info(grasps_dir, places_dir)
        
        # Demonstrate template usage
        demonstrate_template_usage(loaded_templates)
        
        print(f"\n✅ All demonstrations completed in {temp_path}")
    
    print("\n🎯 Summary:")
    print("   This example shows how to organize TSR templates in separate YAML files")
    print("   for better version control and collaborative development.")


if __name__ == "__main__":
    main()
