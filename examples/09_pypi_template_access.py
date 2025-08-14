#!/usr/bin/env python
"""
PyPI Template Access Example: How to use templates when installed from PyPI.

This example demonstrates how users can access TSR templates when the package
is installed from PyPI using 'pip install tsr'.
"""

import numpy as np

from tsr import (
    list_available_templates,
    load_package_template,
    load_package_templates_by_category,
    get_package_templates,
    TSRLibraryRelational,
    EntityClass,
    TaskCategory,
    TaskType
)


def demonstrate_template_discovery():
    """Demonstrate discovering available templates."""
    print("\n🔍 Template Discovery")
    print("=" * 50)
    
    # List all available templates in the package
    available_templates = list_available_templates()
    print(f"✅ Found {len(available_templates)} templates in package:")
    for template in available_templates:
        print(f"   - {template}")
    
    # Get the package templates directory
    template_dir = get_package_templates()
    print(f"\n📁 Package templates directory: {template_dir}")
    print(f"   Directory exists: {template_dir.exists()}")


def demonstrate_individual_template_loading():
    """Demonstrate loading individual templates."""
    print("\n📂 Individual Template Loading")
    print("=" * 50)
    
    # Load specific templates by category and name
    mug_grasp = load_package_template("grasps", "mug_side_grasp.yaml")
    mug_place = load_package_template("places", "mug_on_table.yaml")
    
    print(f"✅ Loaded {mug_grasp.name}")
    print(f"   Description: {mug_grasp.description}")
    print(f"   Subject: {mug_grasp.subject_entity.value}")
    print(f"   Reference: {mug_grasp.reference_entity.value}")
    print(f"   Task: {mug_grasp.task_category.value}/{mug_grasp.variant}")
    
    print(f"\n✅ Loaded {mug_place.name}")
    print(f"   Description: {mug_place.description}")
    print(f"   Subject: {mug_place.subject_entity.value}")
    print(f"   Reference: {mug_place.reference_entity.value}")
    print(f"   Task: {mug_place.task_category.value}/{mug_place.variant}")


def demonstrate_category_loading():
    """Demonstrate loading all templates from a category."""
    print("\n📚 Category Template Loading")
    print("=" * 50)
    
    # Load all templates from grasps category
    grasp_templates = load_package_templates_by_category("grasps")
    print(f"✅ Loaded {len(grasp_templates)} grasp templates:")
    for template in grasp_templates:
        print(f"   - {template.name}: {template.description}")
    
    # Load all templates from places category
    place_templates = load_package_templates_by_category("places")
    print(f"\n✅ Loaded {len(place_templates)} place templates:")
    for template in place_templates:
        print(f"   - {template.name}: {template.description}")


def demonstrate_library_integration():
    """Demonstrate integrating package templates with the library."""
    print("\n📚 Library Integration")
    print("=" * 50)
    
    # Create library and load package templates
    library = TSRLibraryRelational()
    
    # Load and register package templates
    grasp_templates = load_package_templates_by_category("grasps")
    place_templates = load_package_templates_by_category("places")
    
    # Register grasp templates
    for template in grasp_templates:
        library.register_template(
            subject=template.subject_entity,
            reference=template.reference_entity,
            task=TaskType(template.task_category, template.variant),
            template=template,
            description=template.description
        )
    
    # Register place templates
    for template in place_templates:
        library.register_template(
            subject=template.subject_entity,
            reference=template.reference_entity,
            task=TaskType(template.task_category, template.variant),
            template=template,
            description=template.description
        )
    
    # Query available templates
    available = library.list_available_templates()
    print(f"✅ Registered {len(available)} templates in library:")
    for subject, reference, task, description in available:
        print(f"   {subject.value} -> {reference.value} ({task}): {description}")


def demonstrate_template_usage():
    """Demonstrate using loaded templates."""
    print("\n🎮 Template Usage")
    print("=" * 50)
    
    # Load a template from the package
    template = load_package_template("grasps", "mug_side_grasp.yaml")
    
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
    
    print(f"✅ Using {template.name} from package")
    print(f"   Instantiated at mug pose: [{mug_pose[0,3]:.3f}, {mug_pose[1,3]:.3f}, {mug_pose[2,3]:.3f}]")
    print(f"   Sampled poses:")
    for i, pose in enumerate(poses):
        print(f"     {i+1}: [{pose[0,3]:.3f}, {pose[1,3]:.3f}, {pose[2,3]:.3f}]")


def demonstrate_installation_workflow():
    """Demonstrate the complete PyPI installation workflow."""
    print("\n📦 PyPI Installation Workflow")
    print("=" * 50)
    
    print("1. Install package from PyPI:")
    print("   pip install tsr")
    print()
    
    print("2. Import and discover templates:")
    print("   from tsr import list_available_templates")
    print("   templates = list_available_templates()")
    print()
    
    print("3. Load specific templates:")
    print("   from tsr import load_package_template")
    print("   template = load_package_template('grasps', 'mug_side_grasp.yaml')")
    print()
    
    print("4. Use templates in your code:")
    print("   tsr = template.instantiate(object_pose)")
    print("   pose = tsr.sample()")
    print()
    
    print("✅ No git clone needed - everything works from PyPI!")


def main():
    """Demonstrate PyPI template access functionality."""
    print("PyPI Template Access Example")
    print("=" * 60)
    print("This example shows how users can access TSR templates")
    print("when the package is installed from PyPI using 'pip install tsr'")
    print()
    
    # Demonstrate all functionality
    demonstrate_template_discovery()
    demonstrate_individual_template_loading()
    demonstrate_category_loading()
    demonstrate_library_integration()
    demonstrate_template_usage()
    demonstrate_installation_workflow()
    
    print(f"\n🎯 Summary")
    print("=" * 50)
    print("✅ Templates are included in the PyPI package")
    print("✅ Easy discovery with list_available_templates()")
    print("✅ Simple loading with load_package_template()")
    print("✅ Category-based loading with load_package_templates_by_category()")
    print("✅ Full integration with TSRLibraryRelational")
    print("✅ No additional downloads or git clones needed")
    
    print(f"\n💡 Key Benefits:")
    print("   - One-line installation: pip install tsr")
    print("   - Templates included in package")
    print("   - Easy discovery and loading")
    print("   - Works offline after installation")
    print("   - Version-controlled templates")


if __name__ == "__main__":
    main()
