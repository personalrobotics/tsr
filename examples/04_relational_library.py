#!/usr/bin/env python
"""
Relational Library Example: Task-based TSR generation and querying.

This example demonstrates the relational library for task-based TSR generation:
- Registering TSR generators for specific entity/task combinations
- Querying available TSRs for given scenarios
- Discovering available tasks for entities
- Example: Grasp and place operations
"""

import numpy as np
from numpy import pi

from tsr import (
    TSRTemplate, TSRLibraryRelational,
    TaskCategory, TaskType, EntityClass
)


def main():
    """Demonstrate relational library for task-based TSR generation."""
    print("=== Relational Library Example ===")
    
    # Create library
    library = TSRLibraryRelational()
    
    # Define TSR generators for different tasks
    def mug_grasp_generator(T_ref_world):
        """Generate TSR templates for grasping a mug."""
        # Side grasp template
        side_template = TSRTemplate(
            T_ref_tsr=np.eye(4),
            Tw_e=np.array([
                [0, 0, 1, -0.05],  # Approach from -z
                [1, 0, 0, 0],
                [0, 1, 0, 0.05],
                [0, 0, 0, 1]
            ]),
            Bw=np.array([
                [0, 0], [0, 0], [-0.01, 0.01],  # Translation bounds
                [0, 0], [0, 0], [-pi, pi]       # Rotation bounds
            ]),
            subject_entity=EntityClass.GENERIC_GRIPPER,
            reference_entity=EntityClass.MUG,
            task_category=TaskCategory.GRASP,
            variant="side"
        )
        
        # Top grasp template
        top_template = TSRTemplate(
            T_ref_tsr=np.eye(4),
            Tw_e=np.array([
                [0, 0, 1, 0],      # Approach from -z
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1]
            ]),
            Bw=np.array([
                [-0.01, 0.01], [-0.01, 0.01], [0, 0],  # Translation bounds
                [0, 0], [0, 0], [-pi, pi]              # Rotation bounds
            ]),
            subject_entity=EntityClass.GENERIC_GRIPPER,
            reference_entity=EntityClass.MUG,
            task_category=TaskCategory.GRASP,
            variant="top"
        )
        
        return [side_template, top_template]
    
    def mug_place_generator(T_ref_world):
        """Generate TSR templates for placing a mug."""
        place_template = TSRTemplate(
            T_ref_tsr=np.eye(4),
            Tw_e=np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0.02],  # 2cm above surface
                [0, 0, 0, 1]
            ]),
            Bw=np.array([
                [-0.1, 0.1], [-0.1, 0.1], [0, 0],      # Translation bounds
                [0, 0], [0, 0], [-pi/4, pi/4]          # Rotation bounds
            ]),
            subject_entity=EntityClass.MUG,
            reference_entity=EntityClass.TABLE,
            task_category=TaskCategory.PLACE,
            variant="on"
        )
        return [place_template]
    
    # Register generators
    library.register(
        subject=EntityClass.GENERIC_GRIPPER,
        reference=EntityClass.MUG,
        task=TaskType(TaskCategory.GRASP, "side"),
        generator=mug_grasp_generator
    )
    
    library.register(
        subject=EntityClass.MUG,
        reference=EntityClass.TABLE,
        task=TaskType(TaskCategory.PLACE, "on"),
        generator=mug_place_generator
    )
    
    # Query available TSRs
    mug_pose = np.array([
        [1, 0, 0, 0.5],
        [0, 1, 0, 0.0],
        [0, 0, 1, 0.3],
        [0, 0, 0, 1]
    ])
    
    table_pose = np.array([
        [1, 0, 0, 0.0],
        [0, 1, 0, 0.0],
        [0, 0, 1, 0.0],
        [0, 0, 0, 1]
    ])
    
    # Get grasp TSRs
    grasp_tsrs = library.query(
        subject=EntityClass.GENERIC_GRIPPER,
        reference=EntityClass.MUG,
        task=TaskType(TaskCategory.GRASP, "side"),
        T_ref_world=mug_pose
    )
    
    # Get place TSRs
    place_tsrs = library.query(
        subject=EntityClass.MUG,
        reference=EntityClass.TABLE,
        task=TaskType(TaskCategory.PLACE, "on"),
        T_ref_world=table_pose
    )
    
    print(f"Found {len(grasp_tsrs)} grasp TSRs")
    print(f"Found {len(place_tsrs)} place TSRs")
    
    # Discover available tasks
    mug_tasks = library.list_tasks_for_reference(EntityClass.MUG)
    table_tasks = library.list_tasks_for_reference(EntityClass.TABLE)
    
    print(f"Tasks for MUG: {[str(task) for task in mug_tasks]}")
    print(f"Tasks for TABLE: {[str(task) for task in table_tasks]}")
    
    # Filter tasks by subject
    gripper_tasks = library.list_tasks_for_reference(
        EntityClass.MUG, 
        subject_filter=EntityClass.GENERIC_GRIPPER
    )
    print(f"Gripper tasks for MUG: {[str(task) for task in gripper_tasks]}")
    
    # Filter tasks by prefix
    grasp_tasks = library.list_tasks_for_reference(
        EntityClass.MUG, 
        task_prefix="grasp"
    )
    print(f"Grasp tasks for MUG: {[str(task) for task in grasp_tasks]}")
    
    print()


if __name__ == "__main__":
    main()
