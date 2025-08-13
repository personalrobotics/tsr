#!/usr/bin/env python
"""
Advanced Sampling Example: Weighted sampling from multiple TSRs.

This example demonstrates advanced sampling utilities:
- Computing weights based on TSR volumes
- Choosing TSRs with weighted random sampling
- Sampling poses from multiple TSRs
- Working with TSR templates and sampling
"""

import numpy as np
from numpy import pi

from tsr import (
    TSR, TSRTemplate,
    sample_from_tsrs, weights_from_tsrs, choose_tsr,
    sample_from_templates, instantiate_templates
)


def main():
    """Demonstrate advanced sampling from multiple TSRs."""
    print("=== Advanced Sampling Example ===")
    
    # Create multiple TSRs for different grasp approaches
    side_tsr = TSR(
        T0_w=np.eye(4),
        Tw_e=np.eye(4),
        Bw=np.array([
            [0, 0], [0, 0], [-0.01, 0.01],  # Translation bounds
            [0, 0], [0, 0], [-pi, pi]       # Rotation bounds
        ])
    )
    
    top_tsr = TSR(
        T0_w=np.eye(4),
        Tw_e=np.eye(4),
        Bw=np.array([
            [-0.01, 0.01], [-0.01, 0.01], [0, 0],  # Translation bounds
            [0, 0], [0, 0], [-pi, pi]              # Rotation bounds
        ])
    )
    
    # Compute weights based on TSR volumes
    tsrs = [side_tsr, top_tsr]
    weights = weights_from_tsrs(tsrs)
    print(f"TSR weights: {weights}")
    
    # Choose a TSR with probability proportional to weight
    selected_tsr = choose_tsr(tsrs)
    print(f"Selected TSR: {selected_tsr}")
    
    # Sample directly from multiple TSRs
    pose = sample_from_tsrs(tsrs)
    print(f"Sampled pose:\n{pose}")
    
    # Verify the pose is valid
    is_valid = any(tsr.contains(pose) for tsr in tsrs)
    print(f"Pose is valid: {is_valid}")
    
    # Demonstrate sampling from templates
    print("\n--- Template Sampling ---")
    
    # Create templates for different grasp approaches
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
        ])
    )
    
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
        ])
    )
    
    # Object pose
    object_pose = np.array([
        [1, 0, 0, 0.5],  # Object at x=0.5
        [0, 1, 0, 0.0],
        [0, 0, 1, 0.3],
        [0, 0, 0, 1]
    ])
    
    # Instantiate templates
    templates = [side_template, top_template]
    instantiated_tsrs = instantiate_templates(templates, object_pose)
    print(f"Instantiated {len(instantiated_tsrs)} TSRs from templates")
    
    # Sample from templates
    template_pose = sample_from_templates(templates, object_pose)
    print(f"Sampled pose from templates:\n{template_pose}")
    
    # Verify template pose is valid
    template_is_valid = any(tsr.contains(template_pose) for tsr in instantiated_tsrs)
    print(f"Template pose is valid: {template_is_valid}")
    
    print()


if __name__ == "__main__":
    main()
