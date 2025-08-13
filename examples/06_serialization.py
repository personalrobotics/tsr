#!/usr/bin/env python
"""
Serialization Example: Save and load TSRs and TSRChains.

This example demonstrates how to serialize TSRs and TSRChains to various
formats (dictionary, JSON, YAML) and load them back. It also shows
TSRTemplate serialization with semantic context.
"""

import numpy as np
from tsr.core.tsr import TSR
from tsr.core.tsr_chain import TSRChain
from tsr.core.tsr_template import TSRTemplate
from tsr.schema import EntityClass, TaskCategory, TaskType


def main():
    """Run the serialization examples."""
    print("TSR Library - Serialization Example")
    print("=" * 50)
    
    # Create a sample TSR
    print("\n1. Basic TSR Serialization")
    print("-" * 30)
    
    tsr = TSR(
        T0_w=np.array([
            [1, 0, 0, 0.5],
            [0, 1, 0, 0.0],
            [0, 0, 1, 0.3],
            [0, 0, 0, 1]
        ]),
        Tw_e=np.array([
            [0, 0, 1, -0.05],
            [1, 0, 0, 0],
            [0, 1, 0, 0.05],
            [0, 0, 0, 1]
        ]),
        Bw=np.array([
            [0, 0],           # x: fixed position
            [0, 0],           # y: fixed position
            [-0.01, 0.01],    # z: small tolerance
            [0, 0],           # roll: fixed
            [0, 0],           # pitch: fixed
            [-np.pi, np.pi]   # yaw: full rotation
        ])
    )
    
    # Test dictionary serialization
    tsr_dict = tsr.to_dict()
    print(f"TSR serialized to dict: {len(tsr_dict)} fields")
    
    # Test roundtrip
    tsr_from_dict = TSR.from_dict(tsr_dict)
    print(f"TSR from dict matches original: {np.allclose(tsr.T0_w, tsr_from_dict.T0_w)}")
    
    # Test JSON serialization
    print("\n2. JSON Serialization")
    print("-" * 30)
    
    tsr_json = tsr.to_json()
    print(f"TSR serialized to JSON: {len(tsr_json)} characters")
    
    # Test roundtrip
    tsr_from_json = TSR.from_json(tsr_json)
    print(f"TSR from JSON matches original: {np.allclose(tsr.T0_w, tsr_from_json.T0_w)}")
    
    # Test YAML serialization
    print("\n3. YAML Serialization")
    print("-" * 30)
    
    tsr_yaml = tsr.to_yaml()
    print("TSR serialized to YAML:")
    print(tsr_yaml)
    
    # Test roundtrip
    tsr_from_yaml = TSR.from_yaml(tsr_yaml)
    print(f"TSR from YAML matches original: {np.allclose(tsr.T0_w, tsr_from_yaml.T0_w)}")
    
    # Test TSRChain serialization
    print("\n4. TSRChain Serialization")
    print("-" * 30)
    
    # Create a TSR chain
    tsr1 = TSR(
        T0_w=np.eye(4),
        Tw_e=np.eye(4),
        Bw=np.array([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [-np.pi, np.pi]])
    )
    tsr2 = TSR(
        T0_w=np.eye(4),
        Tw_e=np.eye(4),
        Bw=np.array([[-0.1, 0.1], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]])
    )
    
    chain = TSRChain([tsr1, tsr2])
    
    # Test dictionary serialization
    chain_dict = chain.to_dict()
    print(f"TSRChain serialized to dict: {len(chain_dict)} fields")
    
    # Test roundtrip
    chain_from_dict = TSRChain.from_dict(chain_dict)
    print(f"Chain serialization successful: {len(chain_from_dict.TSRs) == len(chain.TSRs)}")
    
    # Test YAML serialization
    chain_yaml = chain.to_yaml()
    print("TSRChain serialized to YAML:")
    print(chain_yaml)
    
    # Test cross-format roundtrip
    print("\n5. Cross-Format Roundtrip")
    print("-" * 30)
    
    # TSR: dict -> JSON -> YAML -> TSR
    tsr_dict_2 = tsr.to_dict()
    tsr_json_2 = TSR.from_dict(tsr_dict_2).to_json()
    tsr_yaml_2 = TSR.from_json(tsr_json_2).to_yaml()
    tsr_final = TSR.from_yaml(tsr_yaml_2)
    
    print(f"Cross-format roundtrip successful: {np.allclose(tsr.T0_w, tsr_final.T0_w)}")
    
    # Test TSRTemplate serialization
    print("\n6. TSRTemplate Serialization")
    print("-" * 30)
    
    # Create a TSR template with semantic context
    template = TSRTemplate(
        T_ref_tsr=np.eye(4),
        Tw_e=np.array([
            [0, 0, 1, -0.05],
            [1, 0, 0, 0],
            [0, 1, 0, 0.05],
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
        name="Cylinder Side Grasp",
        description="Grasp a cylindrical object from the side with 5cm approach distance"
    )
    
    # Test dictionary serialization
    template_dict = template.to_dict()
    print(f"TSRTemplate serialized to dict: {len(template_dict)} fields")
    print(f"  - name: {template_dict['name']}")
    print(f"  - subject_entity: {template_dict['subject_entity']}")
    print(f"  - task_category: {template_dict['task_category']}")
    print(f"  - variant: {template_dict['variant']}")
    
    # Test roundtrip
    template_from_dict = TSRTemplate.from_dict(template_dict)
    print(f"Template from dict matches original: {template.name == template_from_dict.name}")
    print(f"  - Semantic context preserved: {template.subject_entity == template_from_dict.subject_entity}")
    
    # Test YAML serialization
    template_yaml = template.to_yaml()
    print("\nTSRTemplate serialized to YAML:")
    print(template_yaml)
    
    # Test roundtrip
    template_from_yaml = TSRTemplate.from_yaml(template_yaml)
    print(f"Template from YAML matches original: {template.name == template_from_yaml.name}")
    
    # Test template instantiation after serialization
    print("\n7. Template Instantiation After Serialization")
    print("-" * 30)
    
    # Instantiate the original template
    cylinder_pose = np.array([
        [1, 0, 0, 0.5],  # Cylinder at x=0.5
        [0, 1, 0, 0.0],
        [0, 0, 1, 0.3],
        [0, 0, 0, 1]
    ])
    
    original_tsr = template.instantiate(cylinder_pose)
    serialized_tsr = template_from_yaml.instantiate(cylinder_pose)
    
    print(f"Instantiated TSRs match: {np.allclose(original_tsr.T0_w, serialized_tsr.T0_w)}")
    
    # Test template library serialization
    print("\n8. Template Library Serialization")
    print("-" * 30)
    
    # Create multiple templates
    templates = [
        TSRTemplate(
            T_ref_tsr=np.eye(4),
            Tw_e=np.array([
                [0, 0, 1, -0.05],
                [1, 0, 0, 0],
                [0, 1, 0, 0.05],
                [0, 0, 0, 1]
            ]),
            Bw=np.array([
                [0, 0], [0, 0], [-0.01, 0.01],
                [0, 0], [0, 0], [-np.pi, np.pi]
            ]),
            subject_entity=EntityClass.GENERIC_GRIPPER,
            reference_entity=EntityClass.MUG,
            task_category=TaskCategory.GRASP,
            variant="side",
            name="Mug Side Grasp",
            description="Grasp mug from the side"
        ),
        TSRTemplate(
            T_ref_tsr=np.eye(4),
            Tw_e=np.array([
                [0, 0, 1, -0.05],
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1]
            ]),
            Bw=np.array([
                [0, 0], [0, 0], [-0.01, 0.01],
                [0, 0], [0, 0], [-np.pi, np.pi]
            ]),
            subject_entity=EntityClass.GENERIC_GRIPPER,
            reference_entity=EntityClass.MUG,
            task_category=TaskCategory.GRASP,
            variant="top",
            name="Mug Top Grasp",
            description="Grasp mug from the top"
        ),
        TSRTemplate(
            T_ref_tsr=np.eye(4),
            Tw_e=np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0.02],
                [0, 0, 0, 1]
            ]),
            Bw=np.array([
                [-0.1, 0.1], [0, 0], [0, 0],
                [0, 0], [0, 0], [-np.pi/4, np.pi/4]
            ]),
            subject_entity=EntityClass.MUG,
            reference_entity=EntityClass.TABLE,
            task_category=TaskCategory.PLACE,
            variant="on",
            name="Table Placement",
            description="Place mug on table surface"
        )
    ]
    
    # Serialize template library
    template_library = [t.to_dict() for t in templates]
    
    # Save to YAML (simulated)
    import yaml
    library_yaml = yaml.dump(template_library, default_flow_style=False)
    print("Template library serialized to YAML:")
    print(library_yaml)
    
    # Load from YAML (simulated)
    loaded_library = yaml.safe_load(library_yaml)
    loaded_templates = [TSRTemplate.from_dict(t) for t in loaded_library]
    
    print(f"Loaded {len(loaded_templates)} templates:")
    for i, t in enumerate(loaded_templates):
        print(f"  {i+1}. {t.name} ({t.subject_entity} -> {t.reference_entity}, {t.task_category}/{t.variant})")
    
    print("\n✅ Serialization example completed successfully!")


if __name__ == "__main__":
    main()
