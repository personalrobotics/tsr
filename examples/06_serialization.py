#!/usr/bin/env python
"""
Serialization Example: TSR persistence and data exchange.

This example demonstrates TSR serialization capabilities:
- Converting TSRs to/from dictionaries
- JSON serialization for data exchange
- YAML serialization for configuration
- TSR chain serialization
"""

import numpy as np
from numpy import pi

from tsr import TSR, TSRChain


def main():
    """Demonstrate TSR serialization and persistence."""
    print("=== Serialization Example ===")
    
    # Create a TSR
    tsr = TSR(
        T0_w=np.eye(4),
        Tw_e=np.eye(4),
        Bw=np.array([
            [-0.1, 0.1], [-0.1, 0.1], [-0.1, 0.1],  # Translation bounds
            [-pi/4, pi/4], [-pi/4, pi/4], [-pi/4, pi/4]  # Rotation bounds
        ])
    )
    
    print("--- Dictionary Serialization ---")
    # Convert to dictionary
    tsr_dict = tsr.to_dict()
    print(f"TSR as dictionary: {tsr_dict}")
    
    # Convert back to TSR
    tsr_from_dict = TSR.from_dict(tsr_dict)
    print(f"TSR from dict matches original: {np.allclose(tsr.T0_w, tsr_from_dict.T0_w)}")
    
    print("\n--- JSON Serialization ---")
    # Convert to JSON
    tsr_json = tsr.to_json()
    print(f"TSR as JSON: {tsr_json[:100]}...")
    
    # Convert back from JSON
    tsr_from_json = TSR.from_json(tsr_json)
    print(f"TSR from JSON matches original: {np.allclose(tsr.T0_w, tsr_from_json.T0_w)}")
    
    print("\n--- YAML Serialization ---")
    # Convert to YAML
    tsr_yaml = tsr.to_yaml()
    print(f"TSR as YAML:\n{tsr_yaml}")
    
    # Convert back from YAML
    tsr_from_yaml = TSR.from_yaml(tsr_yaml)
    print(f"TSR from YAML matches original: {np.allclose(tsr.T0_w, tsr_from_yaml.T0_w)}")
    
    print("\n--- TSR Chain Serialization ---")
    # Create a TSR chain
    chain = TSRChain(
        sample_start=False,
        sample_goal=True,
        constrain=True,
        TSRs=[tsr]
    )
    
    # Serialize chain to dictionary
    chain_dict = chain.to_dict()
    print(f"Chain as dictionary: {chain_dict}")
    
    # Deserialize chain
    chain_from_dict = TSRChain.from_dict(chain_dict)
    print(f"Chain serialization successful: {len(chain_from_dict.TSRs) == len(chain.TSRs)}")
    
    # Serialize chain to JSON
    chain_json = chain.to_json()
    print(f"Chain as JSON: {chain_json[:100]}...")
    
    # Deserialize from JSON
    chain_from_json = TSRChain.from_json(chain_json)
    print(f"Chain JSON serialization successful: {len(chain_from_json.TSRs) == len(chain.TSRs)}")
    
    print("\n--- Cross-Format Roundtrip ---")
    # Test roundtrip: TSR -> Dict -> JSON -> YAML -> TSR
    tsr_dict_2 = tsr.to_dict()
    tsr_json_2 = TSR.from_dict(tsr_dict_2).to_json()
    tsr_yaml_2 = TSR.from_json(tsr_json_2).to_yaml()
    tsr_final = TSR.from_yaml(tsr_yaml_2)
    
    print(f"Cross-format roundtrip successful: {np.allclose(tsr.T0_w, tsr_final.T0_w)}")
    
    print()


if __name__ == "__main__":
    main()
