# SPDX-License-Identifier: BSD-2-Clause
# Authors: Siddhartha Srinivasa and contributors to TSR

import numpy as np
from typing import List, Optional, Union
from .tsr import TSR


class TSRChain:
    """
    Core TSRChain class — geometry-only, robot-agnostic.
    
    A TSRChain represents a sequence of TSRs that can be used for:
    - Sampling start/goal poses
    - Constraining trajectories
    - Complex motion planning tasks
    """
    
    def __init__(self, sample_start: bool = False, sample_goal: bool = False, 
                 constrain: bool = False, TSR: Optional[TSR] = None, 
                 TSRs: Optional[List[TSR]] = None):
        """
        Initialize a TSRChain.
        
        Args:
            sample_start: Whether to use this chain for sampling start poses
            sample_goal: Whether to use this chain for sampling goal poses
            constrain: Whether to use this chain for trajectory constraints
            TSR: Single TSR to add to the chain
            TSRs: List of TSRs to add to the chain
        """
        self.sample_start = sample_start
        self.sample_goal = sample_goal
        self.constrain = constrain
        self.TSRs = []
        
        if TSR is not None:
            self.TSRs.append(TSR)
        
        if TSRs is not None:
            self.TSRs.extend(TSRs)
    
    def append(self, tsr: TSR):
        """Add a TSR to the end of the chain."""
        self.TSRs.append(tsr)
    
    def is_valid(self, xyzrpy_list: List[np.ndarray], ignoreNAN: bool = False) -> bool:
        """
        Check if a list of xyzrpy poses is valid for this chain.
        
        Args:
            xyzrpy_list: List of 6-vectors, one for each TSR in the chain
            ignoreNAN: If True, ignore NaN values in xyzrpy_list
        """
        if len(xyzrpy_list) != len(self.TSRs):
            return False
        
        for tsr, xyzrpy in zip(self.TSRs, xyzrpy_list):
            if not tsr.is_valid(xyzrpy, ignoreNAN):
                return False
        
        return True
    
    def to_transform(self, xyzrpy_list: List[np.ndarray]) -> np.ndarray:
        """
        Convert a list of xyzrpy poses to a world-frame transform.
        
        This computes the composition of all TSR transforms in the chain.
        """
        if len(xyzrpy_list) != len(self.TSRs):
            raise ValueError(f"Expected {len(self.TSRs)} xyzrpy vectors, got {len(xyzrpy_list)}")
        
        # Start with identity transform
        result = np.eye(4)
        
        # Compose all TSR transforms
        for tsr, xyzrpy in zip(self.TSRs, xyzrpy_list):
            tsr_transform = tsr.to_transform(xyzrpy)
            result = result @ tsr_transform
        
        return result
    
    def sample_xyzrpy(self, xyzrpy_list: Optional[List[np.ndarray]] = None) -> List[np.ndarray]:
        """
        Sample xyzrpy poses for all TSRs in the chain.
        
        Args:
            xyzrpy_list: Optional list of xyzrpy vectors to fix some dimensions
        """
        if xyzrpy_list is None:
            # Use NANBW for each TSR when no input is provided
            from tsr.core.tsr import NANBW
            xyzrpy_list = [NANBW] * len(self.TSRs)
        
        if len(xyzrpy_list) != len(self.TSRs):
            raise ValueError(f"Expected {len(self.TSRs)} xyzrpy vectors, got {len(xyzrpy_list)}")
        
        result = []
        for tsr, xyzrpy in zip(self.TSRs, xyzrpy_list):
            sampled = tsr.sample_xyzrpy(xyzrpy)
            result.append(sampled)
        
        return result
    
    def sample(self, xyzrpy_list: Optional[List[np.ndarray]] = None) -> np.ndarray:
        """
        Sample a world-frame transform from this TSR chain.
        
        Args:
            xyzrpy_list: Optional list of xyzrpy vectors to fix some dimensions
        """
        sampled_xyzrpy = self.sample_xyzrpy(xyzrpy_list)
        return self.to_transform(sampled_xyzrpy)
    
    def distance(self, trans: np.ndarray) -> float:
        """
        Compute the distance from a transform to this TSR chain.
        
        This is the minimum distance over all valid poses in the chain.
        """
        # For now, use a simple approach: find the minimum distance to any TSR
        # A more sophisticated approach would optimize over the chain composition
        min_distance = float('inf')
        
        for tsr in self.TSRs:
            distance, _ = tsr.distance(trans)  # Unpack tuple
            min_distance = min(min_distance, distance)
        
        return min_distance
    
    def contains(self, trans: np.ndarray) -> bool:
        """Check if a transform is within this TSR chain."""
        # For now, check if the transform is within any TSR
        # A more sophisticated approach would check the chain composition
        for tsr in self.TSRs:
            if tsr.contains(trans):
                return True
        return False
    
    def to_xyzrpy(self, trans: np.ndarray) -> List[np.ndarray]:
        """
        Convert a world-frame transform to xyzrpy poses for each TSR in the chain.
        
        Note: This is an approximation for chains with multiple TSRs.
        """
        if len(self.TSRs) == 1:
            return [self.TSRs[0].to_xyzrpy(trans)]
        
        # For multiple TSRs, we need to decompose the transform
        # This is a simplified approach - in practice, you might need more sophisticated decomposition
        result = []
        current_trans = trans.copy()
        
        for tsr in self.TSRs:
            xyzrpy = tsr.to_xyzrpy(current_trans)
            result.append(xyzrpy)
            
            # Update transform for next TSR (remove this TSR's contribution)
            tsr_transform = tsr.to_transform(xyzrpy)
            current_trans = np.linalg.inv(tsr_transform) @ current_trans
        
        return result
    
    def to_dict(self) -> dict:
        """Convert TSRChain to dictionary representation."""
        return {
            'sample_start': self.sample_start,
            'sample_goal': self.sample_goal,
            'constrain': self.constrain,
            'tsrs': [tsr.to_dict() for tsr in self.TSRs]
        }
    
    @staticmethod
    def from_dict(data: dict) -> 'TSRChain':
        """Create TSRChain from dictionary representation."""
        tsrs = [TSR.from_dict(tsr_data) for tsr_data in data['tsrs']]
        return TSRChain(
            sample_start=data.get('sample_start', False),
            sample_goal=data.get('sample_goal', False),
            constrain=data.get('constrain', False),
            TSRs=tsrs
        )
    
    def to_json(self) -> str:
        """Convert TSRChain to JSON string."""
        import json
        return json.dumps(self.to_dict())
    
    @staticmethod
    def from_json(json_str: str) -> 'TSRChain':
        """Create TSRChain from JSON string."""
        import json
        data = json.loads(json_str)
        return TSRChain.from_dict(data)
    
    def to_yaml(self) -> str:
        """Convert TSRChain to YAML string."""
        import yaml
        return yaml.dump(self.to_dict())
    
    @staticmethod
    def from_yaml(yaml_str: str) -> 'TSRChain':
        """Create TSRChain from YAML string."""
        import yaml
        data = yaml.safe_load(yaml_str)
        return TSRChain.from_dict(data) 