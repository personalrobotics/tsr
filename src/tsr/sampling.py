from __future__ import annotations

from typing import List, Sequence, Optional
import numpy as np
from numpy import pi

try:
    from tsr.core.tsr import TSR as CoreTSR  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    CoreTSR = object  # type: ignore[assignment]


def _interval_sum(Bw: np.ndarray) -> float:
    """Sum of Bw interval widths with rotational widths clamped to 2π.
    
    This helper function computes the "volume" of a TSR by summing the
    widths of all bounds, with rotational bounds clamped to 2π to avoid
    infinite volumes from full rotations.
    
    Args:
        Bw: (6,2) bounds matrix where each row [i,:] is [min, max] for dimension i
        
    Returns:
        Sum of interval widths, with rotational bounds clamped to 2π
        
    Raises:
        ValueError: If Bw is not shape (6,2)
    """
    if Bw.shape != (6, 2):
        raise ValueError(f"Bw must be shape (6,2), got {Bw.shape}")
    widths = np.asarray(Bw[:, 1] - Bw[:, 0], dtype=float)
    widths[3:6] = np.minimum(widths[3:6], 2.0 * pi)
    widths = np.maximum(widths, 0.0)
    return float(np.sum(widths))


def weights_from_tsrs(tsrs: Sequence[CoreTSR]) -> np.ndarray:
    """Compute non-negative weights ∝ sum of Bw widths; fallback to uniform if all zero.
    
    This function computes weights for TSRs based on their geometric volumes.
    TSRs with larger bounds (more freedom) get higher weights, making them
    more likely to be selected during sampling.
    
    Args:
        tsrs: Sequence of TSR objects
        
    Returns:
        Array of non-negative weights, one per TSR. Weights are proportional
        to the sum of bound widths. If all TSRs have zero volume, returns
        uniform weights.
        
    Raises:
        ValueError: If tsrs is empty
        
    Examples:
        >>> # Create TSRs with different volumes
        >>> tsr1 = TSR(T0_w=np.eye(4), Tw_e=np.eye(4), 
        ...            Bw=np.array([[0,0], [0,0], [0,0], [0,0], [0,0], [-pi,pi]]))
        >>> tsr2 = TSR(T0_w=np.eye(4), Tw_e=np.eye(4),
        ...            Bw=np.array([[0,0], [0,0], [0,0], [0,0], [0,0], [0,0]]))
        >>> weights = weights_from_tsrs([tsr1, tsr2])
        >>> weights[0] > weights[1]  # tsr1 has higher weight (2π volume)
        True
    """
    if len(tsrs) == 0:
        raise ValueError("Expected at least one TSR.")
    w = np.array([_interval_sum(t.Bw) for t in tsrs], dtype=float)
    if not np.any(w > 0.0):
        w = np.ones_like(w)
    return w


def choose_tsr_index(tsrs: Sequence[CoreTSR], rng: Optional[np.random.Generator] = None) -> int:
    """Choose an index with probability proportional to weight.
    
    This function selects a TSR index using weighted random sampling.
    TSRs with larger volumes (computed via weights_from_tsrs) are more
    likely to be selected.
    
    Args:
        tsrs: Sequence of TSR objects
        rng: Optional random number generator. If None, uses default RNG.
        
    Returns:
        Index of selected TSR (0 <= index < len(tsrs))
        
    Examples:
        >>> # Create TSRs with different volumes
        >>> tsr1 = TSR(T0_w=np.eye(4), Tw_e=np.eye(4), 
        ...            Bw=np.array([[0,0], [0,0], [0,0], [0,0], [0,0], [-pi,pi]]))
        >>> tsr2 = TSR(T0_w=np.eye(4), Tw_e=np.eye(4),
        ...            Bw=np.array([[0,0], [0,0], [0,0], [0,0], [0,0], [0,0]]))
        >>> 
        >>> # Choose with default RNG
        >>> index = choose_tsr_index([tsr1, tsr2])
        >>> 0 <= index < 2
        True
        >>> 
        >>> # Choose with custom RNG for reproducibility
        >>> rng = np.random.default_rng(42)
        >>> index = choose_tsr_index([tsr1, tsr2], rng)
        >>> 0 <= index < 2
        True
    """
    rng = rng or np.random.default_rng()
    w = weights_from_tsrs(tsrs)
    p = w / np.sum(w)
    return int(rng.choice(len(tsrs), p=p))


def choose_tsr(tsrs: Sequence[CoreTSR], rng: Optional[np.random.Generator] = None) -> CoreTSR:
    """Choose a TSR with probability proportional to weight.
    
    This function selects a TSR object using weighted random sampling.
    It's a convenience wrapper around choose_tsr_index that returns
    the TSR object instead of its index.
    
    Args:
        tsrs: Sequence of TSR objects
        rng: Optional random number generator. If None, uses default RNG.
        
    Returns:
        Selected TSR object
        
    Examples:
        >>> # Create TSRs with different volumes
        >>> tsr1 = TSR(T0_w=np.eye(4), Tw_e=np.eye(4), 
        ...            Bw=np.array([[0,0], [0,0], [0,0], [0,0], [0,0], [-pi,pi]]))
        >>> tsr2 = TSR(T0_w=np.eye(4), Tw_e=np.eye(4),
        ...            Bw=np.array([[0,0], [0,0], [0,0], [0,0], [0,0], [0,0]]))
        >>> 
        >>> # Choose a TSR
        >>> selected = choose_tsr([tsr1, tsr2])
        >>> selected in [tsr1, tsr2]
        True
    """
    return tsrs[choose_tsr_index(tsrs, rng)]


def sample_from_tsrs(tsrs: Sequence[CoreTSR], rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Weighted-select a TSR and return a sampled 4×4 transform.
    
    This function combines TSR selection and sampling into a single operation.
    It first selects a TSR using weighted random sampling (based on volume),
    then samples a pose from that TSR.
    
    Args:
        tsrs: Sequence of TSR objects
        rng: Optional random number generator. If None, uses default RNG.
        
    Returns:
        4×4 transformation matrix representing a valid pose from one of the TSRs
        
    Examples:
        >>> # Create multiple TSRs for different grasp approaches
        >>> side_tsr = TSR(T0_w=np.eye(4), Tw_e=np.eye(4),
        ...                Bw=np.array([[0,0], [0,0], [-0.01,0.01], [0,0], [0,0], [-pi,pi]]))
        >>> top_tsr = TSR(T0_w=np.eye(4), Tw_e=np.eye(4),
        ...               Bw=np.array([[-0.01,0.01], [-0.01,0.01], [0,0], [0,0], [0,0], [-pi,pi]]))
        >>> 
        >>> # Sample from multiple TSRs
        >>> pose = sample_from_tsrs([side_tsr, top_tsr])
        >>> pose.shape
        (4, 4)
        >>> np.allclose(pose[3, :], [0, 0, 0, 1])  # Valid transform
        True
    """
    return choose_tsr(tsrs, rng).sample()


# (Optional) helpers for TSRTemplate lists
try:
    from tsr.core.tsr_template import TSRTemplate  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    TSRTemplate = object  # type: ignore[assignment]


def instantiate_templates(templates: Sequence["TSRTemplate"], T_ref_world: np.ndarray) -> List[CoreTSR]:
    """Instantiate a list of templates at a reference pose.
    
    This function converts a list of TSR templates into concrete TSRs
    by instantiating each template at the given reference pose.
    
    Args:
        templates: Sequence of TSRTemplate objects
        T_ref_world: 4×4 pose of the reference entity in world frame
        
    Returns:
        List of instantiated TSR objects
        
    Examples:
        >>> # Create templates for different grasp approaches
        >>> side_template = TSRTemplate(
        ...     T_ref_tsr=np.eye(4),
        ...     Tw_e=np.array([[0,0,1,-0.05], [1,0,0,0], [0,1,0,0.05], [0,0,0,1]]),
        ...     Bw=np.array([[0,0], [0,0], [-0.01,0.01], [0,0], [0,0], [-pi,pi]])
        ... )
        >>> top_template = TSRTemplate(
        ...     T_ref_tsr=np.eye(4),
        ...     Tw_e=np.array([[0,0,1,0], [1,0,0,0], [0,1,0,0], [0,0,0,1]]),
        ...     Bw=np.array([[-0.01,0.01], [-0.01,0.01], [0,0], [0,0], [0,0], [-pi,pi]])
        ... )
        >>> 
        >>> # Instantiate at object pose
        >>> object_pose = np.array([[1,0,0,0.5], [0,1,0,0], [0,0,1,0.3], [0,0,0,1]])
        >>> tsrs = instantiate_templates([side_template, top_template], object_pose)
        >>> len(tsrs)
        2
        >>> all(isinstance(tsr, TSR) for tsr in tsrs)
        True
    """
    return [tmpl.instantiate(T_ref_world) for tmpl in templates]


def sample_from_templates(
    templates: Sequence["TSRTemplate"], T_ref_world: np.ndarray, rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """Instantiate templates, weighted-select one TSR, and sample a transform.
    
    This function combines template instantiation, TSR selection, and sampling
    into a single operation. It's useful when you have multiple TSR templates
    and want to sample a pose from one of them.
    
    Args:
        templates: Sequence of TSRTemplate objects
        T_ref_world: 4×4 pose of the reference entity in world frame
        rng: Optional random number generator. If None, uses default RNG.
        
    Returns:
        4×4 transformation matrix representing a valid pose from one of the templates
        
    Examples:
        >>> # Create templates for different grasp approaches
        >>> side_template = TSRTemplate(
        ...     T_ref_tsr=np.eye(4),
        ...     Tw_e=np.array([[0,0,1,-0.05], [1,0,0,0], [0,1,0,0.05], [0,0,0,1]]),
        ...     Bw=np.array([[0,0], [0,0], [-0.01,0.01], [0,0], [0,0], [-pi,pi]])
        ... )
        >>> top_template = TSRTemplate(
        ...     T_ref_tsr=np.eye(4),
        ...     Tw_e=np.array([[0,0,1,0], [1,0,0,0], [0,1,0,0], [0,0,0,1]]),
        ...     Bw=np.array([[-0.01,0.01], [-0.01,0.01], [0,0], [0,0], [0,0], [-pi,pi]])
        ... )
        >>> 
        >>> # Sample from templates
        >>> object_pose = np.array([[1,0,0,0.5], [0,1,0,0], [0,0,1,0.3], [0,0,0,1]])
        >>> pose = sample_from_templates([side_template, top_template], object_pose)
        >>> pose.shape
        (4, 4)
        >>> np.allclose(pose[3, :], [0, 0, 0, 1])  # Valid transform
        True
    """
    tsrs = instantiate_templates(templates, T_ref_world)
    return sample_from_tsrs(tsrs, rng)