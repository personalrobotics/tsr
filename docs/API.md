# TSR Library API Documentation

This document provides comprehensive API documentation for the Task Space Regions (TSR) library.

## Table of Contents

1. [Core TSR Classes](#core-tsr-classes)
2. [TSR Templates](#tsr-templates)
3. [Schema System](#schema-system)
4. [Relational Library](#relational-library)
5. [Sampling Utilities](#sampling-utilities)
6. [Serialization](#serialization)
7. [Utility Functions](#utility-functions)

## Core TSR Classes

### TSR

The core Task Space Region class for representing pose constraints.

```python
class TSR:
    def __init__(self, T0_w=None, Tw_e=None, Bw=None):
        """
        Initialize a TSR.
        
        Args:
            T0_w: 4×4 transform from world frame to TSR frame (default: identity)
            Tw_e: 4×4 transform from TSR frame to end-effector frame (default: identity)
            Bw: (6,2) bounds matrix for [x,y,z,roll,pitch,yaw] (default: zeros)
        """
```

**Methods:**

- `sample(xyzrpy=NANBW) -> np.ndarray`: Sample a 4×4 transform from the TSR
- `contains(trans) -> bool`: Check if a transform is within the TSR bounds
- `distance(trans) -> tuple[float, np.ndarray]`: Compute geodesic distance to TSR
- `to_dict() -> dict`: Convert TSR to dictionary
- `from_dict(x) -> TSR`: Create TSR from dictionary
- `to_json() -> str`: Convert TSR to JSON string
- `from_json(x) -> TSR`: Create TSR from JSON string

### TSRChain

Compose multiple TSRs for complex constraints.

```python
class TSRChain:
    def __init__(self, sample_start=False, sample_goal=False, constrain=False, TSRs=None):
        """
        Initialize a TSR chain.
        
        Args:
            sample_start: Whether to sample start pose
            sample_goal: Whether to sample goal pose
            constrain: Whether to apply constraint over trajectory
            TSRs: List of TSR objects
        """
```

**Methods:**

- `append(tsr)`: Add TSR to chain
- `sample() -> np.ndarray`: Sample pose from chain
- `contains(trans) -> bool`: Check if transform satisfies chain
- `distance(trans) -> tuple[float, np.ndarray]`: Compute distance to chain

## TSR Templates

### TSRTemplate

Scene-agnostic TSR definitions that can be instantiated at any reference pose.

```python
@dataclass(frozen=True)
class TSRTemplate:
    T_ref_tsr: np.ndarray  # Transform from reference frame to TSR frame
    Tw_e: np.ndarray       # Transform from TSR frame to subject frame
    Bw: np.ndarray         # (6,2) bounds in TSR frame
```

**Methods:**

- `instantiate(T_ref_world: np.ndarray) -> TSR`: Create concrete TSR at reference pose

**Example:**
```python
# Create template for grasping cylindrical objects
template = TSRTemplate(
    T_ref_tsr=np.eye(4),
    Tw_e=np.array([
        [0, 0, 1, -0.05],  # Approach from -z, 5cm offset
        [1, 0, 0, 0],      # x-axis perpendicular to cylinder
        [0, 1, 0, 0.05],   # y-axis along cylinder axis
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

# Instantiate at specific object pose
object_pose = np.array([[1,0,0,0.5], [0,1,0,0], [0,0,1,0.3], [0,0,0,1]])
tsr = template.instantiate(object_pose)
```

## Schema System

### TaskCategory

Controlled vocabulary for high-level manipulation tasks.

```python
class TaskCategory(str, Enum):
    GRASP = "grasp"      # Pick up an object
    PLACE = "place"      # Put down an object
    DISCARD = "discard"  # Throw away an object
    INSERT = "insert"    # Insert object into receptacle
    INSPECT = "inspect"  # Examine object closely
    PUSH = "push"        # Push/move object
    ACTUATE = "actuate"  # Operate controls/mechanisms
```

### TaskType

Structured task type combining category and variant.

```python
@dataclass(frozen=True)
class TaskType:
    category: TaskCategory
    variant: str  # e.g., "side", "on", "opening"
    
    def __str__(self) -> str:
        """Return 'category/variant' string representation."""
        
    @staticmethod
    def from_str(s: str) -> "TaskType":
        """Create TaskType from 'category/variant' string."""
```

**Example:**
```python
grasp_side = TaskType(TaskCategory.GRASP, "side")
place_on = TaskType(TaskCategory.PLACE, "on")
print(grasp_side)  # "grasp/side"
print(place_on)    # "place/on"
```

### EntityClass

Unified vocabulary for scene entities.

```python
class EntityClass(str, Enum):
    # Grippers/tools
    GENERIC_GRIPPER = "generic_gripper"
    ROBOTIQ_2F140 = "robotiq_2f140"
    SUCTION = "suction"
    
    # Objects/fixtures
    MUG = "mug"
    BIN = "bin"
    PLATE = "plate"
    BOX = "box"
    TABLE = "table"
    SHELF = "shelf"
    VALVE = "valve"
```

## Relational Library

### TSRLibraryRelational

Registry for task-based TSR generation and querying.

```python
class TSRLibraryRelational:
    def __init__(self):
        """Initialize empty relational TSR library."""
```

**Methods:**

- `register(subject, reference, task, generator)`: Register TSR generator
- `query(subject, reference, task, T_ref_world) -> List[TSR]`: Query TSRs
- `list_tasks_for_reference(reference, subject_filter=None, task_prefix=None) -> List[TaskType]`: List available tasks

**Example:**
```python
# Define TSR generator
def mug_grasp_generator(T_ref_world):
    """Generate TSR templates for grasping a mug."""
    side_template = TSRTemplate(
        T_ref_tsr=np.eye(4),
        Tw_e=np.array([[0,0,1,-0.05], [1,0,0,0], [0,1,0,0.05], [0,0,0,1]]),
        Bw=np.array([[0,0], [0,0], [-0.01,0.01], [0,0], [0,0], [-np.pi,np.pi]])
    )
    return [side_template]

# Register generator
library = TSRLibraryRelational()
library.register(
    subject=EntityClass.GENERIC_GRIPPER,
    reference=EntityClass.MUG,
    task=TaskType(TaskCategory.GRASP, "side"),
    generator=mug_grasp_generator
)

# Query TSRs
mug_pose = np.array([[1,0,0,0.5], [0,1,0,0], [0,0,1,0.3], [0,0,0,1]])
tsrs = library.query(
    subject=EntityClass.GENERIC_GRIPPER,
    reference=EntityClass.MUG,
    task=TaskType(TaskCategory.GRASP, "side"),
    T_ref_world=mug_pose
)
```

## Sampling Utilities

### Core Functions

- `weights_from_tsrs(tsrs: Sequence[TSR]) -> np.ndarray`: Compute weights proportional to TSR volumes
- `choose_tsr_index(tsrs: Sequence[TSR], rng=None) -> int`: Choose TSR index with weighted sampling
- `choose_tsr(tsrs: Sequence[TSR], rng=None) -> TSR`: Choose TSR with weighted sampling
- `sample_from_tsrs(tsrs: Sequence[TSR], rng=None) -> np.ndarray`: Sample pose from multiple TSRs

### Template Functions

- `instantiate_templates(templates: Sequence[TSRTemplate], T_ref_world: np.ndarray) -> List[TSR]`: Instantiate templates
- `sample_from_templates(templates: Sequence[TSRTemplate], T_ref_world: np.ndarray, rng=None) -> np.ndarray`: Sample from templates

**Example:**
```python
# Create multiple TSRs for different grasp approaches
side_tsr = TSR(T0_w=np.eye(4), Tw_e=np.eye(4),
               Bw=np.array([[0,0], [0,0], [-0.01,0.01], [0,0], [0,0], [-np.pi,np.pi]]))
top_tsr = TSR(T0_w=np.eye(4), Tw_e=np.eye(4),
              Bw=np.array([[-0.01,0.01], [-0.01,0.01], [0,0], [0,0], [0,0], [-np.pi,np.pi]]))

# Sample from multiple TSRs
pose = sample_from_tsrs([side_tsr, top_tsr])

# Get weights for analysis
weights = weights_from_tsrs([side_tsr, top_tsr])
```

## Serialization

### TSR Serialization

```python
# Dictionary format
tsr_dict = tsr.to_dict()
tsr_from_dict = TSR.from_dict(tsr_dict)

# JSON format
tsr_json = tsr.to_json()
tsr_from_json = TSR.from_json(tsr_json)

# YAML format (requires PyYAML)
tsr_yaml = tsr.to_yaml()
tsr_from_yaml = TSR.from_yaml(tsr_yaml)
```

### TSRChain Serialization

```python
# Dictionary format
chain_dict = chain.to_dict()
chain_from_dict = TSRChain.from_dict(chain_dict)

# JSON format
chain_json = chain.to_json()
chain_from_json = TSRChain.from_json(chain_json)
```

## Utility Functions

### Angle Wrapping

- `wrap_to_interval(angles: np.ndarray, lower_bound: float = -np.pi) -> np.ndarray`: Wrap angles to interval

### Distance Calculations

- `geodesic_distance(T1: np.ndarray, T2: np.ndarray, weight: float = 1.0) -> float`: Compute geodesic distance between transforms
- `geodesic_error(T1: np.ndarray, T2: np.ndarray, weight: float = 1.0) -> tuple[float, float]`: Compute geodesic error components

**Example:**
```python
# Wrap angles to [-π, π]
angles = np.array([3*np.pi, -2*np.pi, np.pi/2])
wrapped = wrap_to_interval(angles)
# Result: [π, 0, π/2]

# Compute distance between transforms
T1 = np.eye(4)
T2 = np.array([[1,0,0,1], [0,1,0,0], [0,0,1,0], [0,0,0,1]])
distance = geodesic_distance(T1, T2)
```

## Error Handling

The library uses standard Python exceptions:

- `ValueError`: Invalid input parameters (e.g., wrong array shapes)
- `KeyError`: No generator registered for entity/task combination
- `TypeError`: Incorrect argument types

## Performance Notes

- **Sampling**: < 1ms per TSR sample
- **Distance calculations**: < 10ms for complex TSRs
- **Memory efficient**: Minimal overhead for large TSR libraries
- **Thread-safe**: Safe for concurrent access

## Best Practices

1. **Use TSR templates** for reusable, scene-agnostic TSR definitions
2. **Register generators** in the relational library for task-based TSR generation
3. **Use weighted sampling** when multiple TSRs are available
4. **Cache TSR instances** when the same template is used repeatedly
5. **Validate inputs** before creating TSRs to avoid runtime errors

