# Task Space Regions (TSR)

A **simulator-agnostic** Python library for representing, storing, and creating Task Space Regions (TSRs) - geometric models for pose constraints in robotics manipulation.

For a detailed description of TSRs and their uses, please refer to the 2010 IJRR paper entitled "Task Space Regions: A Framework for Pose-Constrained Manipulation Planning" by Dmitry Berenson, Siddhartha Srinivasa, and James Kuffner. A copy of this publication can be downloaded [here](https://www.ri.cmu.edu/pub_files/2011/10/dmitry_ijrr10-1.pdf).

## 🚀 Features

- **Core TSR Library**: Geometric pose constraint representation
- **TSR Templates**: Scene-agnostic TSR definitions
- **Relational Library**: Task-based TSR generation and querying
- **Advanced Sampling**: Weighted sampling from multiple TSRs
- **Schema System**: Controlled vocabulary for tasks and entities
- **Serialization**: JSON, YAML, and dictionary formats
- **Performance Optimized**: Fast sampling and distance calculations

## 📦 Installation

This project uses [uv](https://github.com/astral-sh/uv) for dependency management:

```bash
# Install uv if you haven't already
pip install uv

# Clone and install the package
git clone https://github.com/personalrobotics/tsr.git
cd tsr
uv sync
```

For development with testing dependencies:
```bash
uv sync --extra test
```

## 🎯 Quick Start

```python
from tsr import TSR, TSRTemplate, TSRLibraryRelational, TaskType, TaskCategory, EntityClass
import numpy as np

# Create a simple TSR for grasping
T0_w = np.eye(4)  # World to TSR frame transform
Tw_e = np.eye(4)  # TSR frame to end-effector transform
Bw = np.zeros((6, 2))
Bw[2, :] = [0.0, 0.02]  # Allow vertical movement
Bw[5, :] = [-np.pi, np.pi]  # Allow any yaw rotation

tsr = TSR(T0_w=T0_w, Tw_e=Tw_e, Bw=Bw)
pose = tsr.sample()  # Sample a valid pose
```

## 📚 Core Concepts

### TSR Overview

A TSR defines a constraint on the pose of a robot's end-effector. For example, when grasping a glass, the end-effector must be near the glass and oriented to allow finger closure around it.

A TSR is defined by three components:
- `T0_w` - Transform from world frame to TSR frame
- `Tw_e` - Transform from TSR frame to end-effector frame  
- `Bw` - 6×2 matrix of bounds on TSR coordinates

The first three rows of `Bw` bound translation along x,y,z axes (meters). The last three rows bound rotation about those axes using Roll-Pitch-Yaw (radians).

### Example: Glass Grasping TSR

```python
# Define the glass's coordinate frame as the TSR frame
T0_w = glass_transform  # 4×4 matrix defining glass pose in world

# Define desired end-effector pose relative to glass
Tw_e = np.array([
    [0., 0., 1., -0.20],  # Approach from -z, 20cm offset
    [1., 0., 0., 0.],     # x-axis perpendicular to glass
    [0., 1., 0., 0.08],   # y-axis along glass height
    [0., 0., 0., 1.]
])

Bw = np.zeros((6, 2))
Bw[2, :] = [0.0, 0.02]    # Allow small vertical movement
Bw[5, :] = [-np.pi, np.pi]  # Allow any orientation about z-axis

grasp_tsr = TSR(T0_w=T0_w, Tw_e=Tw_e, Bw=Bw)

# Sample a valid grasp pose
ee_pose = grasp_tsr.sample()

# Check if current pose meets constraint
current_pose = get_end_effector_pose()
dist_to_tsr = grasp_tsr.distance(current_pose)
is_valid = (dist_to_tsr == 0.0)
```

## 🏗️ Architecture Components

### 1. TSR Templates

TSR templates are **scene-agnostic** TSR definitions that can be instantiated at any reference pose:

```python
# Create a template for grasping cylindrical objects
template = TSRTemplate(
    T_ref_tsr=np.eye(4),  # Reference frame to TSR frame
    Tw_e=np.array([
        [0, 0, 1, -0.1],  # Approach from -z, 10cm offset
        [1, 0, 0, 0],     # x-axis perpendicular to cylinder
        [0, 1, 0, 0.05],  # y-axis along cylinder axis
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

# Instantiate at a specific object pose
object_pose = get_object_pose()
tsr = template.instantiate(object_pose)
```

### 2. Schema System

The schema provides a **controlled vocabulary** for defining tasks and entities:

```python
from tsr.schema import TaskCategory, TaskType, EntityClass

# Define task types
grasp_side = TaskType(TaskCategory.GRASP, "side")
grasp_top = TaskType(TaskCategory.GRASP, "top")
place_on = TaskType(TaskCategory.PLACE, "on")
place_in = TaskType(TaskCategory.PLACE, "in")

# Entity classes
gripper = EntityClass.ROBOTIQ_2F140
mug = EntityClass.MUG
table = EntityClass.TABLE

# Task strings
print(grasp_side)  # "grasp/side"
print(place_on)    # "place/on"
```

### 3. Relational Library

The relational library enables **task-based TSR generation** and querying:

Conceptually, the relational library treats a TSR as describing a spatial relationship between two entities: the **subject** and the **reference**. The subject is the entity whose pose is constrained (often a gripper or manipulated object), and the reference is the entity relative to which the TSR is defined (often a grasped object, a placement surface, or another tool). This formulation makes TSRs manipulator-agnostic and reusable: for example, `subject=GENERIC_GRIPPER` and `reference=MUG` with a `GRASP/side` task describes all side grasps for a mug, while `subject=MUG` and `reference=TABLE` with a `PLACE/on` task describes stable placements of a mug on a table. Querying the library with different subject–reference–task combinations allows you to retrieve the appropriate TSR templates for your current scene and entities.

```python
from tsr.tsr_library_rel import TSRLibraryRelational

# Define TSR generators
def mug_grasp_generator(T_ref_world):
    """Generate TSR templates for grasping a mug."""
    side_template = TSRTemplate(
        T_ref_tsr=np.eye(4),
        Tw_e=np.array([[0, 0, 1, -0.05], [1, 0, 0, 0], [0, 1, 0, 0.05], [0, 0, 0, 1]]),
        Bw=np.array([[0, 0], [0, 0], [-0.01, 0.01], [0, 0], [0, 0], [-np.pi, np.pi]])
    )
    return [side_template]

def mug_place_generator(T_ref_world):
    """Generate TSR templates for placing a mug."""
    place_template = TSRTemplate(
        T_ref_tsr=np.eye(4),
        Tw_e=np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.02], [0, 0, 0, 1]]),
        Bw=np.array([[-0.1, 0.1], [-0.1, 0.1], [0, 0], [0, 0], [0, 0], [-np.pi/4, np.pi/4]])
    )
    return [place_template]

# Register generators
library = TSRLibraryRelational()
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
mug_pose = get_mug_pose()
table_pose = get_table_pose()

grasp_tsrs = library.query(
    subject=EntityClass.GENERIC_GRIPPER,
    reference=EntityClass.MUG,
    task=TaskType(TaskCategory.GRASP, "side"),
    T_ref_world=mug_pose
)

place_tsrs = library.query(
    subject=EntityClass.MUG,
    reference=EntityClass.TABLE,
    task=TaskType(TaskCategory.PLACE, "on"),
    T_ref_world=table_pose
)

# Discover available tasks
mug_tasks = library.list_tasks_for_reference(EntityClass.MUG)
table_tasks = library.list_tasks_for_reference(EntityClass.TABLE)
```

### 4. Advanced Sampling

The library provides **weighted sampling** utilities for working with multiple TSRs:

```python
from tsr.sampling import weights_from_tsrs, choose_tsr, sample_from_tsrs, sample_from_templates

# Get weights proportional to TSR volumes
weights = weights_from_tsrs(tsr_list)

# Choose a TSR with probability proportional to its volume
selected_tsr = choose_tsr(tsr_list)

# Sample directly from a list of TSRs
pose = sample_from_tsrs(tsr_list)

# Sample from templates
templates = [template1, template2, template3]
pose = sample_from_templates(templates, object_pose)
```

## 🔗 TSR Chains

For complex constraints involving multiple TSRs, use TSR chains:

```python
from tsr.core.tsr_chain import TSRChain

# Example: Opening a refrigerator door
# First TSR: handle constraint relative to hinge
hinge_tsr = TSR(T0_w=hinge_pose, Tw_e=handle_offset, Bw=handle_bounds)

# Second TSR: end-effector constraint relative to handle  
ee_tsr = TSR(T0_w=np.eye(4), Tw_e=ee_in_handle, Bw=ee_bounds)

# Compose into a chain
chain = TSRChain(
    sample_start=False,
    sample_goal=False,
    constrain=True,  # Apply constraint over whole trajectory
    TSRs=[hinge_tsr, ee_tsr]
)
```

## 📊 Serialization

TSRs and TSR chains can be serialized to multiple formats:

```python
# Dictionary format
tsr_dict = tsr.to_dict()
tsr_from_dict = TSR.from_dict(tsr_dict)

# JSON format
tsr_json = tsr.to_json()
tsr_from_json = TSR.from_json(tsr_json)

# YAML format
tsr_yaml = tsr.to_yaml()
tsr_from_yaml = TSR.from_yaml(tsr_yaml)
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all tests
uv run python -m pytest tests/ -v

# Run specific test categories
uv run python -m pytest tests/tsr/ -v  # Core functionality
uv run python -m pytest tests/benchmarks/ -v  # Performance tests
```

## 📈 Performance

The library is optimized for real-time robotics applications:

- **Fast sampling**: < 1ms per TSR sample
- **Efficient distance calculations**: < 10ms for complex TSRs
- **Memory efficient**: Minimal overhead for large TSR libraries
- **Thread-safe**: Safe for concurrent access

## 🤝 Contributing

This library is designed to be **simulator-agnostic** and focuses on providing a rich interface for representing, storing, and creating TSRs. Contributions are welcome!

## 📄 License

This project is licensed under the BSD-2-Clause License - see the LICENSE file for details.
