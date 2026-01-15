# Task Space Regions (TSR)

A **simulator-agnostic** Python library for representing, storing, and creating Task Space Regions (TSRs) - geometric models for pose constraints in robotics manipulation.

For a detailed description of TSRs and their uses, please refer to the 2010 IJRR paper entitled "Task Space Regions: A Framework for Pose-Constrained Manipulation Planning" by Dmitry Berenson, Siddhartha Srinivasa, and James Kuffner. A copy of this publication can be downloaded [here](https://www.ri.cmu.edu/pub_files/2011/10/dmitry_ijrr10-1.pdf).

## 🚀 Features

- **Core TSR Library**: Geometric pose constraint representation
- **TSR Templates**: Scene-agnostic TSR definitions with **semantic context**
- **Gripper Preshape**: Optional gripper configuration (DOF values) for templates
- **Relational Library**: Task-based TSR generation and querying with **template descriptions**
- **Advanced Sampling**: Weighted sampling from multiple TSRs
- **Schema System**: Controlled vocabulary for tasks and entities
- **YAML Serialization**: Human-readable template storage with semantic context
- **Template Libraries**: Easy sharing and version control of template collections
- **Performance Optimized**: Fast sampling and distance calculations

## 📦 Installation

**Add to your project (using uv):**
```bash
uv add git+https://github.com/personalrobotics/tsr.git
```

**Or add to `pyproject.toml`:**
```toml
dependencies = [
    "tsr @ git+https://github.com/personalrobotics/tsr.git",
]
```

**For development (clone and install):**
```bash
git clone https://github.com/personalrobotics/tsr.git
cd tsr
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

### Using Package Templates

```python
from tsr import load_package_template, list_available_templates

# Discover available templates
templates = list_available_templates()
print(templates)  # ['grasps/mug_side_grasp.yaml', 'places/mug_on_table.yaml']

# Load and use a template
mug_grasp = load_package_template("grasps", "mug_side_grasp.yaml")
object_pose = get_object_pose()  # Your object pose
tsr = mug_grasp.instantiate(object_pose)
pose = tsr.sample()
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

## 🎨 TSR Geometric Primitives

A powerful insight: TSRs can represent **9 fundamental geometric shapes** through their 6-DOF bounds `[x, y, z, roll, pitch, yaw]`. Each DOF can be fixed (single value) or a range, and rotation bounds can sweep position to create curved surfaces.

**Units:** All distances are in **meters**, all angles are in **degrees**.

### The 9 Primitives

**Cartesian Primitives** (position bounds only):

| Primitive | Description | DOF Pattern |
|-----------|-------------|-------------|
| **Point** | Single location | x, y, z all fixed |
| **Line** | Along one axis | 1 axis varies, 2 fixed |
| **Plane** | Flat 2D region | 2 axes vary, 1 fixed |
| **Box** | 3D volume | All 3 axes vary |

**Cylindrical Primitives** (rotation sweeps position):

| Primitive | Description | DOF Pattern |
|-----------|-------------|-------------|
| **Ring** | Circle/arc around axis | radius fixed, angle varies, height fixed |
| **Disk** | Filled circle/annulus | radius varies, angle varies, height fixed |
| **Cylinder** | Cylinder surface | radius fixed, angle varies, height varies |
| **Shell** | Thick-walled cylinder | radius varies, angle varies, height varies |

**Spherical Primitives** (multiple rotations sweep position):

| Primitive | Description | DOF Pattern |
|-----------|-------------|-------------|
| **Sphere** | Spherical surface | radius fixed, pitch varies, yaw varies |

### How It Works

The key insight is that **rotation bounds sweep the position**. For example, if you set `x=0.04` (fixed radius) and `yaw=[0°, 360°]`, the yaw rotation sweeps the point around the z-axis, creating a ring.

```
Raw DOF → Geometric Shape

x: 0.04 (fixed)     ┐
y: 0                │
z: [0.02, 0.08]     │ → Cylinder surface around z-axis
roll: 0             │
pitch: 0            │
yaw: [0°, 360°]     ┘
```

### Primitive Examples

**Point** - Precise grasp location:
```yaml
position:
  type: point
  x: 0
  y: 0
  z: 0.12
```

**Plane** - Placement on surface with tolerance:
```yaml
position:
  type: plane
  x: [-0.1, 0.1]
  y: [-0.1, 0.1]
  z: 0
```

**Cylinder** - Side grasp avoiding handle:
```yaml
position:
  type: cylinder
  axis: z
  radius: 0.04
  height: [0.02, 0.08]
  angle: [30, 330]      # degrees - excludes handle region
```

**Ring** - Grasp at fixed height, any angle:
```yaml
position:
  type: ring
  axis: z
  radius: 0.04
  angle: [0, 360]
```

**Disk** - Grasp with variable reach:
```yaml
position:
  type: disk
  axis: z
  radius: [0.03, 0.05]  # inner to outer radius
  angle: [0, 360]
```

**Shell** - Volumetric grasp region:
```yaml
position:
  type: shell
  axis: z
  radius: [0.03, 0.05]
  height: [0.02, 0.08]
  angle: [0, 360]
```

### Mapping to Raw TSR Bounds

Each primitive expands to the underlying `[x, y, z, roll, pitch, yaw]` bounds:

| Primitive | x | y | z | roll | pitch | yaw |
|-----------|---|---|---|------|-------|-----|
| Point | val | val | val | 0 | 0 | 0 |
| Line (z) | 0 | 0 | [a,b] | 0 | 0 | 0 |
| Plane (xy) | [a,b] | [c,d] | val | 0 | 0 | 0 |
| Box | [a,b] | [c,d] | [e,f] | 0 | 0 | 0 |
| Ring (z) | r | 0 | val | 0 | 0 | [θ₁,θ₂] |
| Disk (z) | [r₁,r₂] | 0 | val | 0 | 0 | [θ₁,θ₂] |
| Cylinder (z) | r | 0 | [a,b] | 0 | 0 | [θ₁,θ₂] |
| Shell (z) | [r₁,r₂] | 0 | [a,b] | 0 | 0 | [θ₁,θ₂] |
| Sphere | r | 0 | 0 | 0 | [φ₁,φ₂] | [θ₁,θ₂] |

This primitive system allows you to think geometrically while the library handles the underlying TSR math.

### Complete Template Format

The new human-friendly template format combines primitives with task semantics:

```yaml
name: Side grasp mug (avoiding handle)
description: Grasp mug from side, excluding handle region
task: grasp
subject: gripper
reference: mug

position:
  type: cylinder
  axis: z
  radius: 0.04
  height: [0.02, 0.08]
  angle: [30, 330]        # avoids handle at 0°

orientation:
  approach: radial        # gripper points toward cylinder axis

standoff: 0.05            # 5cm approach distance

gripper:
  aperture: 0.06          # gripper opening
```

**Template fields:**
- `name`, `description` - Human-readable identifiers
- `task` - Task type: `grasp`, `place`, `constrain`
- `subject` - What's being constrained (gripper, object)
- `reference` - Reference frame entity
- `position` - Geometric primitive (see above)
- `orientation` - Approach direction and freedoms
- `standoff` - Offset distance along approach
- `gripper` - Optional gripper configuration

**Orientation approaches:**
- `radial` - Point toward reference axis (for cylinder/ring grasps)
- `axial` - Point along reference axis
- `+x`, `-x`, `+y`, `-y`, `+z`, `-z` - Fixed direction

**Loading templates:**
```python
from tsr.core.tsr_primitive import load_template_file, load_template_yaml

# From file
template = load_template_file("templates/grasps/mug_side_grasp.yaml")

# From string
template = load_template_yaml(yaml_string)

# Access parsed components
print(template.Bw)      # 6x2 bounds array
print(template.Tw_e)    # 4x4 end-effector transform
```

### Example Templates

The library includes 21 ready-to-use templates covering common manipulation scenarios:

**Grasps:** mug (side, handle, avoid-handle), bottle, bowl rim, box top, pen, screwdriver, spray bottle, jar lid, knife handle

**Placements:** mug on table, bottle in rack, plate stack, book on shelf, cup on coaster

**Tasks:** pour, turn valve, open drawer, wipe surface, handover

See [`templates/README.md`](templates/README.md) for the complete reference with examples for each primitive.

## 🏗️ Architecture Components

### 1. TSR Templates

TSR templates are **scene-agnostic** TSR definitions that can be instantiated at any reference pose:

```python
# Create a template for grasping cylindrical objects with semantic context
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
    ]),
    subject_entity=EntityClass.GENERIC_GRIPPER,
    reference_entity=EntityClass.MUG,
    task_category=TaskCategory.GRASP,
    variant="side",
    name="Cylinder Side Grasp",
    description="Grasp a cylindrical object from the side with 10cm approach distance"
)

# Instantiate at a specific object pose
object_pose = get_object_pose()
tsr = template.instantiate(object_pose)
```

### 2. Template Generators

The library provides **template generators** for common primitive objects and tasks:

```python
from tsr import (
    generate_cylinder_grasp_template,
    generate_box_grasp_template,
    generate_place_template,
    generate_transport_template,
    generate_mug_grasp_template,
    generate_box_place_template
)

# Generate cylinder grasp templates
side_grasp = generate_cylinder_grasp_template(
    subject_entity=EntityClass.GENERIC_GRIPPER,
    reference_entity=EntityClass.MUG,
    variant="side",
    cylinder_radius=0.04,
    cylinder_height=0.12,
    approach_distance=0.05,
    preshape=np.array([0.08])  # 8cm aperture for parallel jaw gripper
)

# Generate box grasp templates
top_grasp = generate_box_grasp_template(
    subject_entity=EntityClass.GENERIC_GRIPPER,
    reference_entity=EntityClass.BOX,
    variant="top",
    box_length=0.15,
    box_width=0.10,
    box_height=0.08,
    approach_distance=0.03,
    preshape=np.array([0.0, 0.5, 0.5, 0.0, 0.5, 0.5])  # 6-DOF hand configuration
)

# Generate placement templates
place_template = generate_place_template(
    subject_entity=EntityClass.MUG,
    reference_entity=EntityClass.TABLE,
    variant="on",
    surface_height=0.0,
    placement_tolerance=0.1
)

# Use convenience functions
mug_grasp = generate_mug_grasp_template()  # Default mug parameters
box_place = generate_box_place_template()  # Default box placement
```

### 3. Gripper Preshape Configuration

TSR templates support **optional gripper preshape configuration** to specify the desired gripper state (DOF values) that should be achieved before or during TSR execution:

```python
# Parallel jaw gripper with specific aperture
parallel_grasp = generate_mug_grasp_template(
    variant="side",
    preshape=np.array([0.08])  # 8cm aperture
)

# Multi-finger hand with joint angle configuration
multi_finger_grasp = generate_box_grasp_template(
    variant="side_x",
    preshape=np.array([0.0, 0.5, 0.5, 0.0, 0.5, 0.5])  # 6-DOF hand
)

# Template without preshape (default behavior)
place_template = TSRTemplate(...)  # preshape will be None
```

**Preshape Features:**
- **Gripper-Aware TSRs**: Specify required gripper configurations
- **Flexible DOF Support**: Works with any gripper type (parallel jaw, multi-finger, etc.)
- **Optional Field**: Backward compatible - preshape is `None` by default
- **Serialization Support**: Preshape values are preserved in YAML/JSON
- **Library Integration**: Preshape information available in relational library queries
```

### 4. Bundled Template Access

The package includes **21 pre-built templates** that can be loaded directly:

```python
from tsr.core.tsr_primitive import load_template_file

# Load a grasp template
template = load_template_file("templates/grasps/mug_side_grasp.yaml")

# Use with TSR
from tsr.core.tsr import TSR
import numpy as np

object_pose = np.eye(4)
tsr = TSR(T0_w=object_pose, Tw_e=template.Tw_e, Bw=template.Bw)
grasp_pose = tsr.sample()
```

**Included templates:**
- **Grasps (11)**: mug, bottle, bowl, box, pen, screwdriver, spray bottle, jar lid, knife
- **Placements (5)**: table, rack, shelf, stack, coaster
- **Tasks (5)**: pour, valve, drawer, wipe, handover

See [`templates/README.md`](templates/README.md) for the complete list.

### 5. Schema System

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

### 6. Relational Library

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
        Bw=np.array([[0, 0], [0, 0], [-0.01, 0.01], [0, 0], [0, 0], [-np.pi, np.pi]]),
        subject_entity=EntityClass.GENERIC_GRIPPER,
        reference_entity=EntityClass.MUG,
        task_category=TaskCategory.GRASP,
        variant="side"
    )
    return [side_template]

def mug_place_generator(T_ref_world):
    """Generate TSR templates for placing a mug."""
    place_template = TSRTemplate(
        T_ref_tsr=np.eye(4),
        Tw_e=np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.02], [0, 0, 0, 1]]),
        Bw=np.array([[-0.1, 0.1], [-0.1, 0.1], [0, 0], [0, 0], [0, 0], [-np.pi/4, np.pi/4]]),
        subject_entity=EntityClass.MUG,
        reference_entity=EntityClass.TABLE,
        task_category=TaskCategory.PLACE,
        variant="on"
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

### 7. Enhanced Template-Based Library

The library also supports **direct template registration** with descriptions for easier management:

```python
# Register templates directly with descriptions
library.register_template(
    subject=EntityClass.GENERIC_GRIPPER,
    reference=EntityClass.MUG,
    task=TaskType(TaskCategory.GRASP, "side"),
    template=side_template,
    description="Side grasp with 5cm approach distance"
)

library.register_template(
    subject=EntityClass.GENERIC_GRIPPER,
    reference=EntityClass.MUG,
    task=TaskType(TaskCategory.GRASP, "top"),
    template=top_template,
    description="Top grasp with vertical approach"
)

# Query templates with descriptions
templates_with_desc = library.query_templates(
    EntityClass.GENERIC_GRIPPER,
    EntityClass.MUG,
    TaskType(TaskCategory.GRASP, "side"),
    include_descriptions=True
)

# Browse available templates
available = library.list_available_templates(
    subject=EntityClass.GENERIC_GRIPPER,
    task_category="grasp"
)

# Get template information
info = library.get_template_info(
    EntityClass.GENERIC_GRIPPER,
    EntityClass.MUG,
    TaskType(TaskCategory.GRASP, "side")
)
```


### 8. Advanced Sampling

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

TSRs, TSR chains, and **TSR templates** can be serialized to multiple formats:

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

# TSR Template serialization with semantic context
template_yaml = template.to_yaml()
template_from_yaml = TSRTemplate.from_yaml(template_yaml)
```

### YAML Template Example

Templates serialize to **human-readable YAML** with full semantic context:

```yaml
name: Cylinder Side Grasp
description: Grasp a cylindrical object from the side with 10cm approach distance
subject_entity: generic_gripper
reference_entity: mug
task_category: grasp
variant: side
T_ref_tsr:
  - [1.0, 0.0, 0.0, 0.0]
  - [0.0, 1.0, 0.0, 0.0]
  - [0.0, 0.0, 1.0, 0.0]
  - [0.0, 0.0, 0.0, 1.0]
Tw_e:
  - [0.0, 0.0, 1.0, -0.1]    # Approach from -z, 10cm offset
  - [1.0, 0.0, 0.0, 0.0]     # x-axis perpendicular to cylinder
  - [0.0, 1.0, 0.0, 0.05]    # y-axis along cylinder axis
  - [0.0, 0.0, 0.0, 1.0]
Bw:
  - [0.0, 0.0]               # x: fixed position
  - [0.0, 0.0]               # y: fixed position
  - [-0.01, 0.01]            # z: small tolerance
  - [0.0, 0.0]               # roll: fixed
  - [0.0, 0.0]               # pitch: fixed
  - [-3.14159, 3.14159]      # yaw: full rotation
```

### Template Library Serialization

Save and load entire template libraries:

```python
# Save template library to YAML
templates = [template1, template2, template3]
template_library = [t.to_dict() for t in templates]

import yaml
with open('grasp_templates.yaml', 'w') as f:
    yaml.dump(template_library, f, default_flow_style=False)

# Load template library from YAML
with open('grasp_templates.yaml', 'r') as f:
    loaded_library = yaml.safe_load(f)
    loaded_templates = [TSRTemplate.from_dict(t) for t in loaded_library]
```

## 📖 Examples

The library includes comprehensive examples demonstrating all features:

```bash
# Run all examples
uv run python examples/run_all_examples.py

# Run individual examples
uv run python examples/01_basic_tsr.py          # Basic TSR creation and sampling
uv run python examples/02_tsr_chains.py         # TSR chain composition
uv run python examples/03_tsr_templates.py      # Template creation and instantiation
uv run python examples/04_relational_library.py # Library registration and querying
uv run python examples/05_sampling.py           # Advanced sampling techniques
uv run python examples/06_serialization.py      # YAML serialization with semantic context
uv run python examples/07_template_file_management.py  # Template file organization
uv run python examples/08_template_generators.py       # Template generators for primitive objects
uv run python examples/09_template_access.py           # Bundled template access demonstration
uv run python examples/10_preshape_example.py          # Gripper preshape configuration examples

### Example Output: YAML Serialization

The serialization example demonstrates the new YAML features:

```yaml
# Template library with semantic context
- name: Mug Side Grasp
  description: Grasp mug from the side
  subject_entity: generic_gripper
  reference_entity: mug
  task_category: grasp
  variant: side
  T_ref_tsr: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
  Tw_e: [[0, 0, 1, -0.05], [1, 0, 0, 0], [0, 1, 0, 0.05], [0, 0, 0, 1]]
  Bw: [[0, 0], [0, 0], [-0.01, 0.01], [0, 0], [0, 0], [-3.14159, 3.14159]]

- name: Table Placement
  description: Place mug on table surface
  subject_entity: mug
  reference_entity: table
  task_category: place
  variant: on
  T_ref_tsr: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
  Tw_e: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.02], [0, 0, 0, 1]]
  Bw: [[-0.1, 0.1], [0, 0], [0, 0], [0, 0], [0, 0], [-0.785398, 0.785398]]
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

## 🎯 Key Benefits

### Semantic Context & YAML Serialization
- **Self-Documenting Templates**: Human-readable YAML with clear semantic meaning
- **Template Libraries**: Easy sharing and version control of template collections
- **Rich Integration**: Semantic context enables better task-based generation
- **Backward Compatibility**: Existing code continues to work seamlessly

### Enhanced Library Management
- **Template Descriptions**: Document and browse available templates
- **Flexible Registration**: Both generator-based and template-based approaches
- **Rich Querying**: Filter and search templates by semantic criteria
- **Template Browsing**: Discover available templates with descriptions

### Bundled Templates
- **21 Ready-to-Use Templates**: Grasps, placements, and manipulation tasks
- **Human-Friendly Format**: YAML with geometric primitives
- **LLM Authorable**: Simple enough for language models to generate
- **9 Geometric Primitives**: point, line, plane, box, ring, disk, cylinder, shell, sphere

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
