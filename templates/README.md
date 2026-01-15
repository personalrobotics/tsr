# TSR Templates

Human and LLM-friendly templates for Task Space Regions.

## Quick Start

```python
from tsr.core.tsr_primitive import load_template_file
from tsr.core.tsr import TSR
import numpy as np

# Load a template
template = load_template_file("templates/grasps/mug_side_grasp.yaml")

# Create TSR at object pose
object_pose = np.eye(4)
object_pose[0, 3] = 0.5  # object at x=0.5
tsr = TSR(T0_w=object_pose, Tw_e=template.Tw_e, Bw=template.Bw)

# Sample a valid pose
grasp_pose = tsr.sample()
```

## Directory Structure

```
templates/
├── grasps/     # Grasping objects
├── places/     # Placing objects
├── tasks/      # Complex manipulation tasks
└── README.md   # This file
```

## Template Format

Templates use a human-readable YAML format with geometric primitives:

```yaml
name: Mug Side Grasp (Avoiding Handle)
description: Grasp mug body from the side, keeping clear of handle
task: grasp
subject: gripper
reference: mug

position:
  type: cylinder        # geometric primitive
  axis: z
  radius: 0.04          # meters
  height: [0.03, 0.08]  # [min, max] meters
  angle: [45, 315]      # degrees - avoids handle at 0°

orientation:
  approach: radial      # gripper points toward cylinder axis

standoff: 0.05          # approach distance in meters

gripper:
  aperture: 0.09        # gripper opening
```

**Units:** All distances in **meters**, all angles in **degrees**.

## The 9 Geometric Primitives

| Primitive | Description | Key Parameters |
|-----------|-------------|----------------|
| **point** | Fixed location | `x`, `y`, `z` |
| **line** | Along one axis | `axis`, `range` |
| **plane** | Flat 2D region | two axes with ranges |
| **box** | 3D volume | `x`, `y`, `z` ranges |
| **ring** | Circle around axis | `axis`, `radius`, `angle` |
| **disk** | Filled circle | `axis`, `radius` range, `angle` |
| **cylinder** | Cylinder surface | `axis`, `radius`, `height`, `angle` |
| **shell** | Thick cylinder | `radius` range, `height` range |
| **sphere** | Spherical surface | `radius`, `pitch`, `yaw` |

## Available Templates

### Grasps (`grasps/`)

| Template | Primitive | Description |
|----------|-----------|-------------|
| `mug_side_grasp.yaml` | cylinder | Full 360° side grasp |
| `mug_side_grasp_avoid_handle.yaml` | cylinder | Side grasp avoiding handle region |
| `mug_handle_grasp.yaml` | point | Grasp the mug handle |
| `bottle_grasp.yaml` | cylinder | Symmetric bottle body grasp |
| `bowl_rim_grasp.yaml` | ring | Pinch bowl rim from any angle |
| `box_top_grasp.yaml` | plane | Top-down box grasp |
| `pen_grasp.yaml` | cylinder (x) | Grasp pen perpendicular to axis |
| `screwdriver_grasp.yaml` | cylinder (x) | Tool handle grasp |
| `spray_bottle_trigger.yaml` | point | Specific trigger-ready pose |
| `jar_lid_grasp.yaml` | shell | Grasp lid rim for twisting |
| `knife_handle_grasp.yaml` | line | Slide along knife handle |

### Placements (`places/`)

| Template | Primitive | Description |
|----------|-----------|-------------|
| `mug_on_table.yaml` | plane | Place mug upright on table |
| `bottle_in_rack.yaml` | line | Place bottle horizontally in rack |
| `plate_stack.yaml` | point | Stack plate centered on pile |
| `book_on_shelf.yaml` | box | Place book upright, spine out |
| `cup_on_coaster.yaml` | disk | Center cup on coaster |

### Tasks (`tasks/`)

| Template | Primitive | Description |
|----------|-----------|-------------|
| `pour_from_bottle.yaml` | point | Tilted pouring pose |
| `turn_valve.yaml` | ring | Grip valve wheel rim |
| `open_drawer.yaml` | line | Grasp drawer handle |
| `wipe_table.yaml` | plane | Keep sponge flat on surface |
| `handover_object.yaml` | sphere | Reachable handover zone |

## Examples by Primitive

### Point - Precise location

```yaml
# Grasp mug handle at specific point
position:
  type: point
  x: 0.06     # handle offset from center
  y: 0
  z: 0.05    # mid-height
```

### Line - Along one axis

```yaml
# Grasp anywhere along drawer handle
position:
  type: line
  axis: x
  range: [-0.05, 0.05]
```

### Plane - Flat region

```yaml
# Place anywhere on table surface
position:
  type: plane
  x: [-0.15, 0.15]
  y: [-0.15, 0.15]
  z: 0
```

### Cylinder - Around an axis

```yaml
# Side grasp avoiding handle (at 0°)
position:
  type: cylinder
  axis: z
  radius: 0.04
  height: [0.03, 0.08]
  angle: [45, 315]      # skip 0° region
```

### Ring - Circle at fixed height

```yaml
# Grasp valve wheel rim
position:
  type: ring
  axis: x
  radius: 0.08
  height: 0
  angle: [0, 360]
```

### Disk - Filled circle

```yaml
# Place cup centered on coaster
position:
  type: disk
  axis: z
  radius: [0, 0.02]     # must be nearly centered
  height: 0
  angle: [0, 360]
```

### Shell - Thick cylinder

```yaml
# Grasp jar lid rim
position:
  type: shell
  axis: z
  radius: [0.03, 0.035]   # outer rim
  height: [0, 0.015]
  angle: [0, 360]
```

### Sphere - Spherical region

```yaml
# Handover in reachable zone
position:
  type: sphere
  radius: 0.45
  pitch: [-30, 30]
  yaw: [-45, 45]
```

## Orientation Options

| Approach | Description |
|----------|-------------|
| `radial` | Point toward reference axis (for cylinders) |
| `axial` | Point along reference axis |
| `+x`, `-x`, `+y`, `-y`, `+z`, `-z` | Fixed direction |

Additional freedoms:
```yaml
orientation:
  approach: -z
  yaw: free           # or: [-45, 45]
  roll: [-10, 10]     # degrees
```

## Creating New Templates

1. Choose the appropriate primitive for your constraint
2. Define position bounds in the object's reference frame
3. Specify approach direction and any orientation freedoms
4. Set standoff distance for grasps
5. Test with the parser:

```python
from tsr.core.tsr_primitive import load_template_file
template = load_template_file("my_template.yaml")
print(template.Bw)  # Check the bounds
```

## Raw DOF Specification

For constraints outside the 9 primitives, use raw specification:

```yaml
position:
  type: raw
  x: 0.1
  y: [-0.05, 0.05]
  z: [0, 0.1]
  roll: 0
  pitch: [-10, 10]
  yaw: [0, 180]
```
