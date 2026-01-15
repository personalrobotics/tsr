# Task Space Regions (TSR)

A Python library for defining pose constraints in robotics manipulation using human-friendly geometric primitives.

Based on the IJRR paper ["Task Space Regions: A Framework for Pose-Constrained Manipulation Planning"](https://www.ri.cmu.edu/pub_files/2011/10/dmitry_ijrr10-1.pdf) by Berenson, Srinivasa, and Kuffner.

## Installation

**Add to your project:**
```bash
uv add git+https://github.com/personalrobotics/tsr.git
```

**Or in `pyproject.toml`:**
```toml
dependencies = [
    "tsr @ git+https://github.com/personalrobotics/tsr.git",
]
```

**For development:**
```bash
git clone https://github.com/personalrobotics/tsr.git
cd tsr
uv sync --extra test
```

## Quick Start

```python
from tsr.core.tsr_primitive import load_template_file
from tsr.core.tsr import TSR
import numpy as np

# Load a template
template = load_template_file("templates/grasps/mug_side_grasp.yaml")

# Create TSR at object pose
mug_pose = np.eye(4)
mug_pose[0, 3] = 0.5  # mug at x=0.5

tsr = TSR(T0_w=mug_pose, Tw_e=template.Tw_e, Bw=template.Bw)

# Sample valid grasp poses
grasp_pose = tsr.sample()

# Check if a pose is valid
distance, _ = tsr.distance(grasp_pose)
is_valid = tsr.contains(grasp_pose)
```

## Template Format

Templates use human-readable YAML with geometric primitives:

```yaml
name: Mug Side Grasp (Avoiding Handle)
description: Grasp mug body from the side, keeping clear of handle
task: grasp
subject: gripper
reference: mug

position:
  type: cylinder
  axis: z
  radius: 0.04
  height: [0.03, 0.08]
  angle: [45, 315]      # degrees - avoids handle at 0°

orientation:
  approach: radial      # gripper points toward cylinder axis

standoff: 0.05          # 5cm approach distance

gripper:
  aperture: 0.09
```

**Units:** distances in **meters**, angles in **degrees**.

## The 9 Geometric Primitives

TSRs can represent 9 fundamental shapes through their 6-DOF bounds:

| Primitive | Description | Use Case |
|-----------|-------------|----------|
| **point** | Fixed location | Precise grasp point |
| **line** | Along one axis | Drawer handle, knife grip |
| **plane** | Flat 2D region | Table placement |
| **box** | 3D volume | Tolerance region |
| **ring** | Circle around axis | Valve wheel, bowl rim |
| **disk** | Filled circle | Coaster placement |
| **cylinder** | Cylinder surface | Side grasps |
| **shell** | Thick cylinder | Jar lid rim |
| **sphere** | Spherical surface | Handover zone |

The key insight: **rotation bounds sweep position**. Setting `x=radius` and `yaw=[0°, 360°]` sweeps the point around the z-axis, creating a ring.

## Included Templates

The library includes 21 ready-to-use templates:

**Grasps (11):** mug (side, handle, avoid-handle), bottle, bowl rim, box top, pen, screwdriver, spray bottle, jar lid, knife handle

**Placements (5):** table, rack, shelf, stack, coaster

**Tasks (5):** pour, valve turn, drawer open, wipe surface, handover

See [`templates/README.md`](templates/README.md) for complete documentation with examples for each primitive.

## Orientation Options

| Approach | Description |
|----------|-------------|
| `radial` | Point toward reference axis |
| `axial` | Point along reference axis |
| `+x`, `-x`, `+y`, `-y`, `+z`, `-z` | Fixed direction |

Additional freedoms:
```yaml
orientation:
  approach: -z
  yaw: free           # or: [-45, 45]
  roll: [-10, 10]     # degrees
```

## Core TSR API

For advanced use, work directly with TSR components:

```python
from tsr.core.tsr import TSR
import numpy as np

# TSR defined by three components:
# - T0_w: Transform from world to TSR frame
# - Tw_e: Transform from TSR frame to end-effector
# - Bw: 6x2 bounds matrix [x, y, z, roll, pitch, yaw]

T0_w = np.eye(4)  # TSR at origin
Tw_e = np.eye(4)  # No offset
Bw = np.zeros((6, 2))
Bw[2, :] = [0, 0.1]           # z: 0 to 10cm
Bw[5, :] = [-np.pi, np.pi]    # yaw: full rotation

tsr = TSR(T0_w=T0_w, Tw_e=Tw_e, Bw=Bw)

pose = tsr.sample()              # Sample a pose
xyzrpy = tsr.sample_xyzrpy()     # Sample raw coordinates
distance, closest = tsr.distance(pose)  # Distance to TSR
is_inside = tsr.contains(pose)   # Containment check
```

## TSR Chains

For coupled constraints (e.g., door opening), use TSR chains:

```python
from tsr.core.tsr_chain import TSRChain

chain = TSRChain(
    constrain=True,  # Apply over whole trajectory
    TSRs=[hinge_tsr, handle_tsr]
)

pose = chain.sample()
```

## Testing

```bash
uv run python -m pytest tests/ -v
```

## License

BSD-2-Clause License - see LICENSE file.
