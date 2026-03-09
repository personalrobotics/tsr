# TSR Examples

Runnable scripts demonstrating the TSR library.

```bash
uv run python examples/<script>.py
```

## Examples

| Example | Description |
|---------|-------------|
| `parallel_jaw_grasp.py` | Generate and visualize cylinder side-grasp templates with a parallel jaw gripper |
| `01_basic_tsr.py` | Core TSR API: create, sample, contains, distance |
| `02_templates.py` | Save and load serialized TSRTemplate YAML files |
| `03_tsr_chains.py` | Chaining TSRs for articulated constraints |
| `04_placements.py` | Placement constraints with reference frame offsets |

## Grasp template generation (`parallel_jaw_grasp.py`)

The canonical pattern for generating grasp TSRs from object geometry:

```python
from examples.parallel_jaw_grasp import ParallelJawGripper

gripper = ParallelJawGripper(finger_length=0.055, max_aperture=0.140)

# Returns 2*k TSRTemplates: k depth levels × 2 roll orientations
templates = gripper.grasp_cylinder(
    object_radius=0.040,
    height_range=(0.02, 0.10),
    reference="mug",
)
# "Mug Cylinder Side Grasp — shallow, roll 0°"
# "Mug Cylinder Side Grasp — shallow, roll 180°"
# "Mug Cylinder Side Grasp — mid, roll 0°"
# ...

# Bind to a scene object and sample
tsr = templates[0].instantiate(mug_pose)
grasp_pose = tsr.sample()
```

Key design decisions:
- `preshape` defaults to `2 * object_radius + clearance` (minimum viable jaw opening)
- `clearance` (default 10% of `finger_length`) applies to height ends and radial depth limits
- Radial approach cannot be encoded in `Bw` (couples with yaw), so k discrete depths are baked into `Tw_e`

## Basic TSR (`01_basic_tsr.py`)

```python
from tsr import TSR
import numpy as np

Bw = np.zeros((6, 2))
Bw[2, :] = [0.0, 0.10]       # z: 0–10 cm
Bw[5, :] = [-np.pi, np.pi]   # yaw: full rotation

tsr = TSR(T0_w=object_pose, Tw_e=gripper_offset, Bw=Bw)
pose     = tsr.sample()
is_valid = tsr.contains(pose)
distance, _ = tsr.distance(pose)
```

## Templates (`02_templates.py`)

```python
from tsr import TSRTemplate, save_template, load_template

template = TSRTemplate(
    T_ref_tsr=np.eye(4), Tw_e=Tw_e, Bw=Bw,
    task="grasp", subject="gripper", reference="mug",
    name="Mug Side Grasp",
)
save_template(template, "my_grasp.yaml")
template = load_template("my_grasp.yaml")

tsr = template.instantiate(mug_pose)
```

## TSR Chains (`03_tsr_chains.py`)

```python
from tsr import TSRChain

chain = TSRChain(TSRs=[hinge_tsr, handle_tsr])
pose  = chain.sample()
```
