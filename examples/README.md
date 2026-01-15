# TSR Examples

Simple examples demonstrating the TSR library.

## Running Examples

```bash
# Run any example
uv run python examples/01_basic_tsr.py
```

## Examples

| Example | Description |
|---------|-------------|
| `01_basic_tsr.py` | Core TSR API: create, sample, contains, distance |
| `02_templates.py` | Loading human-readable YAML templates |
| `03_tsr_chains.py` | Chaining TSRs for articulated constraints |
| `04_placements.py` | Using reference_frame for placement constraints |

## Quick Overview

### 1. Basic TSR (`01_basic_tsr.py`)

Create TSRs directly with matrices:

```python
from tsr import TSR

tsr = TSR(T0_w=object_pose, Tw_e=gripper_offset, Bw=bounds)
pose = tsr.sample()
is_valid = tsr.contains(pose)
```

### 2. Templates (`02_templates.py`)

Load human-readable templates instead of writing matrices:

```python
from tsr.core.tsr_primitive import load_template_file

template = load_template_file("templates/grasps/mug_side_grasp.yaml")
tsr = TSR(T0_w=mug_pose, Tw_e=template.Tw_e, Bw=template.Bw)
```

### 3. TSR Chains (`03_tsr_chains.py`)

Chain TSRs for articulated objects like doors:

```python
from tsr import TSRChain

chain = TSRChain(TSRs=[hinge_tsr, handle_tsr])
pose = chain.sample()
```

### 4. Placements (`04_placements.py`)

Handle reference frames for placements:

```python
template = load_template_file("templates/places/mug_on_table.yaml")
print(template.reference_frame)  # "bottom"

# Transform from COM to bottom frame
mug_bottom = mug_com @ T_com_to_bottom
tsr = TSR(T0_w=mug_bottom, Tw_e=template.Tw_e, Bw=template.Bw)
```
