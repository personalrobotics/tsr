# TSR Template Library

Hand-authored TSR templates for two manipulation narratives. Each template is a
YAML file loadable via `load_template()` / `load_package_template()`.

## Assurance boundary

These are **illustrative recipes, not geometrically certified grasps.** They specify
relative pose constraints and deliberately omit the primitive dimensions and gripper
geometry that the analytic grasp oracle needs, so none of the geometric-soundness
guarantees that apply to the `grasp_*` factory output (see `docs/ARCHITECTURE.md`)
apply here. What is guaranteed, and tested for every file in the default suite, is
**representation validity**: the file loads through the public API, instantiates, and
satisfies `sample() ⇔ contains() ⇔ distance ≈ 0`. Whether a recipe is geometrically
feasible for your gripper and object in your scene is yours to check.

## Narratives

### 1. Tool Use — Screwdriver

Pick up a screwdriver, drive a screw, drop the tool into a toolchest.

| Step | File | task | subject | reference |
|------|------|------|---------|-----------|
| Grasp screwdriver | `grasps/screwdriver_grasp.yaml` | grasp | gripper | screwdriver |
| Drive screw | `tasks/drive_screw.yaml` | actuate | screwdriver_tip | screw |
| Drop in toolchest | `places/toolchest_drop.yaml` | place | screwdriver | toolchest |

### 2. Everyday Manipulation — Mug of Water

Pick up a full mug, carry it to the sink, pour it out, set it on the table.

| Step | File | task | subject | reference |
|------|------|------|---------|-----------|
| Grasp mug by handle | `grasps/mug_handle_grasp.yaml` | grasp | gripper | mug |
| Transport upright | `tasks/mug_transport_upright.yaml` | transport | mug | world |
| Pour into sink | `tasks/mug_pour_into_sink.yaml` | pour | mug | sink |
| Place on table | `places/mug_on_table.yaml` | place | mug | table |

## Usage

```python
from tsr import load_package_template, load_package_templates_by_category
import numpy as np

# Load a single template: (category, file name)
t = load_package_template("grasps", "screwdriver_grasp.yaml")

# Bind to an object pose and sample
screwdriver_pose = np.eye(4)
screwdriver_pose[:3, 3] = [0.4, 0.0, 0.1]
tsr = t.instantiate(screwdriver_pose)
gripper_pose = tsr.sample()

# Load all grasp templates (the category is the directory name)
grasps = load_package_templates_by_category("grasps")
```

## Coordinate Frame Convention

All templates use a right-handed frame attached to the reference object:

- **x** — object's primary axis (handle axis for screwdriver, body axis for mug)
- **y** — object's secondary axis
- **z** — up / out-of-surface

End-effector frame convention (gripper):

- **z** — approach direction (toward object)
- **y** — finger opening direction
- **x** — palm normal (right-hand rule: x = y × z)
