# TSR Template Library

Hand-authored TSR templates for two manipulation narratives. Each template is a
YAML file loadable via `load_template()` / `load_package_template()`.

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

# Load a single template
t = load_package_template("grasps/screwdriver_grasp.yaml")

# Bind to an object pose and sample
screwdriver_pose = np.eye(4)
screwdriver_pose[:3, 3] = [0.4, 0.0, 0.1]
tsr = t.instantiate(screwdriver_pose)
gripper_pose = tsr.sample()

# Load all grasp templates
grasps = load_package_templates_by_category("grasp")
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
