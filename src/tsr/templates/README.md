# TSR Templates

This directory contains TSR template YAML files organized by task category.

## Directory Structure

```
templates/
├── grasps/          # Grasping templates
├── places/          # Placement templates
├── tools/           # Tool manipulation templates
└── README.md        # This file
```

## Template Organization

### Grasps (`grasps/`)
Templates for grasping different objects:
- `mug_side_grasp.yaml` - Side grasp for cylindrical objects
- `mug_top_grasp.yaml` - Top grasp for open containers
- `box_side_grasp.yaml` - Side grasp for rectangular objects

### Places (`places/`)
Templates for placing objects:
- `mug_on_table.yaml` - Place mug on flat surface
- `bottle_in_shelf.yaml` - Place bottle in shelf compartment

### Tools (`tools/`)
Templates for tool manipulation:
- `screwdriver_grasp.yaml` - Grasp screwdriver handle
- `wrench_grasp.yaml` - Grasp wrench handle

## Usage

```python
from tsr import TemplateIO

# Load a specific template
template = TemplateIO.load_template("templates/grasps/mug_side_grasp.yaml")

# Load all templates from a category
grasp_templates = TemplateIO.load_templates_from_directory("templates/grasps/")

# Load templates by category
templates_by_category = TemplateIO.load_templates_by_category("templates/")
```

## Template Format

Each template YAML file contains:
- **Semantic context**: subject, reference, task category, variant
- **Geometric parameters**: T_ref_tsr, Tw_e, Bw matrices
- **Metadata**: name, description
- **Optional preshape**: gripper configuration as DOF values

Example:
```yaml
name: Mug Side Grasp
description: Grasp mug from the side with 5cm approach distance
subject_entity: generic_gripper
reference_entity: mug
task_category: grasp
variant: side
T_ref_tsr: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
Tw_e: [[0, 0, 1, -0.05], [1, 0, 0, 0], [0, 1, 0, 0.05], [0, 0, 0, 1]]
Bw: [[0, 0], [0, 0], [-0.01, 0.01], [0, 0], [0, 0], [-3.14159, 3.14159]]
preshape: [0.08]  # Optional: 8cm aperture for parallel jaw gripper
```

## Preshape Configuration

Templates can include optional `preshape` fields to specify gripper configurations:

### Parallel Jaw Grippers
```yaml
preshape: [0.08]  # Single value: aperture in meters
```

### Multi-Finger Hands
```yaml
preshape: [0.0, 0.5, 0.5, 0.0, 0.5, 0.5]  # Multiple values: joint angles
```

### No Preshape
Omit the `preshape` field or set to `null` for templates that don't require specific gripper configuration.

## Contributing

When adding new templates:
1. Use descriptive filenames
2. Include comprehensive descriptions
3. Add preshape configuration when gripper state is important
4. Test the template with the library
5. Update this README if adding new categories
