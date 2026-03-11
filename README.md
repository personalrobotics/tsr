# Task Space Regions (TSR)

A Python library for pose-constrained manipulation planning using Task Space Regions.

Based on the IJRR paper ["Task Space Regions: A Framework for Pose-Constrained Manipulation Planning"](https://www.ri.cmu.edu/pub_files/2011/10/dmitry_ijrr10-1.pdf) by Berenson, Srinivasa, and Kuffner.

![TSR Cylinder Side Grasp](assets/tsr_viz.png)

## Installation

```bash
uv add git+https://github.com/personalrobotics/tsr.git
```

For visualization support:
```bash
uv add "tsr[viz] @ git+https://github.com/personalrobotics/tsr.git"
```

For development:
```bash
git clone https://github.com/personalrobotics/tsr.git
cd tsr
uv sync --extra test
```

## Quick Start

### Generate grasp templates from object geometry

```python
import numpy as np
from tsr.hands import ParallelJawGripper

gripper = ParallelJawGripper(finger_length=0.055, max_aperture=0.140)

# Cylinder — side + top + bottom: 4*k templates (default k=3: 12 total)
templates = gripper.grasp_cylinder(
    cylinder_radius=0.040,   # 4 cm radius
    cylinder_height=0.120,   # 12 cm tall
    reference="mug",
)
for t in templates:
    print(t.name)
    # "Mug Cylinder Side Grasp — shallow, roll 0°"
    # "Mug Cylinder Side Grasp — shallow, roll 180°"
    # ...

# Box — all six faces, two finger orientations per face: up to 2*6*k templates
templates = gripper.grasp_box(box_x=0.08, box_y=0.06, box_z=0.18, reference="box")

# Sphere — full SO(3) approach: k templates
templates = gripper.grasp_sphere(object_radius=0.040, reference="ball")

# Torus — side (all minor angles) + span (if aperture allows): up to 2*k*n_minor + 2*k
templates = gripper.grasp_torus(
    torus_radius=0.035,   # major radius R: center to tube center
    tube_radius=0.015,    # minor radius r: tube cross-section
    reference="handle",
)

# Instantiate at a specific object pose and sample
mug_pose = np.eye(4)
mug_pose[:3, 3] = [0.5, 0.0, 0.0]   # mug at x=0.5m

grasp_poses = [t.instantiate(mug_pose).sample() for t in templates]
```

### Torus grasps in detail

The torus primitive covers handles, rings, and rotary knobs with two complementary modes:

**Side grasps** (`grasp_torus_side`) — approach from `n_minor` discrete angles α around
the tube cross-section, with full azimuthal yaw freedom around the ring:

```
α = −π/2  from below      α = −π/4  from below-outside
α =  0    from outside     α = +π/4  from above-outside
α = +π/2  from above
```

Depth range: fingertips sweep from the tube centerline (shallowest) to the
inner tube surface (deepest), or stop when the palm reaches the outer surface.
Returns `[]` if `finger_length ≤ tube_radius` (finger too short to reach centerline).

**Span grasps** (`grasp_torus_span`) — approach from above/below with fingers spanning
the full outer diameter `2*(R+r)`. Only generated when `2*(R+r) + clearance ≤ max_aperture`.

```python
# Side grasps: 5 angles × 3 depths × 2 flips = 30 templates
side = gripper.grasp_torus_side(torus_radius=0.035, tube_radius=0.015, n_minor=5, k=3)

# Span grasps: 3 top + 3 bottom = 6 templates (silently [] if torus too wide)
span = gripper.grasp_torus_span(torus_radius=0.035, tube_radius=0.015, k=3)

# Combined (recommended): side + span
all_templates = gripper.grasp_torus(torus_radius=0.035, tube_radius=0.015)
```

### Work directly with TSRs

```python
from tsr import TSR, TSRTemplate, TSRChain
import numpy as np

# A TSR is defined by three components:
#   T0_w : 4×4 transform — world frame to TSR frame
#   Tw_e : 4×4 transform — TSR frame to end-effector at Bw=0
#   Bw   : 6×2 bounds — [x, y, z, roll, pitch, yaw]

T0_w = np.eye(4)
Tw_e = np.eye(4)
Bw   = np.zeros((6, 2))
Bw[2, :] = [0.0,  0.10]       # z: 0–10 cm
Bw[5, :] = [-np.pi, np.pi]    # yaw: full rotation

tsr = TSR(T0_w=T0_w, Tw_e=Tw_e, Bw=Bw)

pose     = tsr.sample()                      # random SE(3) pose in the region
distance, _ = tsr.distance(pose)            # distance to nearest valid pose
is_valid = tsr.contains(pose)               # containment check
```

### Save and load templates

```python
from tsr import TSRTemplate, save_template, load_template

template = TSRTemplate(
    T_ref_tsr=np.eye(4),
    Tw_e=Tw_e,
    Bw=Bw,
    task="grasp",
    subject="gripper",
    reference="mug",
    name="Mug Side Grasp",
    description="Side grasp on a cylindrical mug body",
)

save_template(template, "my_grasp.yaml")
template = load_template("my_grasp.yaml")

# Bind to an object pose at runtime
tsr = template.instantiate(mug_pose)
```

## Gripper frame convention

All TSR templates in this library use a canonical gripper frame:

```
z = approach direction  (toward object surface)
y = finger opening direction
x = palm normal         (right-hand rule: x = y × z)
```

AnyGrasp / GraspNet uses `x = approach` — convert with:
```python
R_convert = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
```

## TSR Chains

For coupled constraints (e.g., door opening, constrained transport):

```python
from tsr import TSRChain

chain = TSRChain(TSRs=[hinge_tsr, handle_tsr])
pose  = chain.sample()
```

## Visualization

```python
from tsr.viz import TSRVisualizer, cylinder_renderer, parallel_jaw_renderer, plasma_colors

poses  = [t.instantiate(mug_pose).sample() for t in templates]
colors = plasma_colors(len(templates))

TSRVisualizer(
    title="Cylinder Side Grasp",
    focus=(0., 0., 0.06),
).render(
    reference_renderer=cylinder_renderer(radius=0.04, height=0.12),
    subject_renderer=parallel_jaw_renderer(finger_length=0.055, half_aperture=0.07),
    poses=poses,
    colors=colors,
    out="grasp_viz.png",
)
```

Requires the `viz` extra: `uv sync --extra viz`.

## Documentation

- **[Tutorial](docs/tutorial.md)** — TSR theory, math, and worked examples
- **[Examples](examples/)** — Runnable scripts (`uv run python examples/<script>.py`)

## Testing

```bash
uv run pytest tests/ -v
```

## License

BSD-2-Clause — see LICENSE file.
