# Task Space Regions (TSR) Tutorial

A comprehensive guide to understanding and using Task Space Regions for robotic manipulation.

## Table of Contents

1. [Introduction](#introduction)
2. [What is a TSR?](#what-is-a-tsr)
3. [Mathematical Formulation](#mathematical-formulation)
4. [The Bounds Matrix](#the-bounds-matrix)
5. [TSR Chains](#tsr-chains)
6. [TSR Templates](#tsr-templates)
7. [Generating Grasp Templates](#generating-grasp-templates)
8. [Visualization](#visualization)
9. [Practical Examples](#practical-examples)

---

## Introduction

When programming robots to manipulate objects, we often need to specify **where** and **how** the robot should approach, grasp, or place something. The naive approach — specifying a single exact pose — is brittle:

- Manufacturing tolerances mean objects are never exactly where we expect
- A mug can be placed anywhere on a table, not just one specific spot
- A drawer handle can be grasped at any point along its length
- A pouring motion must keep the end-effector in a valid tilt range *throughout* the trajectory

**Task Space Regions (TSRs)** solve this by specifying a *region* of valid poses rather than a single pose. A TSR can express:

- **Grasp constraints**: all valid end-effector poses that achieve force closure on an object
- **Placement constraints**: all valid poses where an object can be set down
- **Trajectory-wide constraints**: a region the end-effector must stay inside for every waypoint along a motion (e.g., keep bottle tilted while pouring, keep tray level while carrying)

Because TSRs are continuous regions, motion planners can sample from them, check containment, and compute distance-to-valid — all without enumerating discrete poses.

---

## What is a TSR?

A **Task Space Region** defines a continuous set of valid end-effector poses relative to some reference frame. Instead of "grasp the mug at exactly this pose," a TSR says "grasp the mug anywhere around its cylindrical body, at any height, approaching radially."

TSRs are useful for:

- **Grasping**: valid pre-grasp configurations for a given object geometry
- **Placement**: where objects can be set down (on a table, in a bin, on a shelf)
- **Motion constraints**: keeping the end-effector in a valid region during manipulation (pouring, carrying, wiping)
- **Articulated objects**: constraints that move with doors, drawers, rotating handles

---

## Mathematical Formulation

A TSR is defined by three components:

### 1. World-to-TSR Transform: $T_0^w$

This 4×4 homogeneous transformation matrix positions the TSR frame in the world. It represents where the constraint region is located.

$$T_0^w = \begin{bmatrix} R_{3\times3} & t_{3\times1} \\ 0_{1\times3} & 1 \end{bmatrix}$$

**Example**: For grasping a mug, $T_0^w$ is the mug's pose. For placing on a table, it is the table surface's pose.

### 2. TSR-to-End-Effector Transform: $T_w^e$

This 4×4 matrix defines the nominal (center) pose of the end-effector relative to the TSR frame when all Bw displacements are zero.

**Example**: For a side grasp on a cylinder, $T_w^e$ encodes the gripper's standoff distance and approach orientation.

### 3. Bounds Matrix: $B_w$

A 6×2 matrix defining allowed deviations from the nominal pose:

$$B_w = \begin{bmatrix}
x_{min} & x_{max} \\
y_{min} & y_{max} \\
z_{min} & z_{max} \\
roll_{min} & roll_{max} \\
pitch_{min} & pitch_{max} \\
yaw_{min} & yaw_{max}
\end{bmatrix}$$

### Sampling from a TSR

1. Sample a displacement $\Delta = (\Delta x, \Delta y, \Delta z, \Delta roll, \Delta pitch, \Delta yaw)$ uniformly from $B_w$
2. Convert $\Delta$ to a 4×4 matrix $T_\Delta$
3. Compute the end-effector pose: $T_{ee} = T_0^w \cdot T_\Delta \cdot T_w^e$

### Checking if a Pose is Valid

1. Compute $T_{rel} = (T_0^w)^{-1} \cdot T_{ee} \cdot (T_w^e)^{-1}$
2. Extract displacement $\Delta$ from $T_{rel}$
3. Check each component against $B_w$

```python
from tsr import TSR
import numpy as np

tsr = TSR(T0_w=object_pose, Tw_e=gripper_offset, Bw=Bw)

pose        = tsr.sample()            # random SE(3) pose in the region
is_valid    = tsr.contains(pose)      # True/False containment check
distance, _ = tsr.distance(pose)      # distance to nearest valid pose (0 if inside)
```

---

## The Bounds Matrix

The 6×2 bounds matrix $B_w$ is the heart of a TSR. Each row constrains one degree of freedom:

| Row | DOF | Units | Description |
|-----|-----|-------|-------------|
| 0 | X | meters | Translation along local X axis |
| 1 | Y | meters | Translation along local Y axis |
| 2 | Z | meters | Translation along local Z axis |
| 3 | Roll | radians | Rotation about local X axis |
| 4 | Pitch | radians | Rotation about local Y axis |
| 5 | Yaw | radians | Rotation about local Z axis |

### Special Values

- **Fixed DOF**: set min = max (e.g., `[0, 0]` pins the axis)
- **Free rotation**: `[-π, π]` for full 360° freedom
- **Symmetric range**: `[-r, r]` centers the range at zero

### Common $B_w$ Patterns

**Object on a table** (placement):
```python
Bw = np.array([
    [-0.15, 0.15],    # x: 30 cm range on surface
    [-0.15, 0.15],    # y: 30 cm range on surface
    [0.0,   0.0 ],    # z: exactly on surface
    [0.0,   0.0 ],    # roll: upright
    [0.0,   0.0 ],    # pitch: upright
    [-np.pi, np.pi],  # yaw: any orientation
])
```

**Side grasp on a cylinder** (any angle around, limited height band):
```python
Bw = np.array([
    [0.0,  0.0 ],         # x: no radial freedom (baked into Tw_e)
    [0.0,  0.0 ],         # y: no tangential freedom
    [-0.03, 0.03],        # z: 6 cm height band
    [0.0,  0.0 ],         # roll: fixed by Tw_e
    [0.0,  0.0 ],         # pitch: fixed
    [-np.pi, np.pi],      # yaw: approach from any angle
])
```

**Pouring constraint** (trajectory-wide):
```python
Bw = np.array([
    [-0.02, 0.02],          # x: small tolerance over glass
    [-0.02, 0.02],          # y: small tolerance over glass
    [0.15,  0.25],          # z: 10 cm height band above glass
    [-np.pi/4, -np.pi/6],   # roll: tilted 30–45° for pouring
    [0.0,   0.0 ],          # pitch
    [-np.pi, np.pi],        # yaw: any rotation around glass axis
])
```

---

## TSR Chains

A **TSR Chain** connects multiple TSRs in sequence, where each TSR's frame is relative to the previous one. This is essential for **articulated objects** and **multi-constraint trajectories**.

### Example: Door Handle Grasp

The handle's reachable pose depends on the door's current angle:

```
World → Door Frame (hinge) → Handle Frame (on door) → Grasp Frame
```

```python
from tsr import TSR, TSRChain

# TSR 1: door rotation about hinge
door_tsr = TSR(
    T0_w=door_hinge_pose,
    Tw_e=np.eye(4),
    Bw=np.array([
        [0, 0], [0, 0], [0, 0],          # no translation
        [0, 0], [0, 0], [-np.pi/2, 0],   # door angle: closed to 90° open
    ])
)

# TSR 2: handle fixed on door
handle_tsr = TSR(
    T0_w=np.eye(4),         # relative to previous TSR's frame
    Tw_e=handle_offset,
    Bw=np.zeros((6, 2))
)

# TSR 3: grasp tolerance on handle
grasp_tsr = TSR(
    T0_w=np.eye(4),
    Tw_e=np.eye(4),
    Bw=np.array([
        [-0.02, 0.02], [0, 0], [0, 0],    # small position tolerance
        [0, 0], [0, 0], [-np.pi, np.pi],  # any grasp rotation
    ])
)

chain = TSRChain(TSRs=[door_tsr, handle_tsr, grasp_tsr])
pose  = chain.sample()
```

### How Chains Work

1. Sample/evaluate the first TSR to get $T_1$
2. Use $T_1$ as the world frame for TSR 2, producing $T_2$
3. Continue until the final pose $T_n$

The result is valid for any door angle within the allowed range — the planner doesn't need to enumerate door positions.

---

## TSR Templates

A **TSRTemplate** is a serializable TSR that is not yet bound to a specific scene object. It stores the constraint relative to a reference object's local frame. At runtime you bind it by supplying the reference object's world pose.

```python
from tsr import TSRTemplate, save_template, load_template
import numpy as np

# Build a template (reference object frame as origin)
template = TSRTemplate(
    T_ref_tsr=np.eye(4),    # TSR frame relative to object
    Tw_e=Tw_e,
    Bw=Bw,
    task="place",
    subject="mug",
    reference="table",
    name="Mug on Table",
    description="Place mug upright anywhere on table surface",
)

# Save to disk
save_template(template, "templates/mug_on_table.yaml")

# Load later
template = load_template("templates/mug_on_table.yaml")

# Bind to the actual table pose at runtime → TSR
table_pose = perception.get_table_pose()
tsr  = template.instantiate(table_pose)
pose = tsr.sample()
```

Templates separate **constraint design** (done offline, per object type) from **constraint binding** (done at runtime, per scene instance). A library of templates can be reused across any scene where those objects appear.

---

## Generating Grasp Templates

For common geometries, grasp templates can be generated programmatically from object dimensions rather than hand-authored. The `ParallelJawGripper` class in `examples/parallel_jaw_grasp.py` demonstrates the pattern.

### Gripper Frame Convention

All templates in this library use a canonical EE frame:

```
z = approach direction  (toward object surface)
y = finger opening direction
x = palm normal         (right-hand rule: x = y × z)
```

AnyGrasp / GraspNet uses `x = approach` — convert with:
```python
R_convert = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
```

### Side Grasp on a Cylinder

The fundamental challenge for cylinder side grasps: **the radial approach direction couples with yaw** in $B_w$ and cannot be encoded as a free DOF directly. Instead, `grasp_cylinder` generates `2*k` templates — `k` discrete approach depths × 2 roll orientations — each with the radial standoff baked into $T_w^e$.

```python
from examples.parallel_jaw_grasp import ParallelJawGripper

# Robotiq 2F-140: 140mm max aperture, 55mm finger length
gripper = ParallelJawGripper(finger_length=0.055, max_aperture=0.140)

# Returns 6 TSRTemplates: 3 depth levels × 2 roll orientations
templates = gripper.grasp_cylinder(
    object_radius=0.040,          # 4 cm mug radius
    height_range=(0.02, 0.10),    # graspable height band on mug
    reference="mug",
)

for t in templates:
    print(t.name)
    # "Mug Cylinder Side Grasp — shallow, roll 0°"
    # "Mug Cylinder Side Grasp — shallow, roll 180°"
    # "Mug Cylinder Side Grasp — mid, roll 0°"
    # ...

# Bind to detected mug pose and sample
mug_pose = perception.get_object_pose("mug")
grasp_poses = [t.instantiate(mug_pose).sample() for t in templates]
```

**What `grasp_cylinder` produces:**

| Parameter | Effect |
|-----------|--------|
| `k` (default 3) | Number of discrete approach depths: shallow (fingertips near surface) → deep (palm near surface) |
| `clearance` (default 10% of `finger_length`) | Safety buffer at height ends and both depth limits |
| `preshape` (default `2 * r + clearance`) | Jaw opening — minimum viable to span the cylinder |
| `angle_range` (default `[0, 2π]`) | Yaw freedom (restrict to front hemisphere for wall-mounted objects) |

The two roll orientations (0° and 180° around the approach axis) are needed for asymmetric hands. For a symmetric hand they produce identical contact geometry but different palm normals.

---

## Visualization

The `tsr.viz` module provides a PyVista-based renderer. Unlike matplotlib's painter's algorithm, PyVista uses a true depth buffer — geometry behind the object is automatically occluded.

```python
from tsr.viz import TSRVisualizer, cylinder_renderer, parallel_jaw_renderer, plasma_colors

mug_pose = np.eye(4)
poses  = [t.instantiate(mug_pose).sample() for t in templates]

# One color per TSR template (groups visually)
colors = []
n_per  = 3
tsr_colors = plasma_colors(len(templates))
for i, t in enumerate(templates):
    tsr = t.instantiate(mug_pose)
    poses.extend([tsr.sample() for _ in range(n_per)])
    colors.extend([tsr_colors[i]] * n_per)

TSRVisualizer(
    title=f"{len(templates)} templates · {len(poses)} sampled gripper poses",
    focus=(0., 0., 0.06),      # look at the mug's mid-height
    camera_az=215.,             # azimuth (degrees)
    camera_el=25.,              # elevation (degrees)
).render(
    reference_renderer=cylinder_renderer(radius=0.04, height=0.12),
    subject_renderer=parallel_jaw_renderer(finger_length=0.055, half_aperture=0.07),
    poses=poses,
    colors=colors,
    out="assets/tsr_viz.png",
)
```

Requires the `viz` extra: `uv sync --extra viz`.

**Custom renderers** follow these signatures:

```python
# reference_renderer: draws the scene object
def my_object_renderer(pl: pv.Plotter) -> None: ...

# subject_renderer: draws one end-effector at a given pose
def my_ee_renderer(pl: pv.Plotter, pose_4x4: np.ndarray, color: tuple) -> None: ...
```

---

## Practical Examples

### Example 1: Grasp a Mug with a Parallel Jaw Gripper

```python
from examples.parallel_jaw_grasp import ParallelJawGripper
import numpy as np

gripper   = ParallelJawGripper(finger_length=0.055, max_aperture=0.140)
templates = gripper.grasp_cylinder(object_radius=0.040, height_range=(0.02, 0.10), reference="mug")

mug_pose  = perception.get_object_pose("mug")
candidates = [t.instantiate(mug_pose).sample() for t in templates]
```

### Example 2: Place Mug on Table (Placement Constraint)

```python
from tsr import TSR
import numpy as np

table_pose = perception.get_table_pose()

place_tsr = TSR(
    T0_w=table_pose,
    Tw_e=np.eye(4),
    Bw=np.array([
        [-0.20, 0.20],    # x: 40 cm range on table
        [-0.15, 0.15],    # y: 30 cm range on table
        [0.0,   0.0 ],    # z: exactly on surface
        [0.0,   0.0 ],    # roll: upright
        [0.0,   0.0 ],    # pitch: upright
        [-np.pi, np.pi],  # yaw: any orientation
    ])
)

placement = place_tsr.sample()
distance, _ = place_tsr.distance(proposed_pose)   # 0 if valid
```

### Example 3: Pour from Bottle (Trajectory-Wide Constraint)

A TSR used as a trajectory constraint keeps every waypoint in the valid region:

```python
from tsr import TSR

glass_pose = perception.get_object_pose("glass")

pour_tsr = TSR(
    T0_w=glass_pose,
    Tw_e=np.eye(4),
    Bw=np.array([
        [-0.02, 0.02],          # x: over glass
        [-0.02, 0.02],          # y: over glass
        [0.15,  0.25],          # z: 10 cm band above rim
        [-np.pi/4, -np.pi/6],   # roll: 30–45° tilt for pouring
        [0.0,   0.0 ],          # pitch
        [-np.pi, np.pi],        # yaw: any rotation around glass axis
    ])
)

# Pass to planner — every waypoint must satisfy pour_tsr
plan = cbirrt.plan(start=q_start, goal=q_goal, constraints=[pour_tsr])
```

### Example 4: Place in a Bin (Flexible Placement)

Objects can be dropped anywhere inside a bin's opening:

```python
from tsr import TSRTemplate, save_template
import numpy as np

# Template authored relative to the bin's opening frame
Bw = np.array([
    [-0.10, 0.10],    # x: 20 cm bin width
    [-0.07, 0.07],    # y: 14 cm bin depth
    [0.0,   0.0 ],    # z: at opening plane
    [0.0,   0.0 ],    # roll: upright
    [0.0,   0.0 ],    # pitch
    [-np.pi, np.pi],  # yaw: any
])

template = TSRTemplate(
    T_ref_tsr=np.eye(4),
    Tw_e=np.eye(4),
    Bw=Bw,
    task="place",
    subject="can",
    reference="bin",
    name="Can in Bin",
)
save_template(template, "templates/can_in_bin.yaml")

# At runtime
bin_pose = perception.get_object_pose("bin")
tsr  = template.instantiate(bin_pose)
pose = tsr.sample()
```

---

## Summary

| Concept | Purpose |
|---------|---------|
| **$T_0^w$** | Positions the constraint in the world (object/surface pose) |
| **$T_w^e$** | Nominal end-effector offset from TSR origin |
| **$B_w$** | Allowed deviations — the actual constraint region |
| **TSR Chains** | Constraints on articulated objects (doors, drawers) |
| **TSRTemplate** | Serializable, reusable constraint bound at runtime |
| **Grasp generators** | Programmatic templates from object geometry (e.g., `grasp_cylinder`) |
| **TSRVisualizer** | 3D visualization with correct occlusion via PyVista |

TSRs are a single abstraction that covers grasping, placement, and trajectory constraints — the difference lies only in what $T_0^w$ refers to and how $B_w$ shapes the valid region.

For more details, see the [examples](../examples/) directory.

---

## References

1. Berenson, D., Srinivasa, S., Ferguson, D., & Kuffner, J. (2009). *Manipulation planning on constraint manifolds*. IEEE International Conference on Robotics and Automation.

2. Berenson, D., Srinivasa, S., & Kuffner, J. (2011). *Task Space Regions: A framework for pose-constrained manipulation planning*. International Journal of Robotics Research.
