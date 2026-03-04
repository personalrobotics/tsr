# What is a TSR?

A **Task Space Region** (TSR) constrains where one thing (the **subject**) can be, relative to another thing (the **reference**).

A grasp TSR says: "given where the mug (reference) is, where can the gripper (subject) go?"
A placement TSR says: "given where the table (reference) is, where can the mug (subject) go?"

Same math, different roles.

## The three components

A TSR is defined by three matrices:

| Component | What it is | Size |
|-----------|-----------|------|
| `T0_w` | **Reference pose** in world frame | 4×4 |
| `Tw_e` | **Canonical subject pose** in reference frame (at Bw = 0) | 4×4 |
| `Bw` | How much the subject can deviate from canonical | 6×2 |

`Bw` has 6 rows — one per degree of freedom: `[x, y, z, roll, pitch, yaw]`. Each row is `[min, max]`. Translations are in meters, rotations in radians.

### How they compose

To get a concrete subject pose, pick a sample `s` from within `Bw`, then:

```
subject_pose = T0_w  @  xyzrpy_to_transform(s)  @  Tw_e
               ────     ────────────────────────     ────
               reference   deviation from canonical     canonical
               pose        (within bounds)              subject pose
```

Sampling picks a random `s` within Bw and returns the resulting subject pose.

## Gripper frame convention

For **grasp** templates, `Tw_e` encodes where the gripper is relative to the object. This library uses a canonical gripper frame:

```
z = approach direction (toward object surface)
y = finger opening direction
x = palm normal  (right-hand rule: x = y × z)
```

**AnyGrasp / GraspNet** uses a different convention (`x` = approach). To convert AnyGrasp output to this library's format, apply R_y(90°):

```python
R_convert = np.array([[0, 0, -1],
                      [0, 1,  0],
                      [1, 0,  0]])
```

## Serialized template format

Templates are stored as YAML with raw TSR matrices plus semantic metadata.

**Required fields:** `name`, `description`, `task`, `subject`, `reference`, `T_ref_tsr`, `Tw_e`, `Bw`

**Optional fields:** `variant`, `preshape`, and any user-defined metadata.

```yaml
name: Mug Side Grasp
description: Side grasp avoiding handle
task: grasp
subject: gripper
reference: mug

T_ref_tsr:           # 4×4 reference-to-TSR-frame transform (usually identity)
- [1.0, 0.0, 0.0, 0.0]
- [0.0, 1.0, 0.0, 0.0]
- [0.0, 0.0, 1.0, 0.0]
- [0.0, 0.0, 0.0, 1.0]
Tw_e:                # 4×4 canonical subject pose in TSR frame (at Bw = 0)
- [0.0,  0.0, -1.0,  0.09]   # EE x-axis = cylinder axis; EE origin 9cm out
- [0.0,  1.0,  0.0,  0.0 ]   # EE y-axis = tangential (finger opening)
- [1.0,  0.0,  0.0,  0.0 ]   # EE z-axis = -radial (approach toward mug)
- [0.0,  0.0,  0.0,  1.0 ]
Bw:                  # (6,2) bounds over [x, y, z, roll, pitch, yaw]
- [0.0,  0.0 ]       # x: fixed (radius is in Tw_e)
- [0.0,  0.0 ]       # y: fixed
- [0.02, 0.08]       # z: height on mug body
- [0.0,  0.0 ]       # roll: fixed
- [0.0,  0.0 ]       # pitch: fixed
- [0.79, 5.50]       # yaw: 45°–315° (avoids handle at 0°)

# optional
preshape: [0.08]     # gripper aperture in meters
```

`T_ref_tsr` is usually identity (TSR frame = reference frame). It's non-identity when the canonical grasp point is offset from the object origin — for example, a mug handle grasp where the TSR frame is at the handle center, not the mug center.

`task`, `subject`, and `reference` are plain strings with no enum validation. Use any values meaningful to your application.

### Loading and saving

```python
from tsr.io import load_template, save_template, load_templates_from_directory

# Load a single template
template = load_template("templates/grasps/mug_side_grasp.yaml")

# Instantiate at a known object pose
tsr = template.instantiate(mug_pose_in_world)
grasp_pose = tsr.sample()

# Load all templates in a directory
templates = load_templates_from_directory("templates/grasps/")
```

## Generator pattern

Instead of hand-authoring matrices, you can write generator functions that compute `Tw_e` and `Bw` from object geometry.

The canonical pattern for a **cylinder side grasp** (matches the mug side grasp template above):

```python
import numpy as np
from tsr.template import TSRTemplate

def cylinder_side_grasp(
    radius: float,
    height_range: tuple,
    standoff: float = 0.05,
    subject: str = "gripper",
    reference: str = "object",
) -> TSRTemplate:
    """Side grasp for a cylinder. TSR frame = cylinder frame (z = cylinder axis).

    Tw_e columns (EE frame axes in TSR frame):
      col 0 (x̂_EE) = ẑ_TSR   cylinder axis = palm normal
      col 1 (ŷ_EE) = ŷ_TSR   tangential    = finger opening
      col 2 (ẑ_EE) = -x̂_TSR  radially inward = approach toward cylinder
    """
    h0, h1 = height_range
    z_mid  = (h0 + h1) / 2.
    z_half = (h1 - h0) / 2.

    T_ref_tsr = np.eye(4)
    T_ref_tsr[2, 3] = z_mid          # shift TSR frame to midpoint height

    Tw_e = np.array([
        [0., 0., -1., radius + standoff],
        [0., 1.,  0., 0.              ],
        [1., 0.,  0., 0.              ],
        [0., 0.,  0., 1.              ],
    ])

    Bw = np.array([
        [0.,      0.    ],  # x: fixed (radius in Tw_e)
        [0.,      0.    ],  # y: fixed
        [-z_half, z_half],  # z: height variation
        [0.,      0.    ],  # roll: fixed
        [0.,      0.    ],  # pitch: fixed
        [0.,  2*np.pi  ],  # yaw: full rotation
    ])

    return TSRTemplate(
        T_ref_tsr=T_ref_tsr, Tw_e=Tw_e, Bw=Bw,
        task="grasp", subject=subject, reference=reference,
        name=f"{reference.title()} Side Grasp",
        description=f"Side grasp for {reference} (r={radius*100:.0f}cm)",
    )
```

### Why radius goes into Tw_e

The composition formula:

```
subject_pose = T0_w  @  xyzrpy_to_transform(bw)  @  Tw_e
```

`xyzrpy_to_transform` builds `[R | t]` where `R` comes from the angular components and `t` from the translational components. `R` rotates `Tw_e`'s translation — but `t` is **not** rotated by `R`.

- A translation in **Tw_e** sweeps a circle/sphere as angles in Bw vary ✓
- A translation in **Bw** stays fixed regardless of angles ✗

So radius goes in `Tw_e`, not `Bw`.

```
Ring grasp (radius = 0.08, yaw varies):
  Tw_e translation = [0.08, 0, 0]     ← radius here

  yaw=0°   → R_z(0°)  @ [0.08,0,0] = [0.08,  0,    0]  ✓
  yaw=90°  → R_z(90°) @ [0.08,0,0] = [0,     0.08, 0]  ✓
  yaw=180° → R_z(180°)@ [0.08,0,0] = [-0.08, 0,    0]  ✓
```

For gripper-specific generation (e.g., Robotiq 2F-140 grasping various shapes), see `examples/parallel_jaw_grasp.py`.

## Example 1: Mug side grasp (avoiding handle)

**Reference** = mug (z = vertical axis). **Subject** = gripper.

"Given where the mug is, where can the gripper go to grasp it from the side, avoiding the handle?"

```python
template = load_template("templates/grasps/mug_side_grasp_avoid_handle.yaml")
```

**`T_ref_tsr`** = identity (TSR frame = mug frame)

**`Tw_e`** encodes the canonical gripper pose:

```
Tw_e = [0  0  -1   0.09]    EE x-axis = mug z-axis (up)
       [0  1   0   0   ]    EE y-axis = tangential (finger opening)
       [1  0   0   0   ]    EE z-axis = -radial (toward mug center = approach)
       [0  0   0   1   ]

EE origin: 9cm from mug axis (= mug_radius + standoff = 0.04 + 0.05)
```

**`Bw`** = where on the mug the gripper can be:

| DOF | Min | Max | Meaning |
|-----|-----|-----|---------|
| x | 0 | 0 | Fixed (radius is in Tw_e) |
| y | 0 | 0 | Fixed |
| z | 0.03 | 0.08 | Height along mug body |
| roll | 0 | 0 | Fixed |
| pitch | 0 | 0 | Fixed |
| yaw | 0.79 | 5.50 | 45°–315° — skips 0° where the handle is |

When sampling, yaw rotates the gripper around the mug's z-axis (sweeping the 9cm radial offset in a circle), and z slides it up/down the mug body.

## Example 2: Mug on table

**Reference** = table. **Subject** = mug.

"Given where the table is, where can the mug go?"

**`Tw_e`** = mug's canonical pose relative to table:

```
Tw_e = [1   0   0   0]    mug x → +x
       [0  -1   0   0]    mug y → -y  (flipped)
       [0   0  -1   0]    mug z → -z  (bottom faces down = upright)
       [0   0   0   1]
```

**`Bw`**:

| DOF | Min | Max | Meaning |
|-----|-----|-----|---------|
| x | -0.15 | 0.15 | 30cm placement area in x |
| y | -0.15 | 0.15 | 30cm placement area in y |
| z | 0 | 0 | On the surface |
| roll | 0 | 0 | Stay upright |
| pitch | 0 | 0 | Stay upright |
| yaw | -π | π | Any rotation around vertical |

## The key insight

A TSR always answers the same question:

> Given where the **reference** is (`T0_w`), where can the **subject** be?

The `Tw_e` matrix defines the ideal relationship. The `Bw` bounds define the allowed variation. The math is identical whether the subject is a gripper reaching for an object or an object being placed on a surface.

**Templates** store `Tw_e` and `Bw` without committing to `T0_w`. When you know the reference pose at runtime, you instantiate the template:

```python
tsr = template.instantiate(mug_pose_in_world)
grasp_pose = tsr.sample()
```

## Naming conventions

The original TSR paper uses `T0_w` (world-to-TSR), `Tw_e` (TSR-to-end-effector), and `Bw` (bounds in TSR frame). The "end-effector" language comes from the grasp case where the subject is always a gripper. In placement TSRs, the "end-effector" is actually the object being placed.

In this library:
- **reference** = the entity whose pose defines `T0_w` (mug for grasps, table for placements)
- **subject** = the entity whose pose is being constrained (gripper for grasps, mug for placements)
- `task`, `subject`, and `reference` are plain strings in the template format — no enum enforcement
