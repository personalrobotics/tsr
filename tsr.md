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

## Example 1: Mug side grasp (avoiding handle)

**Reference** = mug. **Subject** = gripper.

"Given where the mug is, where can the gripper go to grasp it from the side, avoiding the handle?"

### The YAML template

```yaml
name: Mug Side Grasp (Avoiding Handle)
task: grasp
subject: gripper
reference: mug

position:
  type: cylinder
  axis: z
  radius: 0.04          # mug radius (4cm)
  height: [0.03, 0.08]  # grasp zone on mug body
  angle: [45, 315]       # avoid handle at 0°

orientation:
  approach: radial       # fingers close perpendicular to mug surface

standoff: 0.05           # 5cm approach distance
```

### What this produces

**`T0_w`** = reference pose in world frame — the mug's pose. Set at runtime when you know where the mug is.

**`Tw_e`** = canonical subject pose in reference frame — the gripper's default pose relative to the mug. The parser combines `radial` orientation + `standoff` + `radius`:

- The gripper's z-axis points radially inward (toward the mug center)
- The gripper is offset by `radius + standoff = 0.04 + 0.05 = 0.09m` from the cylinder axis
- At Bw = 0, the gripper is at the starting position on the cylinder surface

```
Tw_e = [0  0  -1   0.09]    gripper z → -x (points toward center)
       [0  1   0   0   ]    gripper y → +y (finger opening direction)
       [1  0   0   0   ]    gripper x → +z
       [0  0   0   1   ]
```

The translation `[0.09, 0, 0]` is the standoff expressed in the TSR frame. It places the gripper 9cm out along the TSR x-axis — and when yaw rotates, this offset rotates with it, keeping the gripper on the cylinder surface.

**`Bw`** = where on the cylinder the gripper is allowed:

| DOF | Min | Max | Meaning |
|-----|-----|-----|---------|
| x | 0 | 0 | Fixed (radius is in Tw_e, not here) |
| y | 0 | 0 | Fixed |
| z | 0.03 | 0.08 | Height along mug body |
| roll | 0 | 0 | Fixed |
| pitch | 0 | 0 | Fixed |
| yaw | 45° (0.79 rad) | 315° (5.50 rad) | Angle around mug — skips 0° where the handle is |

When you sample, the yaw rotates the gripper around the mug's z-axis, and the z translation slides it up/down the mug body. The 0.09m offset in Tw_e rotates with the yaw — this is why radius goes into Tw_e, not Bw.

## Example 2: Mug on table

**Reference** = table. **Subject** = mug.

"Given where the table is, where can the mug go?"

### The YAML template

```yaml
name: Mug on Table
task: place
subject: mug
reference: table
reference_frame: bottom   # expects pose of mug's bottom surface

position:
  type: plane
  x: [-0.15, 0.15]       # generous placement area
  y: [-0.15, 0.15]
  z: 0                   # table surface

orientation:
  approach: -z            # mug upright (z-up)
  yaw: free               # any rotation around vertical

standoff: 0
```

### What this produces

**`T0_w`** = reference pose in world frame — the table's surface pose. Set at runtime.

**`Tw_e`** = canonical subject pose in reference frame — the mug's default pose relative to the table. With `approach: -z` and zero standoff:

- `approach: -z` means the subject's contact direction faces -z (downward)
- For a mug, that means its bottom faces down — it's upright
- This is a 180° rotation about the x-axis relative to the TSR frame

```
Tw_e = [1   0   0   0]    mug x → +x
       [0  -1   0   0]    mug y → -y  (flipped)
       [0   0  -1   0]    mug z → -z  (bottom faces down = upright)
       [0   0   0   1]
```

**`Bw`** = where on the table the mug is allowed:

| DOF | Min | Max | Meaning |
|-----|-----|-----|---------|
| x | -0.15 | 0.15 | 30cm placement area in x |
| y | -0.15 | 0.15 | 30cm placement area in y |
| z | 0 | 0 | On the surface (no floating) |
| roll | 0 | 0 | Stay upright |
| pitch | 0 | 0 | Stay upright |
| yaw | -π | π | Any rotation around vertical |

When you sample, x and y slide the mug around the table surface, and yaw rotates it. The mug stays upright because roll and pitch are locked to zero.

Note: `reference_frame: bottom` tells the caller that this template's `T0_w` expects the mug's bottom-surface pose, not its center of mass. The `reference_frame` is a convenience — it's often much easier to define canonical poses and bounds relative to a specific point on the object (like the bottom of a mug, or the center of a pitcher handle) rather than the object origin. The math doesn't change; it just shifts which pose the caller passes as `T0_w`.

## The key insight

A TSR always answers the same question:

> Given where the **reference** is (`T0_w`), where can the **subject** be?

The `Tw_e` matrix defines the ideal relationship. The `Bw` bounds define the allowed variation. The math is identical whether the subject is a gripper reaching for an object or an object being placed on a surface.

**Templates** store `Tw_e` and `Bw` without committing to `T0_w`. When you know the reference pose at runtime, you instantiate the template to get a concrete TSR:

```python
tsr = TSR(T0_w=mug_pose, Tw_e=template.Tw_e, Bw=template.Bw)
grasp_pose = tsr.sample()
```

## Position primitives

The `position.type` field in a template selects a geometric primitive — the shape of the region where the subject can be. Each primitive maps onto `Bw` and `Tw_e` differently.

### The nine primitives

| Primitive | Shape | Bw DOFs used | Example |
|-----------|-------|-------------|---------|
| **point** | Single point | None (all zero) | Mug handle grasp — one specific pose |
| **line** | Line segment | 1 translation | Edge grasp along a shelf lip |
| **plane** | Rectangle | 2 translations | Placement area on a table |
| **box** | Box volume | 3 translations | Workspace bounds |
| **ring** | Circle | 1 angle + height | Bowl rim grasp |
| **disk** | Annular region | 1 angle + radial band + height | Cup on coaster |
| **cylinder** | Cylindrical surface | 1 angle + height range | Mug side grasp |
| **shell** | Cylindrical band | 1 angle + height range + radial band | Grasping a pipe (varying wall thickness) |
| **sphere** | Spherical surface | pitch + yaw | Handover — present object in front of person |

### Why radius goes into Tw_e

The composition formula is:

```
subject_pose = T0_w  @  xyzrpy_to_transform(bw)  @  Tw_e
```

The `xyzrpy_to_transform` function builds a matrix `[R | t]` where `R` comes from the angular components and `t` from the translational components. Crucially, `R` rotates `Tw_e`'s translation — but `t` itself is **not** rotated by `R`.

This means:
- A translation in **Tw_e** rotates when Bw's angles change (it sweeps a circle/sphere)
- A translation in **Bw** stays fixed regardless of Bw's angles (it just offsets)

For ring, disk, cylinder, shell, and sphere, the radius must trace a circle as the angle varies. So radius goes into `Tw_e`, where it will be rotated by the angular DOFs in `Bw`.

```
Ring (radius=0.08, yaw varies):
  Tw_e translation = [0.08, 0, 0]     ← radius in Tw_e
  Bw yaw = [0°, 360°]

  yaw=0°   → position = R(0°)   @ [0.08,0,0] = [0.08, 0,     0]  ✓
  yaw=90°  → position = R(90°)  @ [0.08,0,0] = [0,    0.08,  0]  ✓
  yaw=180° → position = R(180°) @ [0.08,0,0] = [-0.08, 0,    0]  ✓
```

If radius were in Bw's translation instead, it would not rotate — every yaw angle would offset by the same `[0.08, 0, 0]`, producing wrong positions.

### How each primitive distributes parameters

**Purely translational** (no rotation interaction — safe in Bw):
- **point**: Fixed offset goes into `T_ref_tsr` (the reference-to-TSR-origin transform). Bw is all zeros.
- **line**, **plane**, **box**: Linear ranges go directly into Bw translations.

**Radial** (radius must rotate with angle — goes into Tw_e):
- **ring**: Radius → `Tw_e` translation. Angle → Bw rotation. Height → Bw translation along axis.
- **disk**: Radius midpoint → `Tw_e` translation. Radius half-thickness → Bw translation (radial tolerance). Angle + height → Bw.
- **cylinder**: Radius + standoff → `Tw_e` translation. Angle + height range → Bw.
- **shell**: Radius midpoint + standoff → `Tw_e` translation. Radius half-thickness → Bw. Angle + height range → Bw.
- **sphere**: Radius → `Tw_e` translation. Pitch + yaw → Bw rotations.

### Standoff and approach direction

The `standoff` adds distance along the **approach direction** (the direction the subject faces when contacting the reference). For a radial approach on a cylinder, standoff adds to the radius in `Tw_e`. For a top-down approach (`-z`) on a disk, standoff is along z while radius is along x — they're independent components in `Tw_e`.

The `orientation.approach` field sets the rotation part of `Tw_e`, determining which direction the subject "faces." Common values:
- `radial` — outward from the primitive's axis (for grasping cylinders, rings)
- `-z` — downward (for placements, top grasps)
- `-x` — toward the reference (for handovers)

## Naming conventions

The original TSR paper uses `T0_w` (world-to-TSR), `Tw_e` (TSR-to-end-effector), and `Bw` (bounds in TSR frame). The "end-effector" language comes from the grasp case where the subject is always a gripper. In placement TSRs, the "end-effector" is actually the object being placed.

In this library:
- **reference** = the entity whose pose defines `T0_w` (mug for grasps, table for placements)
- **subject** = the entity whose pose is being constrained (gripper for grasps, mug for placements)
- The YAML templates use `subject:` and `reference:` explicitly
