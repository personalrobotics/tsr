# Architecture

`tsr` is organised in four layers. Lower layers never import upper ones; the
boundary is enforced by `tests/tsr/test_architecture.py`.

| Layer | Module(s) | Role | Dependencies |
|-------|-----------|------|--------------|
| **0 — core** | `tsr.core` (`TSR`, `TSRChain`, math helpers) | The pure-math heart: SE(3) geometry, bounds, sampling, distance, serialisation. Robot-agnostic. | NumPy (+ SciPy for the optimiser paths) |
| **1 — recipe** | `tsr.template` (`TSRTemplate`) | A reusable, serialisable, scene-agnostic recipe that instantiates to a concrete `TSR` at a reference pose. | Layer 0 |
| **2 — factories** | `tsr.hands` (`ParallelJawGripper`, …), `tsr.placement` (`StablePlacer`) | Domain knowledge that *produces* recipes from a described object. | Layers 0–1 |
| **3 — services** | `tsr.io`, `tsr.sampling`, `tsr.viz` (optional `viz` extra) | Persist, select among, and render recipes/TSRs. | Layers 0–2 |

## The spine

> **A factory produces recipes; a recipe instantiates to math.**

```
gripper.grasp_box(...)  ->  [TSRTemplate]          # factory  (Layer 2)
template.instantiate(T_world)  ->  TSR             # recipe   (Layer 1 -> 0)
tsr.sample() / contains() / distance()             # math     (Layer 0)
```

If a piece of code isn't (a) the math, (b) a recipe, (c) a factory that emits
recipes, or (d) a service that consumes them, it's misplaced.

## Core consistency contract (Layer 0)

The heart's correctness *is* its mutual consistency. For any `TSR` and pose:

- `sample()` returns a pose that `is_valid()` accepts and `contains()` confirms.
- `contains(T)` is true ⇔ `distance(T)[0] == 0`.
- The RPY ↔ rotation ↔ transform conversions round-trip (including near gimbal
  lock — see `docs/REVIEW.md` C1).

These are exercised exhaustively by the Hypothesis property tests.

## Factory contract (Layer 2)

Every `grasp_*` / `place_*` factory method obeys one uniform contract:

> **An exception reports an invalid request. An empty list reports an empty
> feasible set.**

- **`raise ValueError`** only when an argument is nonsensical on its own — a
  non-positive size, `k < 1`, a reversed `angle_range`. These indicate a bug in
  the caller and must never be silently swallowed.
- **Return `[]`** whenever valid arguments simply admit no grasp — the object is
  wider than the jaws, or a positive-but-large clearance leaves no band. A 6 mm
  cylinder and a 0.5 m clearance are both *valid*; they just yield an empty
  feasible set. Treating one physical infeasibility differently from another
  would recreate the inconsistency this contract removes. Callers can sweep many
  objects without `try`/`except`.
- **Observability:** infeasibility carries a machine-readable reason
  (`exceeds_aperture`, `cannot_straddle`, `insufficient_clearance_band`,
  `finger_too_short`). The *public* factory method emits a single
  `logger.debug` at its boundary (not at each internal early return), so an empty
  result is diagnosable without turning it into an exception. If debug logs prove
  insufficient, add an explicit diagnostic-result API — exceptions do not carry
  that responsibility.
- Templates carry `task` / `subject` / `reference` metadata and the canonical
  gripper frame convention (`z` = approach, `y` = finger opening, `x = y × z`).

## Geometric grasp contract (primitive parallel-jaw grasps)

The factory contract above governs *when* a method returns templates. This
section governs *what a returned grasp template geometrically means*. It is the
executable specification the `#67` analytic oracle and the `#73` Hypothesis
matrix test against (issue `#66`).

### Four assurance layers

A generated grasp template can be validated at increasing strength. Each layer
is a strictly stronger claim than the one above it, and each is tested
separately — a template may attain one without the next:

1. **Representation validity** — the recipe and instantiated `TSR` are
   well-formed: finite `T_ref_tsr`/`Tw_e`/`Bw`, orthonormal rotation, ordered
   bounds, `sample() ⇔ contains() ⇔ distance≈0`. (Established in 2.1.0.)
2. **Geometric soundness** — *every* pose admitted by the template is a
   kinematically viable parallel-jaw pre-grasp (clauses below). Applies to every
   returned template.
3. **Coverage** — every approach family the API advertises is represented,
   subject to explicitly documented discretization by `k` (depths) and `n_minor`
   (torus minor angles). A statement about *which modes exist*, not about any one
   template.
4. **Embodied conformance** — a *named* gripper realizes the idealized template
   in its pinned MuJoCo collision model without unintended penetration
   (validated downstream; see `#74`).

**Coverage is not soundness, and neither is a sampling distribution.** In
particular the sphere-grasp TSR *covers* SO(3) as a set, but sampling independent
uniform roll/pitch/yaw is **not** Haar-uniform on SO(3) (see `#72`). Distribution
claims require their own proof and are never implied by set coverage.

### Soundness clauses

For a primitive solid `S`, finger length `L`, max aperture `A`, requested
preshape `a`, clearance `c`, and **every** pose admitted by a returned template:

1. All inputs, transforms, and bounds are finite and well formed.
2. `0 < a ≤ A`, and `S`'s relevant span along the closing line is strictly
   smaller than `a` (the object fits *between* the inner pad faces).
3. The palm lies outside `S` with at least the promised clearance `c`.
4. The usable finger-depth interval is nonempty (a positive `c` that leaves no
   band yields `[]`, never a reversed depth sequence).
5. Closing along the canonical `±y_EE` directions meets two opposing surfaces of
   `S` within the usable depth interval.
6. The contact normals oppose the closing directions within tolerance.
7. The whole continuous TSR region (`Bw` extrema, midpoint, and interior)
   preserves clauses 2–6.
8. The template declares its primitive, mode, hand-occupied approach, object-frame
   span axis, insertion depth (index/value), and symmetry variant **structurally**,
   via `TSRTemplate.provenance` (`GraspProvenance`). Tests and oracles read that
   record; they never infer semantics by parsing `name`.

This describes a viable *pre-grasp*. It is **not** a claim of dynamic stability
or force closure under arbitrary friction — an explicit non-goal.

### Tolerances and conventions (defined once)

- Length tolerance is scale-aware: `atol = 1e-9 + 1e-6 · scale`, where `scale`
  is the primitive's characteristic dimension. Angles use `1e-6 rad`.
- Contact-normal opposition holds when the angle between the inward surface
  normal and the closing direction is `≤ 1e-6 rad`.
- **The palm and closing axes are defined at the *concrete* end-effector pose,
  not only at `Bw = 0`.** For a TSR coordinate `ξ ∈ Bw`, the end-effector frame in
  the reference frame is `E(ξ) = T_ref_tsr · xyzrpy_to_trans(ξ) · Tw_e` (and
  `T_ref_world · E(ξ)` in the world frame after `instantiate`). The palm is the
  translation of `E(ξ)`; the approach and closing axes are its `z`/`y` columns.
  The oracle evaluates soundness at a concrete `ξ` this way — clause 7 quantifies
  over all `ξ` in the region, so the palm is a function of `ξ`, not the fixed
  `Tw_e` translation. "Outside `S` with clearance `c`" means signed distance
  `≥ c − atol`.
- Frames follow the library convention: `z_EE` = approach, `y_EE` = finger
  opening (fingers always close along `±y_EE`), `x_EE = y_EE × z_EE`.

### Structured provenance

`GraspProvenance` (on `TSRTemplate.provenance`) is a closed, immutable, lossless
value object round-tripping through dict/JSON/YAML. Each field has **one frame
and one meaning**:

| field | frame | meaning |
|---|---|---|
| `primitive`, `mode` | — | validated as a `(primitive, mode)` pair |
| `depth` | approach axis | **insertion depth from the approached primitive surface** [m], `≥ 0`; uniform across primitives |
| `approach` | object | side/family the **hand occupies** (`"+z"`, `"-x"`, `"radial"`, `"tube"`) — *not* the sign of `z_EE` |
| `span_axis` | object | direction the two pad contacts are separated along: an object axis (`x`/`y`/`z`) for boxes, or a yaw-free family (`tangential`/`diameter`) for radial/spherical grasps |
| `depth_index`/`depth_count` | — | `depth_count` is the number of **emitted depth slots** (`≤ k`; it collapses when the usable band is thin), and `(depth_index, depth_count)` identifies a member of the returned family. At feasibility boundaries the slots may share a `depth` value; deduplicating coincident slots is deferred to the usable-depth helper in `#68`. |
| `variant` | — | symmetry variant producing a distinct pose at the same (mode, depth) |
| `params` | object | primitive-specific extras (torus `minor_index`/`minor_angle`, box `slide_axis`/`span`) |

**Oracle vs coverage fields.** The analytic oracle consumes `primitive`, `mode`,
`depth`, and torus `params.minor_angle`; the closing line comes from the *pose*
(the fingers close along `±y_EE`), never from provenance. `approach`,
`span_axis`, `variant`, and `depth_index`/`depth_count` classify **coverage and
symmetry** only. Modes emitted today:

| primitive | modes | span_axis | variants / params |
|---|---|---|---|
| cylinder | `side`, `top`, `bottom` | `tangential` (side), `diameter` (top/bottom) | side: `roll0`/`rollpi` |
| box | `top`, `bottom`, `face` | object axis `x`/`y`/`z` | `params.slide_axis`, `params.span` |
| sphere | `surface` (full SO(3)) | `diameter` | — |
| torus | `side`, `span` | `tangential` (side), `diameter` (span) | side: `flip0`/`flippi`, `params.minor_*` |

### Worked examples

Each snippet is numerically complete: the gripper fixes `L` (finger length) and
`A` (max aperture); the default `preshape a = span + c` and `clearance c = 0.1 ·
graspable_depth` are shown where they matter, so soundness/infeasibility follows
from the numbers alone. `g = ParallelJawGripper(finger_length=L, max_aperture=A)`.

- **Cylinder** — *sound:* `L=0.08, A=0.14`; `g.grasp_cylinder_side(0.03, 0.12)`.
  Diameter `2r = 0.06 < a ≈ 0.06 + c`; `L=0.08 > r=0.03` so the palm at radial
  standoff `ro = r + L − depth` stays outside the surface. *Infeasible:*
  `g.grasp_cylinder_side(0.10, 0.12)` → `[]` with reason `exceeds_aperture`
  (`2r = 0.20 > A`). (Separately, `L < r` must not place the palm inside — `#69`.)
- **Box** — *sound:* `L=0.055, A=0.14`; `g.grasp_box_top(0.05, 0.06, 0.07)` — for
  span-x the pads straddle `box_x=0.05 < A` and slide along y. *Infeasible
  orientation:* `g.grasp_box_top(0.20, 0.06, 0.07)` drops the span-x orientation
  (`box_x=0.20 > A`) but keeps span-y (per-face, per-orientation; `#70`).
- **Sphere** — *sound:* `L=0.08, A=0.14`; `g.grasp_sphere(0.03)` (mode `surface`,
  full SO(3); `2r=0.06 < a`). *Infeasible:* `g.grasp_sphere(0.10)` → `[]`,
  `exceeds_aperture` (`2r=0.20 > A`).
- **Torus** — *sound:* `L=0.08, A=0.30`; `g.grasp_torus_side(0.06, 0.02)`
  (tube diameter `2r=0.04 < a`). *Infeasible span:* `g.grasp_torus_span(0.06, 0.02)`
  on `A=0.14` → `[]`, because `2(R+r)+c = 0.16+ > A` (outer diameter exceeds the
  jaw). Torus coverage semantics (`n_minor`, inner/outer half) are refined in `#71`.
