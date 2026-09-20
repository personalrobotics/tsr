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
8. The template declares its primitive, mode, approach side, finger-opening
   axis, depth index/value, and symmetry variant **structurally**, via
   `TSRTemplate.provenance` (`GraspProvenance`). Tests and oracles read that
   record; they never infer semantics by parsing `name`.

This describes a viable *pre-grasp*. It is **not** a claim of dynamic stability
or force closure under arbitrary friction — an explicit non-goal.

### Tolerances and conventions (defined once)

- Length tolerance is scale-aware: `atol = 1e-9 + 1e-6 · scale`, where `scale`
  is the primitive's characteristic dimension. Angles use `1e-6 rad`.
- Contact-normal opposition holds when the angle between the inward surface
  normal and the closing direction is `≤ 1e-6 rad`.
- "Palm" is the TSR-frame origin of the end-effector (the `Tw_e` translation);
  "outside `S` with clearance `c`" means signed distance `≥ c − atol`.
- Frames follow the library convention: `z_EE` = approach, `y_EE` = finger
  opening, `x_EE = y_EE × z_EE`.

### Structured provenance

`GraspProvenance` (on `TSRTemplate.provenance`) records `primitive`, `mode`,
`approach`, `opening_axis`, `depth_index`/`depth_count`, `depth`, `variant`, and
a primitive-specific `params` map (e.g. torus `minor_index`/`minor_angle`, box
`slide_axis`). It round-trips through dict/JSON/YAML. Modes emitted today:

| primitive | modes | variants / params |
|---|---|---|
| cylinder | `side`, `top`, `bottom` | side: `roll0`/`rollpi` |
| box | `top`, `bottom`, `face` | `params.slide_axis`, `params.span` |
| sphere | `equatorial` | — |
| torus | `side`, `span` | side: `flip0`/`flippi`, `params.minor_*` |

### Worked examples

- **Cylinder** — *sound:* `grasp_cylinder_side(r=0.03, h=0.12)` on an
  `L=0.08, A=0.14` gripper (diameter `0.06 < a`; palm outside). *Infeasible:*
  the same gripper on `r=0.10` (`2r ≥ A` → `[]`; and reach `L < r` must not
  place the palm inside — see `#69`).
- **Box** — *sound:* `grasp_box_top(0.05, 0.06, 0.07)`. *Infeasible:* any
  dimension `≥ A` for the relevant span → that orientation is omitted (per-face,
  per-orientation; see `#70`).
- **Sphere** — *sound:* `grasp_sphere(0.03)`. *Infeasible:* `grasp_sphere(0.10)`
  on an `A=0.14` gripper (`2r ≥ A` → `[]`).
- **Torus** — *sound:* `grasp_torus_side(R=0.06, r=0.02)` (tube diameter
  `0.04 < a`). *Infeasible span:* `grasp_torus_span` when `2(R+r) + c > A` → `[]`.
