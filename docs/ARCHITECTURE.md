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

### Three assurance layers

A generated grasp template can be validated at increasing strength. Each layer
is a strictly stronger claim than the one above it, and each is tested
separately — a template may attain one without the next:

1. **Representation validity** — the recipe and instantiated `TSR` are
   well-formed: finite `T_ref_tsr`/`Tw_e`/`Bw`, orthonormal rotation, ordered
   bounds, `sample() ⇔ contains() ⇔ distance≈0`. (Established in 2.1.0.)
2. **Geometric soundness** — *every* pose admitted by a template returned by a
   primitive grasp factory is a kinematically viable parallel-jaw pre-grasp
   (clauses below).
3. **Coverage** — every approach family the API advertises is represented,
   subject to explicitly documented discretization by `k` (depths) and `n_minor`
   (torus minor angles). A statement about *which modes exist*, not about any one
   template.

These factory guarantees do not extend automatically to hand-authored
`TSRTemplate` recipes. The YAML examples packaged with sstsr are checked for
representation validity, but omit the primitive dimensions and gripper model needed
by the analytic oracle. Their authors and users remain responsible for geometric
feasibility in the intended scene.

**Coverage is not soundness, and neither is a sampling distribution.** In
particular the sphere-grasp TSR *covers* SO(3) as a set, but sampling independent
uniform roll/pitch/yaw (`TSR.sample`) is **not** Haar-uniform on SO(3). Haar-uniform
sampling over a TSR's rotation box is a separate API, `tsr.sampling.sample_haar`,
which draws `sin(pitch)` uniformly (the ZYX Haar density is `∝ cos(pitch)`) (#72).
Distribution claims require their own proof and are never implied by set coverage.

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
   finger orientation, insertion depth (index/value), and symmetry **structurally**,
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
- **Straddle feasibility is decided on the margin, with slack** (#107, #129). The
  object fits between the open jaws when the margin `preshape − span` is at least
  `4·atol`, i.e. each pad clears the object by `2·atol` — twice the `atol` at which the
  oracle's clause 2 admits a pose, leaving one tolerance of slack per side. Two points matter. First the
  test is on the *margin*, which is exact by Sterbenz, not on `preshape` versus
  `span + 4·atol`: when the margin is orders of magnitude smaller than the span, that
  addition loses it and a margin under the floor is accepted. Second the floor is
  *twice* the oracle's tolerance, because at a margin of exactly `2·atol` the oracle's
  clause-2 bound evaluates to the contact coordinate itself and certification is
  decided by rounding — measured at ~59% of sampled poses failing on a sphere. The
  realized margin is quantized by `ulp(span)`, so that, not `ulp(margin)`, is the
  resolution at which the boundary can be probed. A **default** preshape is therefore
  built by `_default_preshape`, which advances `span + c` until the realized margin is
  at least the requested `c`: a public clearance request is met by the nearest
  representable geometry rather than weakened by rounding (#131). An explicitly
  supplied preshape is never altered — it is judged by its own realized margin.
- **Edge-facing band limits use `m = max(c, 2·atol)`** (#121). A limit that faces an
  edge of the primitive — the approached face, the opposite face or cap, a lateral
  slide edge, a cylinder rim — keeps contacts strictly off that edge, where the
  surface normal is ambiguous and the insertion would be degenerate. `c = 0` is a
  valid request (#104), and a clearance below `atol` is indistinguishable from it, so
  the generator floors these limits. The floor is *twice* `atol` because the pose's
  realized insertion is recovered by a cancelling subtraction, so exactly `atol`
  loses to rounding. This mirrors the scale-aware straddle margin, under which the
  object must clear each pad by `2·atol`. Palm clearance and the default preshape still
  use the requested `c`.

### Structured provenance — three separate layers (#84)

**Provenance declares intent; the pose determines geometric truth.** Three
responsibilities are kept apart so the public record does not become a second
grasp DSL:

1. **`GraspProvenance`** (value object) — a small, immutable, **extensible**
   record of the generator's claim. It validates only *representation*
   invariants (nonempty string labels; finite `depth ≥ 0` canonicalized to float;
   exact integer `depth_index`/`depth_count` with `0 ≤ index < count`; JSON-scalar
   `metadata` frozen into a read-only mapping; exact dict/JSON/YAML round-trip).
   Labels are free strings — it does **not** know which `(primitive, mode)` pairs
   exist or how fields relate, so third-party generators can use their own
   vocabulary. Fields:

   | field | frame | meaning |
   |---|---|---|
   | `primitive`, `mode` | — | free labels selecting the analytic model |
   | `depth` | approach axis | insertion depth from the approached surface [m], `≥ 0` |
   | `approach` | object | side/family the **hand occupies** (not the sign of `z_EE`) |
   | `finger_orientation` | object | direction the pads are separated along — object axis (box) or yaw-free family `tangential`/`diameter` (fingers always close along `±y_EE`) |
   | `depth_index`/`depth_count` | — | index (shallow → deep) and number of **distinct** depth levels in the template's *depth family* (`≤ k`; a one-point band gives `depth_count = 1`, an empty band emits nothing) (#81). A depth family is the templates **within one factory result** whose non-depth fields — `primitive`, `mode`, `approach`, `finger_orientation`, `symmetry`, `metadata` — are identical; it holds one template per depth level. The counters do not identify templates across independently generated or concatenated collections. |
   | `symmetry` | — | distinguishes otherwise-equivalent templates (e.g. a roll flip) |
   | `metadata` | object | descriptive extras only — **never** geometric evidence (torus `minor_*`, box `slide_axis`/`span`) |

2. **Native conformance** (`tsr.hands._conformance.validate_builtin_provenance`)
   owns the sstsr vocabulary and cross-field relational checks the value object
   does not: the built-in `(primitive, mode)` set, allowed `approach`/
   `finger_orientation`/`symmetry` labels, required `metadata`, and relations
   (box `finger_orientation ≠ slide_axis`; box-face orientation lies in the face
   plane; torus `minor_index` in range). Built-in factory tests run it over every
   emitted template; it may evolve without closing the public type.

3. **Geometric truth** — the analytic oracle (`#67`) derives contacts, reach,
   clearance, and the realized depth/minor-angle **from the concrete pose and
   primitive geometry**, and must **not** use `provenance.depth`, a recorded
   minor angle, or other metadata as inputs to the calculation that certifies
   those same values. `primitive`/`mode` select the analytic model; everything
   else is checked against the pose, not trusted. A provenance failure means the
   generator described its output inconsistently; an oracle failure means the
   emitted pose is geometrically wrong.

Built-in modes emitted today (native conformance vocabulary):

| primitive | modes | finger_orientation | symmetry / metadata |
|---|---|---|---|
| cylinder | `side`, `top`, `bottom` | `tangential` (side), `diameter` (top/bottom) | side: `roll0`/`rollpi` |
| box | `top`, `bottom`, `face` | object axis `x`/`y`/`z` | `metadata.slide_axis`, `metadata.span` |
| sphere | `surface` (full SO(3)) | `diameter` | — |
| torus | `side`, `span` | `tangential` (side), `diameter` (span) | side: `flip0`/`flippi`, `metadata.minor_*` |

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

## Verification matrix and release gate (#73)

The per-primitive soundness suites (`test_reach_soundness.py`, `test_box_soundness.py`,
`test_torus_soundness.py`, …) are the deterministic regressions. On top of them,
`tests/tsr/hands/_grasp_matrix.py` drives **every** public `grasp_*` factory — the
combined entry points and the named grippers included — through the independent
oracle. A guard test fails if a new factory is added without joining the matrix.

**Cases.** Primitive dimensions are drawn relative to the hand's reach and aperture
across scale regimes, together with `k`, `n_minor`, restricted yaw and minor-angle
ranges, and explicit or default preshape and clearance. `boundary_cases` additionally
solves one clearance that places a boundary **active for that factory** exactly, then
offsets it by ±1 ulp. Solving for the clearance rather than the gripper keeps the
named hardware in the matrix.

**Boundaries are behavioural, not just biased** (#126). A random case sitting near a
boundary proves little: an empty result satisfies soundness, equivariance, symmetry,
serialization and uniqueness *vacuously*, so a closed interval silently becoming open
(the #124 bug class) would escape. `BOUNDARIES` therefore pins each documented
boundary with dimensions that leave every unrelated constraint slack, and asserts the
**transition** of the family it governs across below / exact / above — empty on the
infeasible side, non-empty and oracle-sound on the feasible side and, where the
interval is closed, at the boundary itself. All dimensions are dyadic multiples of a
scale (small `1e-3`, ordinary `1`, large `1e3`), so identities like `h − h/2 == h/2`
hold exactly in binary and the `nextafter` neighbours really do straddle the
comparison the generator makes:

| boundary | active for | closes at |
|---|---|---|
| cap band (#122) | cylinder top/bottom | `c = h/2` (short cylinder, `h < L`) |
| side height band (#124) | cylinder side | `c = h/2` |
| radial reach | sphere, cylinder side, torus side | `c = L − r` (`L < 2r`) |
| radial far surface | sphere, cylinder side, torus side | `c = r` (`2r ≤ L`) |
| box approach band | each box face | `c = min(L, extent)/2` |
| box slide band (#110) | box top | `c = dim/2` |
| torus span reach | torus span | `c = L − r` |
| aperture limit | box top | `c = A − span` |
| straddle floor (#107, #129) | explicit preshape | margin `= 4·atol(scale)` |

The straddle floor is pinned on the **explicit-preshape** path, where the realized
margin `preshape − span` is ulp-exact, and its boundary value is certified like any
other. Through the *default* preshape the margin is built by `span + c`, so realized
margins are quantized by `ulp(span)` — far coarser than `ulp(c)` — and the meaningful
neighbour is one step of that lattice, not one ulp of the clearance (#131).
`tests/tsr/hands/test_straddle_margin.py` exercises that public path across scales and
every factory's own span and scale (#133).

**Invariants.** Each is a pure check returning failures, shared with the gate:

| # | invariant |
|---|---|
| 1 | every pose (Bw midpoint, per-axis extrema, corners, interior points) has an oracle witness, and its provenance passes native conformance |
| 2 | invalid arguments raise `ValueError`; physical infeasibility returns `[]` |
| 3 | equivariance: `instantiate(T)` maps every pose by `T` and preserves the witness |
| 4 | symmetries: rotation about the object axis, and the x/y/z mirrors, preserve the witness under the mapped side labels; declared opposite-side pairs have equal depth multisets |
| 5 | monotonicity: more reach or a wider aperture never removes a family |
| 6 | declared families are all emitted; structured keys are unique; a combined entry point equals the union of its parts — matched by structured key (emission order is not contractual), then compared as whole templates via `to_dict()`, so semantic and display fields (`subject`, `reference`, `name`, …) are covered too (#127) |
| 7 | dict/JSON/YAML round-trips preserve poses and the witness |

**Mutation gate** (`test_grasp_mutation_gate.py`). A suite that never fails proves
nothing, so each bug class named in #73 is injected and the same checks must reject
it: approach sign, face origin, axis mapping, object extent, clearance term and reach
as black-box mutants over the factory output, plus code mutants that patch the real
generator (`_resolve_clearance`, `_usable_depths`, `_infeasibility_reason`).
Detection is required per (mutant, factory), aggregated over a deterministic corpus —
one case suffices, since a mutation can be harmless in one geometry and fatal in
another. Two exclusions are explicit, and both are *correctness* statements:

- rotating a **sphere's** TSR frame is a symmetry, so the axis-mapping mutant stays
  sound (`KNOWN_SYMMETRIC`, re-certified rather than skipped);
- **inflating** an object is not by itself unsound — a grasp planned for a larger
  object can execute soundly on the real, smaller one — so the extent mutant shrinks.

The corpus includes a deliberately short-fingered hand (`L < 2r`), because the radial
deep limit `min(L, 2r) − c` only *binds* there; with long fingers it is conservative
and exceeding it is still a sound grasp that no check may reject.

**Runtime.** Budgets scale with `TSR_MATRIX_SCALE` (default 1): the matrix and gate
add ~13 s to a ~41 s suite. A separate `.github/workflows/release-gate.yml` runs them
at `TSR_MATRIX_SCALE=10` weekly and on demand (`workflow_dispatch`). It is its own
workflow, with its own concurrency group, so a scheduled run does not also launch the
five-version suite and a push to `main` can neither cancel a running gate nor be
cancelled by one (#128). Run it locally with

```bash
TSR_MATRIX_SCALE=10 uv run pytest tests/tsr/hands/test_grasp_matrix.py tests/tsr/hands/test_grasp_mutation_gate.py
```

Hypothesis runs with `deadline=None` and prints a reproduction blob on failure, so
results are stable across Python 3.10–3.14. A confirmed counterexample is shrunk and
promoted to a deterministic regression next to the primitive it belongs to.
