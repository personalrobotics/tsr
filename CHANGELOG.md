# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Work toward 2.2.0 — establishing the *geometric* correctness of primitive
parallel-jaw grasp templates (epic #65).

### Fixed
- **Gripper parameter hardening and empty-depth semantics** (#68). An excessive
  positive `clearance` (more than half the finger length) used to make the
  cylinder-cap, box, and torus-span factories build depths with
  `linspace(clearance, finger_length - clearance, k)` over a *reversed* interval,
  silently emitting templates with a backwards shallow-to-deep ordering. A shared
  `_usable_depths` helper now returns `[]` with reason `insufficient_clearance_band`
  when the band is empty, under an **exact, scale-independent** boundary policy
  (`hi < lo` → empty; `hi == lo` → one deduplicated slot; `hi > lo` → `k` depths) —
  a positive-width interval never collapses through a default `np.isclose` tolerance,
  and a single-slot template gets a single-slot human label (#105). Input validation
  is centralized and applied uniformly across every factory: finite/positive
  `finger_length`, `max_aperture`, primitive dimensions and `preshape`; finite,
  nonnegative `clearance_fraction` **and every explicit per-call `clearance`**
  (NaN/inf/negative/Boolean raise a `ValueError` naming `clearance`, before it
  derives any preshape/bounds/depths; `clearance=0` stays valid) (#104); integer
  `k >= 1` and `n_minor >= 1`; finite, ordered `angle_range`; and positive
  `cylinder_height` for the side/top/bottom entry points alike. Nonsensical arguments
  raise `ValueError`; valid arguments with an empty feasible set return `[]` with one
  coherent diagnostic log. No generated template contains NaN or infinity.
  Property-tested with non-finite/zero/negative scalars, `nextafter` boundary
  neighbours at small/ordinary/large scale, excessive clearances, and
  integral/non-integral counts.

### Added
- **Independent analytic grasp oracle** (#67, #93–#102, test-side) in
  `tests/tsr/hands/_grasp_oracle.py`: certifies the geometric-soundness clauses of
  the #66 contract for box / cylinder / sphere / torus. Contact geometry is
  **exact per `(primitive, mode)`** — closed-form line/surface intersection derives
  jaw contacts, span, analytic normals, realized insertion depth, torus minor
  angle, and clearance-aware usable bands from the concrete pose (no sampling grid,
  so results are resolution-independent and scale-equivariant; ~0.1 ms/cert). An
  independent signed-distance representation is used only to *verify* those
  analytic contacts, never to search (#93). `mode` (and optional `approach` /
  `finger_orientation`) select and are *checked against* the pose — a mislabeled
  mode fails — and mode-specific clearance bands (cylinder ends, box edges) are
  enforced (#94). All inputs are validated up front: a malformed model raises
  `ValueError`, a malformed pose returns a structured clause-1 failure, and a
  reflection (`det(R) = -1`) or non-finite clearance is rejected (#95). Clearance
  bands are enforced along the full insertion interval for cap/face grasps, not
  just laterally (#96); aperture fit is **pose-relative** (both contacts must lie
  between the open jaws about the palm, so an object translated outside the finite
  opening no longer certifies on span alone, #97); declared `approach` labels use
  the hand-occupied-side convention with a built-in vocabulary that raises on
  unknown labels and fails clause 8 on a geometric mismatch (#98); and the
  torus-side model handles the full approach-minor-angle range with tube-centre
  ray verification, reporting the pose-derived approach minor angle (#99).
  Axis-aligned modes use a true angular tolerance and every accepted contact must
  lie in the concrete posed finger sweep `p + t·z_EE + s·y_EE` (#100); every
  successful witness has finite geometry and a contact plane strictly on the
  forward finger segment `0 < t ≤ L`, with `radial`/`tube`/occupied-side predicates
  checked against a shared pose-derived local frame (#101); and a deterministic
  acceptance matrix covers each mode's boundaries, axis permutations, azimuths,
  scale regimes, and real-factory provenance across all families (#102). Returns a
  structured `GraspWitness` naming the violated clause and its geometry. Its unit
  tests use hand-derived poses only. NumPy-only, no shipped-package change; it
  backs the generator repairs in #68–#73.
- **Executable geometric grasp contract** (#66) in `docs/ARCHITECTURE.md`: four
  assurance layers (representation / soundness / coverage / embodied), eight
  soundness clauses, and the soundness-vs-coverage-vs-distribution distinctions,
  with defined tolerances and one sound + one infeasible worked example per
  primitive.
- **`GraspProvenance`** (#66, #78–#84) — a small, immutable, **extensible** value
  object on `TSRTemplate.provenance` recording each grasp's primitive, mode,
  hand-occupied `approach`, object-frame `finger_orientation`, insertion `depth`,
  `symmetry`, and descriptive `metadata`. It validates only representation
  invariants and round-trips losslessly; it is *not* a closed schema, so external
  grasp generators can use their own labels. The sstsr native vocabulary and
  cross-field relational checks live separately in
  `tsr.hands._conformance.validate_builtin_provenance`, which the built-in factory
  tests run over every emitted template. The design principle — **provenance
  declares intent; the pose determines geometric truth** — keeps these two layers
  and the analytic oracle (#67) distinct. Every `ParallelJawGripper` factory emits
  a conformant record, so tests read grasp modes structurally instead of parsing
  `name`. The sphere grasp is mode `surface` (full SO(3)), not `equatorial`.
  Representation-only; grasp geometry is unchanged.
- **Witness-aware, bounded `TSRChain` inverse membership** (#85). A chain is an
  exact parameterized set, but deciding membership from a pose alone is a bounded
  nonconvex inverse problem: a local solve can produce a positive witness, but
  failing to find one is **not** a proof of nonmembership. New API makes that
  distinction explicit:
  - `TSRChain.sample_with_witness(rng=None) -> ChainSample` returns a sampled
    `pose` and the `(n, 6)` `coordinates` that constructed it — a positive
    membership certificate needing no optimizer.
  - `TSRChain.validate_witness(pose, coordinates, tolerance=EPSILON) -> bool`
    certifies a witness with one bounds check and one forward composition — no
    SciPy, no solve.
  - `TSRChain.solve(pose, initial_guess=None, max_starts=11, max_nfev=2200) ->
    ChainSolveResult` runs a bounded, deterministic multi-start inverse solve that
    warm-starts from `initial_guess`, exits on the first tolerance-satisfying
    witness, and returns `status` (`"satisfied"`/`"not_found"`, never
    `"infeasible"`), `coordinates`, `residual`, `nfev`, and `starts`. `max_starts`
    is a **strict total** cap on optimizer starts and `max_nfev` a **strict total**
    cap on objective evaluations, enforced by an internal global counter rather
    than SciPy's soft `maxfun` (#89, #91); the valid-witness fast path takes one
    forward composition and does not import SciPy (#88); numerical controls are
    validated before SciPy is imported.
  - `ChainSample` and `ChainSolveResult` are exported from `tsr`.
- **`TSRChain.to_transform` canonicalizes wrapping rotational coordinates** before
  clamping (#87): a valid RPY coordinate expressed in `[-pi, pi]` for a wrapping
  interval (e.g. `-3.0` for `[3π/4, -3π/4]`) is wrapped into the component's
  continuous interval instead of being clipped to an unrelated boundary rotation,
  which had silently changed the pose and broken the witness contract. A single
  shared coordinate→continuous-chart helper is used by both `to_transform` and the
  inverse solver's start construction, so a wrapping neighboring-state
  `initial_guess` is canonicalized, not clipped, before it warm-starts the
  optimizer (#90). `validate_witness` on an empty chain now returns `False` instead
  of raising.

### Changed
- **`TSRChain.contains`, `distance`, and `closest_transform` are documented as
  numerical, non-certifying for multi-TSR chains** (#85). Multi-TSR `contains`
  now delegates to `solve` and accepts an optional `initial_guess` for the exact
  warm-start fast path; `False` means no witness was found within the numerical
  budget, not certified nonmembership. `distance` returns `0` with the witness
  when one is found, else a best-found residual (an upper bound on the true
  minimum). Single-TSR `contains` remains the exact closed-form check.

## [2.1.0] — 2026-09

### Added
- **`TSR.closest_transform(trans)`** (#53) and **`TSRChain.closest_transform(trans)`**
  (#63) — return the distance *and* the closest in-bounds pose in the **world
  frame** (`T0_w @ xyzrpy_to_trans(bwopt) @ Tw_e` for a TSR; the composed
  transform for a chain), so planner projection code no longer reconstructs the
  frame composition by hand (a recurring source of frame bugs).
- `Robotiq2F140.PALM_OFFSET_FROM_BASE_MOUNT = 0.100` (base_mount → grasp_site along
  approach), for symmetry with `Robotiq2F85`.

### Fixed
- **Multi-TSR `TSRChain` could still reject a pose from its own `sample()`** (#57).
  The 2.0.1 fix removed the zero-width-gradient stall, but a single midpoint-start
  L-BFGS-B could converge to an ordinary nonzero local minimum (reporting success,
  no warning) and reject a constructively-valid sample — especially with
  non-identity component frames. `TSRChain.distance` now minimises a **smooth
  squared pose residual** (the geodesic's `arccos` is non-smooth at 0 and breeds
  local minima) from **deterministic multi-start** points, and reports the true
  geodesic at the recovered coordinates. Verified across the deterministic
  non-identity-frame stress reproduction and a multi-TSR Hypothesis property.

### Changed
- **`Robotiq2F140.MAX_APERTURE` corrected `0.140 → 0.128 m`** (#60) — the usable
  inner-face-to-inner-face gap at full open, measured from `2f140.xml`. The old
  0.140 was the manufacturer's nominal/outer figure and overstated the gap by
  ~12 mm, claiming feasible grasps of objects that don't fit between the pads
  (same distinction as the `Robotiq2F85`'s 0.085 m). **Behavior change:** some
  wide grasps that previously returned templates now return `[]`.

### Documentation
- Reconciled the preconfigured grippers' `finger_length` documentation (#58). All
  three (`Robotiq2F140` 0.114 m, `Robotiq2F85` 0.059 m, `FrankaHand` 0.037 m) use
  one convention — **palm → pad tip** (finger reach along approach) — now stated
  consistently in the class docstrings, the README table, and the tests. Fixed the
  `Robotiq2F140` docstring (it claimed the wrong derivation and a stray "82 mm"),
  the stale README values (55 mm / 44.5 mm), the "pad-contact midpoint" misnomer,
  and a tutorial example that mislabeled a generic 55 mm gripper as a "2F-140".
- Finished the pad-tip terminology reconciliation (#64): the `FrankaHand.FINGER_LENGTH`
  inline comment and the historical 2.0.0 changelog line no longer say "pad-mid" /
  "pad-contact midpoint"; all named-gripper docs now say **palm → pad tip**.

## [2.0.1] — 2026-09

Patch release: correctness and reproducibility fixes from the post-merge audit
of #54. No API removals; adds an optional `rng` parameter.

### Fixed
- **Multi-TSR `TSRChain` could reject a pose from its own `sample()`** (#57).
  `TSRChain.distance` optimised over all `6·n` coordinates, so a point-bound
  coordinate (e.g. a fixed `[0, 0]`) gave a zero-width finite-difference step,
  a NaN gradient, and a stalled optimiser. It now optimises only the free
  coordinates, holding fixed ones at their value.
- **Documentation described `TSRChain` as "AND"/intersection semantics** (#55).
  A chain is the serial composition of its component TSRs, not the intersection
  of their world-frame pose sets (two serial `x∈[0,1]` TSRs reach `x=1.5`).
  Reworded the changelog, `contains()` docstring, and the semantics tests.

### Added
- **`rng` parameter** on `TSR.sample`/`sample_xyzrpy`, `TSRChain.sample`/
  `sample_xyzrpy`, and `TSRTemplate.sample`; `sample_from_tsrs` /
  `sample_from_templates` forward it (#52). Passing a `numpy.random.Generator`
  now makes the sampled *pose* reproducible, not just the TSR choice. Defaults to
  the global RNG when omitted.

### CI
- The downstream-dispatch job no longer reddens an otherwise-green run when
  `ROBOT_CODE_DISPATCH_TOKEN` is absent; it skips with a visible notice (#56).

## [2.0.0] — 2026-09

First PyPI release. The distribution is named **`sstsr`** (the name `tsr` was
already taken); the import name is unchanged — `import tsr`.

### Packaging & licensing
- **Distribution renamed to `sstsr`.** `pip install sstsr`, then `import tsr`.
- **Relicensed to BSD-2-Clause**, matching the original PrPy/OpenRAVE-derived TSR
  core (resolves the prior README-vs-LICENSE mismatch).
- Grasp/hand classes (`ParallelJawGripper`, `Robotiq2F85`, `Robotiq2F140`,
  `FrankaHand`) are now exported at the top level alongside `StablePlacer`.
- `tsr.__version__` is now available.
- Tested on Python 3.10–3.14 (classifiers and CI matrix extended to 3.13 and 3.14).

### Breaking changes
- **`TablePlacer` removed** — use `tsr.placement.StablePlacer`.
- **`TSRChain.contains()` now tests serial-composition membership** — a chain is
  the set of poses reachable by serially composing a transform from each
  component TSR (consistent with `distance()`), not a match against a single
  world-frame pose.
- **Grasp factories return `[]` for infeasible geometry instead of raising.**
  `grasp_cylinder_*` and `grasp_sphere` previously raised `ValueError` when the
  object exceeded the aperture; they now return `[]`, consistent with the box and
  torus factories. Exceptions are reserved for invalid *arguments* (non-positive
  sizes, `k < 1`, reversed `angle_range`). See `docs/ARCHITECTURE.md`.
- **Gripper geometry corrections** (change sampled poses for existing callers):
  `Robotiq2F140` frame correction and `finger_length`; `FrankaHand.finger_length`
  corrected to **37 mm** (measured from the hand-body forward edge / collar to the
  pad tip, not the finger-joint origin); clearance is now scaled by
  graspable depth, not `finger_length`.

### Fixed
- **RPY near gimbal lock** (`rot_to_rpy`, `rot_within_rpy_bounds`): the
  singularity threshold was far too wide (`1e-3`), snapping pitch to ±90° up to
  ~2.5° early and discarding up to 0.04 rad of rotation. This made `contains()`
  reject poses that `sample()` produced. Tightened to a true-singularity
  threshold; round-trip error dropped from ~0.04 to ~1e-13.
- **`StablePlacer.place_mesh` seating**: objects whose center of mass is not at
  the object-frame origin floated above or sank through the table. The resting
  face now sits at `z = 0` for any COM.
- **`wrap_to_interval`** now guarantees a result in `[lower, lower + 2π)` (a
  floating-point modulo artifact could return exactly `lower + 2π`).
- **`TSRChain.distance`** no longer crashes on outer (wrapping) rotation
  intervals, and a single-TSR chain's `contains()` uses the exact closed-form
  check instead of the optimiser.
- Degenerate meshes (fewer than 4 points, coplanar) raise a clear `ValueError`
  instead of a raw SciPy `QhullError`.
- `grasp_*(k=0)` raises `ValueError` instead of `IndexError`; reversed
  `angle_range` raises instead of silently collapsing yaw freedom.

### Internal
- The pure-math core (`TSR`, `TSRChain`, SE(3) helpers) moved to a `tsr.core`
  subpackage; deep imports such as `from tsr.tsr import TSR` still work via
  back-compat shims.
- Added property-based tests (`hypothesis`) covering the core consistency
  contract, serialization round-trips, and the factory invariants; added an
  architecture-enforcement test (`tsr.core` has no upward imports; `import tsr`
  pulls no heavy viz dependencies).
- Added `docs/REVIEW.md` (pre-release review) and `docs/ARCHITECTURE.md`.
- Removed the dead top-level `templates/` duplicate and the vestigial
  `MANIFEST.in` (hatchling bundles the packaged `tsr/templates/` automatically).

[2.0.0]: https://github.com/personalrobotics/tsr/releases/tag/v2.0.0
