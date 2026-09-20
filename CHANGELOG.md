# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Documentation
- Reconciled the preconfigured grippers' `finger_length` documentation (#58). All
  three (`Robotiq2F140` 0.114 m, `Robotiq2F85` 0.059 m, `FrankaHand` 0.037 m) use
  one convention — **palm → pad tip** (finger reach along approach) — now stated
  consistently in the class docstrings, the README table, and the tests. Fixed the
  `Robotiq2F140` docstring (it claimed the wrong derivation and a stray "82 mm"),
  the stale README values (55 mm / 44.5 mm), the "pad-contact midpoint" misnomer,
  and a tutorial example that mislabeled a generic 55 mm gripper as a "2F-140".

### Added
- `Robotiq2F140.PALM_OFFSET_FROM_BASE_MOUNT = 0.100` (base_mount → grasp_site along
  approach), for symmetry with `Robotiq2F85`.

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
  pad-contact midpoint, not the finger-joint origin); clearance is now scaled by
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
