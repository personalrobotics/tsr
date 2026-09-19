# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
- **`TSRChain.contains()` uses AND semantics** — a transform must satisfy all
  TSRs in the chain simultaneously (was OR).
- **Grasp factories return `[]` for infeasible geometry instead of raising.**
  `grasp_cylinder_*` and `grasp_sphere` previously raised `ValueError` when the
  object exceeded the aperture; they now return `[]`, consistent with the box and
  torus factories. Exceptions are reserved for invalid *arguments* (non-positive
  sizes, `k < 1`, reversed `angle_range`). See `docs/ARCHITECTURE.md`.
- **Gripper geometry corrections** (change sampled poses for existing callers):
  `Robotiq2F140` frame correction and `finger_length`; `FrankaHand.finger_length`
  corrected to 54 mm; clearance is now scaled by graspable depth, not
  `finger_length`.

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
