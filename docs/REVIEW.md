# Pre-release code & architecture review — `tsr` → `sstsr` v2.0.0

Date: 2026-06. Scope: full source review for correctness + architecture ahead of the first
PyPI release. Every "CONFIRMED" item below was reproduced with an executable probe.

## Severity summary

| # | Severity | Area | Status |
|---|----------|------|--------|
| C1 | **High** | Near-gimbal-lock RPY snapping breaks `contains()`/`to_xyzrpy()` | CONFIRMED |
| C2 | **High** | `place_mesh` mis-places objects when COM ≠ object-frame origin | CONFIRMED |
| C3 | Med | `grasp_*(k=0)` → `IndexError` | CONFIRMED |
| C4 | Med | Reversed `angle_range=(max,min)` silently collapses yaw freedom | CONFIRMED |
| C5 | Med | Degenerate/flat/<4-pt mesh → raw `QhullError` | CONFIRMED |
| C6 | Low | `tsr.py:347` dead no-op `numpy.hstack(...)` | CONFIRMED |
| C7 | Med | Oversized-object policy inconsistent (raise vs `[]`) across shapes | CONFIRMED |
| A1 | **High (release)** | `pyproject` name/version still `tsr`/`1.0.0` | — |
| A2 | — | ~~CI triggers on `main`; default is `master`~~ — FALSE ALARM (stale local refs; remote default is `main`, CI is correct) | RETRACTED |
| A3 | Med | `__init__` docstring advertises non-existent `TaskCategory/TaskType/EntityClass` | CONFIRMED |
| A4 | Med | README says BSD-2-Clause; LICENSE/pyproject say MIT (contradiction) | CONFIRMED |
| A5 | Med | README install uses git URL, not `pip install sstsr` | — |
| A6 | Low | Hands not exported at top level (asymmetric with `StablePlacer`) | decision |
| A7 | Low | 3 core files BSD-2-Clause vs MIT elsewhere (derived from PrPy/OpenRAVE) | decision |
| A8 | Low | `TSRChain.__init__` triple alias `TSR=/TSRs=/tsr=` is error-prone | decision |
| A9 | Low | Top-level `templates/` is a dead byte-identical duplicate | — |

## Correctness findings (detail)

### C1 — Near-gimbal-lock RPY snapping (High)
`tsr.py` `rot_to_rpy` (L73) and `rot_within_rpy_bounds` (L192) use `EPSILON=0.001` as the
gimbal-lock detection threshold. When `|rot[2,0]|` is merely *near* 1 (pitch within ~2.5° of
±90°), they snap pitch→±π/2 and yaw→0, discarding up to **0.04 rad** of real rotation.
- `xyzrpy_to_trans(trans_to_xyzrpy(T))` round-trip error up to 0.042 (vs ~1e-15 elsewhere).
- **`contains()` returns False for poses that `sample()` produced and `is_valid()` accepts**
  (reproduced: pitch=−1.602, ~1/1500 random TSRs).
- Root cause: in the general branch `cos(pitch) > 0` and `atan2` is scale-invariant, so the
  general formula is accurate up to *true* gimbal lock (`cos(pitch)≈0`). Fix: tighten the
  threshold to ~`1e-9` (catch only genuine 0/0) and `np.clip(rot[2,0], -1, 1)` before `arcsin`.
  Prototype drops round-trip error 0.04 → 4e-5 and resolves the `contains` inconsistency.

### C2 — `place_mesh` resting-height bug (High)
`stable_placer.py:_tw_e` (L67) sets the object-frame **origin** z to `com_height` (perpendicular
COM→face distance). The resting face then lands at world `z = -(n·com)`, which is 0 only when
the COM lies on the face-normal line through the origin. For an off-center COM the object floats
or penetrates the table.
- Reproduced: unit cube with origin at a corner (COM at center) floats **0.5 m** above the table.
- Correct translation is `-d` (origin→face-plane distance from `hull.equations`), which yields
  resting face at z=0 **and** COM at world height `com_height` (so the margin stays consistent).
  Verified: fix gives min-vertex z=0, COM z=0.5. Primitives are unaffected (COM = origin there).

### C3 — `grasp_*(k=0)` IndexError (Med)
`parallel_jaw.py:_depth_label` (L19-20): depth arrays use `linspace(..., max(k,1))` (rescuing
`k=0` to 1 element) but `_depth_label(0,0)` indexes an empty fallback list → `IndexError`.
Reachable from every public `grasp_*`. Fix: validate `k >= 1`.

### C4 — Reversed `angle_range` (Med)
`angle_range=(max,min)` is never validated; `TSR.__init__` only checks translation bounds, so
the rotational wrap silently treats it as a zero-width outer interval → all yaw freedom lost with
no signal. Fix: validate `angle_range[0] <= angle_range[1]`.

### C5 — Degenerate mesh (Med)
`_stable_poses.py:93` `ConvexHull(vertices)` raises raw `scipy ... QhullError` for <4 points or
coplanar/flat input. Fix: validate shape/point-count and wrap in a friendly `ValueError`.

### C6 — Dead line (Low)
`tsr.py:347` `numpy.hstack((xyz, rpy))` result is discarded. Delete.

### C7 — Inconsistent oversized-object policy (Med)
`grasp_cylinder_*`/`grasp_sphere` **raise ValueError** when the default preshape exceeds the
object, while `_box_face_templates`/`grasp_torus_span` **return `[]`**. Pick one contract
(recommend: return `[]` for infeasible geometry; reserve exceptions for invalid inputs) and
document it. The equality threshold itself is consistent (no off-by-one).

## Architecture / packaging (detail)
- **A2 (RETRACTED):** initially flagged as "CI runs on `main` but the default is `master`," based on
  stale local remote-tracking refs. The remote default branch is in fact `main` (actively developed),
  and there is no `master` on the remote — so `ci.yml` triggering on `main` is correct. No change.
- **A1/A5:** flip `pyproject` to `name="sstsr"`, `version="2.0.0"`; README install → `pip install
  sstsr` (`import tsr`), `sstsr[viz]` extra.
- **A3:** delete the `TaskCategory/TaskType/EntityClass` docstring line (names don't exist).
- **A4/A7:** README says BSD; LICENSE is MIT. `tsr.py`/`tsr_chain.py`/`utils.py` keep BSD-2-Clause
  (derived from Berenson/Koval PrPy/OpenRAVE TSR per CONTRIBUTORS.md; the `f46df56` MIT relicense
  deliberately left them). Recommend: package MIT, add a short Licensing note + `NOTICE` for the
  three BSD files. Do **not** silently flip headers.
- **A6:** `StablePlacer` is re-exported at top level but the grasp classes aren't — asymmetric.
  Recommend exporting `ParallelJawGripper`/`Robotiq2F140`/`FrankaHand`.
- **A8:** `TSRChain([a,b])` silently appends the list as one element (later `AttributeError`).
  Recommend a single `tsrs=` param accepting a TSR or iterable, deprecating the aliases.
- **A9:** top-level `templates/` is byte-identical to `src/tsr/templates/` (the one shipped/used).
  Remove the duplicate; point `MANIFEST.in` at the package copy.

## Verified healthy (no action)
- dict/json/yaml round-trips for TSR/TSRTemplate/TSRChain incl. optional fields — all symmetric.
- All grasp/stable-pose rotation frames orthonormal (det +1) across parameter sweeps.
- Wheel correctly bundles `tsr/templates/**`; clean-venv install loads templates and runs.
- `rpy_to_rot(rot_to_rpy(R)) ≈ R` to 4e-15 (the failure is only in the snapping branch, C1).
