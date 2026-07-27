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

- **Returns `List[TSRTemplate]`.** An **empty list means geometrically
  infeasible** (e.g. the object is too large for the gripper) — never an error.
  Callers can sweep many objects without `try`/`except`.
- **Raises `ValueError` only for invalid inputs** (non-positive sizes,
  `k < 1`, reversed `angle_range`).
- Templates carry `task` / `subject` / `reference` metadata and the canonical
  gripper frame convention (`z` = approach, `y` = finger opening, `x = y × z`).
