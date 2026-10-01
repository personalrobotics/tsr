# SPDX-License-Identifier: BSD-2-Clause
# Authors: Siddhartha Srinivasa and contributors to TSR

"""The conformance corpus: the Python's answers, for the C++ core to be checked against.

The Python in ``src/tsr/core/`` is the source of truth for the rules, and
``cpp/`` is a second implementation of them (issue #165). This records what the
Python answers, so the C++ can be held to it by ``cpp/tests/test_conformance.cpp``
without either implementation reading the other.

The corpus is a set of regions -- identity and non-identity frames, zero-width rows,
full rotations, outer rotational intervals, near-singular pitch -- and, for each, probe
transforms on and off the region with ``contains``, ``distance`` and
``closest_transform``. Everything here is in the part of the contract that *is*
specifiable across implementations; see docs/CPP.md for what is not, and why the chain's
cold inverse is checked by properties instead.

The corpus is checked in, so a rule change shows up as a diff rather than as a silently
different expectation. Regenerate with

    uv run python tools/tsr_conformance.py
    uv run python tools/tsr_conformance.py --check   # semantics unchanged?
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from tsr import TSR

ARTIFACT = Path(__file__).resolve().parent.parent / "tests" / "reference" / "tsr_conformance.json"
SEED = 20260929


def _frame(x, y, z, roll=0.0, pitch=0.0, yaw=0.0) -> np.ndarray:
    return TSR.xyzrpy_to_trans(np.array([x, y, z, roll, pitch, yaw], dtype=float))


def regions() -> list[tuple[str, TSR]]:
    pi = np.pi
    box = np.array([[-0.05, 0.05], [-0.05, 0.05], [0, 0], [0, 0], [0, 0], [-pi, pi]])
    return [
        ("identity_box_free_yaw", TSR(np.eye(4), np.eye(4), box)),
        ("offset_frames", TSR(_frame(1.0, 2.0, 0.0, 0.0, 0.0, 0.5), _frame(0.1, 0.0, 0.0), box)),
        (
            "tilted_frame_band",
            TSR(
                _frame(0.5, -0.2, 0.3, 0.3, -0.4, 1.2),
                np.eye(4),
                np.array([[-2, 2], [-0.6, 0.6], [0, 0], [0, 0], [0, 0], [-pi, pi]]),
            ),
        ),
        ("point_pose", TSR(_frame(0.2, 0.2, 0.2, 0.1, 0.2, 0.3), np.eye(4), np.zeros((6, 2)))),
        (
            "full_rotations",
            TSR(np.eye(4), np.eye(4), np.array([[0, 0.1], [0, 0], [0, 0], [-pi, pi], [-pi, pi], [-pi, pi]])),
        ),
        (
            "outer_yaw_interval",
            TSR(np.eye(4), np.eye(4), np.array([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [3 * pi / 4, -3 * pi / 4]])),
        ),
        (
            "outer_roll_and_pitch_band",
            TSR(
                _frame(0, 0, 1.0),
                _frame(0, 0, -0.1),
                np.array([[-0.3, 0.3], [-0.3, 0.3], [-0.1, 0.1], [2.5, -2.5], [-0.2, 0.2], [-1.0, 1.0]]),
            ),
        ),
        (
            "near_singular_pitch",
            TSR(
                np.eye(4),
                np.eye(4),
                np.array([[-0.1, 0.1], [-0.1, 0.1], [-0.1, 0.1], [-0.5, 0.5], [pi / 2 - 1e-3, pi / 2], [-0.5, 0.5]]),
            ),
        ),
        (
            "wide_roll_window",
            TSR(
                _frame(0, 0, 0, 0, 0, -2.0),
                _frame(0, 0.05, 0),
                np.array([[-0.02, 0.02], [-0.02, 0.02], [-0.02, 0.02], [-3.0, 3.0], [-0.3, 0.3], [-0.1, 0.1]]),
            ),
        ),
    ]


def probes(t: TSR, rng: np.random.Generator) -> list[np.ndarray]:
    out = []
    for _ in range(6):
        out.append(t.sample(rng=rng))
    for _ in range(6):
        s = t.sample(rng=rng)
        off = _frame(*(rng.uniform(-0.3, 0.3, 3)), *(rng.uniform(-1.0, 1.0, 3)))
        out.append(off @ s)
    for pitch in (np.pi / 2 - 1e-7, -np.pi / 2 + 1e-7, np.pi / 2, -np.pi / 2):
        out.append(t.T0_w @ _frame(0.01, -0.01, 0.0, 0.2, pitch, 0.4) @ t.Tw_e)
    out.append(t.T0_w @ _frame(0.0, 0.0, 0.0, np.pi, 0.0, np.pi) @ t.Tw_e)  # an RPY-equivalent representation
    out.append(t.T0_w @ _frame(0.0, 0.0, 0.0, 0.0, 0.0, -np.pi) @ t.Tw_e)  # yaw exactly at the seam
    return out


def record(name: str, t: TSR, rng: np.random.Generator) -> dict[str, Any]:
    entries = []
    for T in probes(t, rng):
        dist, closest = t.closest_transform(T)
        entries.append(
            {
                "T": T.tolist(),
                "contains": bool(t.contains(T)),
                "distance": float(dist),
                "closest": closest.tolist(),
                "closest_contained": bool(t.contains(closest)),
            }
        )
    xy = t.sample_xyzrpy(rng=np.random.default_rng(7))
    return {
        "name": name,
        "T0_w": t.T0_w.tolist(),
        "Tw_e": t.Tw_e.tolist(),
        "Bw": t.Bw.tolist(),
        "continuous_bounds": t.continuous_bounds.tolist(),
        "volume": t.volume,
        "sample_xyzrpy_seed7_python": xy.tolist(),
        "probes": entries,
    }


def generate() -> dict[str, Any]:
    rng = np.random.default_rng(SEED)
    return {
        "artifact": "sstsr conformance corpus for the native TSR runtime",
        "issue": "https://github.com/personalrobotics/sscbirrt/issues/87",
        "seed": SEED,
        "versions": {"sstsr": importlib.metadata.version("sstsr"), "numpy": np.__version__},
        "regions": [record(name, t, rng) for name, t in regions()],
    }


ATOL = 1e-9  # distances and transforms compare within this; sstsr on another platform differs in the last bits


def mismatches(stored: dict[str, Any], fresh: dict[str, Any], atol: float = ATOL) -> list[str]:
    """Where a fresh sstsr run disagrees with the stored corpus: booleans exactly, floats within atol."""
    out = []
    by_name = {r["name"]: r for r in stored["regions"]}
    for r in fresh["regions"]:
        old = by_name.get(r["name"])
        if old is None:
            out.append(f"{r['name']}: not in the stored corpus")
            continue
        if len(old["probes"]) != len(r["probes"]):
            out.append(f"{r['name']}: {len(old['probes'])} stored probes vs {len(r['probes'])} fresh")
            continue
        for i, (x, y) in enumerate(zip(old["probes"], r["probes"])):
            where = f"{r['name']}[{i}]"
            if x["contains"] != y["contains"]:
                out.append(f"{where}.contains: stored={x['contains']} fresh={y['contains']}")
            if abs(x["distance"] - y["distance"]) > atol:
                out.append(f"{where}.distance: stored={x['distance']!r} fresh={y['distance']!r}")
            if not np.allclose(x["closest"], y["closest"], atol=atol, rtol=0.0):
                diff = np.abs(np.array(x["closest"]) - np.array(y["closest"])).max()
                out.append(f"{where}.closest: max |diff| = {diff:.3g}")
            if x["closest_contained"] != y["closest_contained"]:
                out.append(f"{where}.closest_contained: stored={x['closest_contained']} fresh={y['closest_contained']}")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output", type=Path, default=ARTIFACT)
    args = parser.parse_args(argv)
    fresh = generate()
    if args.check:
        stored = json.loads(args.output.read_text())
        bad = mismatches(stored, fresh)
        if bad:
            print("MISMATCH: sstsr's results on the conformance corpus changed:", file=sys.stderr)
            for line in bad:
                print(f"  {line}", file=sys.stderr)
            return 1
        n = sum(len(r["probes"]) for r in fresh["regions"])
        print(
            f"conformance corpus unchanged ({len(fresh['regions'])} regions, {n} probes, within {ATOL}; "
            f"versions {fresh['versions']})"
        )
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fresh, indent=1, sort_keys=True) + "\n")
    n = sum(len(r["probes"]) for r in fresh["regions"])
    print(f"wrote {args.output} ({len(fresh['regions'])} regions, {n} probes, versions {fresh['versions']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
