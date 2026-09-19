# SPDX-License-Identifier: BSD-2-Clause
# Authors: Siddhartha Srinivasa and contributors to TSR

"""Property-based tests for the factory layer (hands, placement).

Encodes the factory contract from docs/ARCHITECTURE.md: every returned template
is a valid TSR recipe with an orthonormal frame; infeasible geometry yields [];
placement seats the resting face at table z=0 (the C2 guarantee).
"""

from __future__ import annotations

import logging

import numpy as np
from hypothesis import assume, given
from hypothesis import strategies as st

from tsr import ParallelJawGripper, StablePlacer

positive = st.floats(min_value=0.01, max_value=0.2, allow_nan=False, width=64)
depth_counts = st.integers(min_value=1, max_value=4)


def _is_se3(T):
    R = T[:3, :3]
    return np.allclose(R @ R.T, np.eye(3), atol=1e-9) and np.isclose(np.linalg.det(R), 1.0, atol=1e-9)


# --------------------------------------------------------------------------
# Hands — the factory contract has three complementary halves:
#   1. Intrinsically invalid arguments raise ValueError.
#   2. Valid arguments never raise on account of geometric infeasibility.
#   3. Every returned template satisfies the geometric + SE(3) invariants.
# Output cardinality is deliberately not pinned: it may be zero or reduced near
# geometric boundaries.
# --------------------------------------------------------------------------


def _all_grasps(gripper, r, h, k):
    """Every grasp family, exercised on a cylinder/box/sphere/torus of scale r."""
    return {
        "cylinder_side": lambda: gripper.grasp_cylinder_side(r, h, k=k),
        "cylinder_top": lambda: gripper.grasp_cylinder_top(r, h, k=k),
        "cylinder_bottom": lambda: gripper.grasp_cylinder_bottom(r, h, k=k),
        "box_top": lambda: gripper.grasp_box_top(2 * r, 2 * r, h, k=k),
        "box_face_x": lambda: gripper.grasp_box_face_x(2 * r, 2 * r, h, k=k),
        "sphere": lambda: gripper.grasp_sphere(r, k=k),
        "torus_side": lambda: gripper.grasp_torus_side(2 * r, r / 2, k=k),
    }


# Contract 2: valid arguments never raise, whatever the feasible set turns out to be.
@given(fl=positive, ma=positive, r=positive, h=positive, k=depth_counts)
def test_valid_args_never_raise(fl, ma, r, h, k):
    gripper = ParallelJawGripper(finger_length=fl, max_aperture=ma)
    for make in _all_grasps(gripper, r, h, k).values():
        make()  # must not raise for any valid positive geometry


# Contract 3: every returned template is a valid SE(3) recipe with ordered bounds.
@given(fl=positive, ma=positive, r=positive, h=positive, k=depth_counts)
def test_returned_templates_are_valid(fl, ma, r, h, k):
    gripper = ParallelJawGripper(finger_length=fl, max_aperture=ma)
    for make in _all_grasps(gripper, r, h, k).values():
        for t in make():
            assert _is_se3(t.Tw_e)  # orthonormal frame, det +1
            assert t.Bw.shape == (6, 2)
            assert np.all(t.Bw[0:3, 0] <= t.Bw[0:3, 1] + 1e-12)  # translation bounds ordered


# Contract 1: intrinsically invalid arguments raise ValueError.
@given(fl=positive, ma=positive, r=positive, h=positive)
def test_nonpositive_size_raises(fl, ma, r, h):
    gripper = ParallelJawGripper(finger_length=fl, max_aperture=ma)
    for bad in (
        lambda: gripper.grasp_cylinder_side(-r, h),
        lambda: gripper.grasp_sphere(-r),
        lambda: gripper.grasp_box_top(-r, r, h),
    ):
        try:
            bad()
            raise AssertionError("expected ValueError for non-positive size")
        except ValueError:
            pass


@given(fl=positive, ma=positive, r=positive, h=positive, bad_k=st.integers(max_value=0))
def test_bad_depth_count_raises(fl, ma, r, h, bad_k):
    gripper = ParallelJawGripper(finger_length=fl, max_aperture=ma)
    try:
        gripper.grasp_cylinder_side(r, h, k=bad_k)
        raise AssertionError("expected ValueError for k < 1")
    except ValueError:
        pass


# Infeasibility (a large object) yields [] rather than an exception.
@given(fl=positive, ma=positive, r=positive, h=positive)
def test_infeasible_cylinder_returns_empty(fl, ma, r, h):
    assume(2 * r >= ma)  # diameter at least the max aperture -> ungraspable
    gripper = ParallelJawGripper(finger_length=fl, max_aperture=ma)
    assert gripper.grasp_cylinder_side(r, h) == []


def test_infeasible_grasp_logs_reason_once(caplog):
    # Observability: an empty feasible set is logged once, at the public boundary,
    # carrying a machine-readable reason (not raised, not silent, not duplicated).
    gripper = ParallelJawGripper(finger_length=0.05, max_aperture=0.05)
    with caplog.at_level(logging.DEBUG, logger="tsr.hands.base"):
        result = gripper.grasp_cylinder_side(0.10, 0.10)  # diameter 0.20 >> aperture 0.05
    assert result == []
    empty_logs = [r.getMessage() for r in caplog.records if "empty feasible set" in r.getMessage()]
    assert len(empty_logs) == 1
    assert "exceeds_aperture" in empty_logs[0]


# --------------------------------------------------------------------------
# Placement
# --------------------------------------------------------------------------


@given(
    lx=st.floats(0.02, 0.3, allow_nan=False),
    ly=st.floats(0.02, 0.3, allow_nan=False),
    lz=st.floats(0.02, 0.3, allow_nan=False),
)
def test_box_placements_rest_on_table(lx, ly, lz):
    placer = StablePlacer(0.3, 0.3)
    verts = np.array([[sx * lx / 2, sy * ly / 2, sz * lz / 2] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    for t in placer.place_box(lx, ly, lz):
        assert _is_se3(t.Tw_e)
        pose = t.instantiate(np.eye(4)).to_transform(np.zeros(6))
        world = (pose[:3, :3] @ verts.T).T + pose[:3, 3]
        assert np.isclose(world[:, 2].min(), 0.0, atol=1e-6)  # resting face on the table


@given(seed=st.integers(0, 2**32 - 1), com_off=st.floats(-0.1, 0.1, allow_nan=False))
def test_mesh_placements_invariants(seed, com_off):
    rng = np.random.default_rng(seed)
    pts = rng.normal(size=(12, 3))
    com = pts.mean(0) + np.array([com_off, 0.0, com_off])
    placer = StablePlacer(0.3, 0.3)
    templates = placer.place_mesh(pts, com)
    margins = []
    for t in templates:
        assert _is_se3(t.Tw_e)
        assert t.stability_margin >= -1e-12
        margins.append(t.stability_margin)
        # Resting face sits on the table for any COM (the C2 guarantee).
        pose = t.instantiate(np.eye(4)).to_transform(np.zeros(6))
        world = (pose[:3, :3] @ pts.T).T + pose[:3, 3]
        assert np.isclose(world[:, 2].min(), 0.0, atol=1e-6)
    # Templates are ordered most-stable first.
    assert margins == sorted(margins, reverse=True)
