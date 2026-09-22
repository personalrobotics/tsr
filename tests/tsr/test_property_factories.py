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
# #68 — common parameter hardening and empty-depth semantics.
# --------------------------------------------------------------------------

nonfinite = st.sampled_from([np.nan, np.inf, -np.inf])


def _every_family(g, r, h, k=3):
    """One call per public grasp family (each returns a list of templates)."""
    return [
        lambda: g.grasp_cylinder(r, h, k=k),
        lambda: g.grasp_box(2 * r, 2 * r, h, k=k),
        lambda: g.grasp_sphere(r, k=k),
        lambda: g.grasp_torus(2 * r, r / 2, k=k),
    ]


@given(bad=st.sampled_from([np.nan, np.inf, -np.inf, 0.0, -0.1]))
def test_non_finite_or_nonpositive_gripper_raises(bad):
    # A malformed gripper is an invalid request -> ValueError at construction (#68).
    for call in (
        lambda: ParallelJawGripper(finger_length=bad, max_aperture=0.1),
        lambda: ParallelJawGripper(finger_length=0.08, max_aperture=bad),
    ):
        try:
            call()
            raise AssertionError(f"expected ValueError for gripper param {bad}")
        except ValueError:
            pass


@given(fl=positive, ma=positive, r=positive, h=positive, bad=nonfinite)
def test_non_finite_geometry_and_preshape_raise(fl, ma, r, h, bad):
    g = ParallelJawGripper(finger_length=fl, max_aperture=ma)
    for call in (
        lambda: g.grasp_cylinder_side(bad, h),
        lambda: g.grasp_cylinder_top(r, bad),  # non-finite cylinder_height
        lambda: g.grasp_sphere(bad),
        lambda: g.grasp_box_top(r, bad, h),
        lambda: g.grasp_torus_side(2 * r, bad),
        lambda: g.grasp_sphere(r, preshape=bad),
        lambda: g.grasp_cylinder_side(r, h, angle_range=(0.0, bad)),
    ):
        try:
            call()
            raise AssertionError(f"expected ValueError for non-finite input {bad}")
        except ValueError:
            pass


@given(fl=positive, ma=positive, r=positive, h=positive, bad=st.sampled_from([0, -1, 2.5, 3.0]))
def test_non_integer_counts_raise(fl, ma, r, h, bad):
    # k and n_minor must be integers >= 1; a float (even 3.0) or < 1 raises (#68).
    g = ParallelJawGripper(finger_length=fl, max_aperture=ma)
    for call in (lambda: g.grasp_sphere(r, k=bad), lambda: g.grasp_torus_side(2 * r, r / 2, n_minor=bad)):
        try:
            call()
            raise AssertionError(f"expected ValueError for count {bad!r}")
        except ValueError:
            pass


@given(fl=positive, ma=positive, r=positive, h=positive, k=depth_counts)
def test_no_template_has_nonfinite_entries(fl, ma, r, h, k):
    # No generated template contains NaN or infinity (#68).
    g = ParallelJawGripper(finger_length=fl, max_aperture=ma)
    for make in _every_family(g, r, h, k):
        for t in make():
            for field in (t.T_ref_tsr, t.Tw_e, t.Bw):
                assert np.all(np.isfinite(field))
            assert np.isfinite(t.preshape).all()
            if t.provenance is not None:
                assert np.isfinite(t.provenance.depth)


@given(fl=positive, ma=st.floats(0.5, 1.0, allow_nan=False), r=st.floats(0.01, 0.05, allow_nan=False), h=positive)
def test_excess_clearance_yields_empty_not_reversed(fl, ma, r, h):
    # A clearance beyond half the finger length empties the [clearance, L-clearance]
    # band for the cap and box families; the factory returns [] rather than silently
    # reversing the shallow-to-deep ordering (#68). The wide aperture rules out
    # exceeds_aperture. (Torus-side/span use a [tube_radius, ...] reach band with a
    # different emptiness threshold and are covered by the #71 soundness suite.)
    g = ParallelJawGripper(finger_length=fl, max_aperture=ma)
    c = fl * 0.6  # strictly greater than fl / 2
    assert g.grasp_cylinder_top(r, h, clearance=c) == []
    assert g.grasp_cylinder_bottom(r, h, clearance=c) == []
    assert g.grasp_box_top(2 * r, 2 * r, h, clearance=c) == []


def test_clearance_band_boundary_nextafter():
    # Deterministic, exact boundary (#68, #105): the [clearance, L-clearance] band is
    # nonempty iff clearance < L/2. At exactly L/2 the endpoints coincide -> one depth;
    # ABOVE returns []; the LOWER nextafter neighbour is a positive-width interval and
    # must emit k ordered depths (it must NOT collapse through a default tolerance).
    fl = 0.08
    half = fl / 2.0  # exactly representable: fl == 2 * half
    g = ParallelJawGripper(finger_length=fl, max_aperture=0.30)
    below = g.grasp_cylinder_top(0.03, 0.12, k=3, clearance=np.nextafter(half, -np.inf))
    assert len(below) == 3
    depths = [t.provenance.depth for t in below]
    assert depths == sorted(depths)  # ordered shallow -> deep, not reversed
    assert len(g.grasp_cylinder_top(0.03, 0.12, k=3, clearance=half)) == 1  # coincident -> one slot
    assert g.grasp_cylinder_top(0.03, 0.12, k=3, clearance=np.nextafter(half, np.inf)) == []  # empty band


@given(scale=st.sampled_from([1e-3, 1.0, 1e3]))
def test_clearance_boundary_is_scale_independent(scale):
    # The exact boundary policy holds at small / ordinary / large scale (#105): a
    # positive-width interval never collapses through a unit-dependent tolerance.
    fl = 0.08 * scale
    half = fl / 2.0
    g = ParallelJawGripper(finger_length=fl, max_aperture=0.30 * scale)
    r, h = 0.03 * scale, 0.12 * scale
    assert len(g.grasp_cylinder_top(r, h, k=3, clearance=np.nextafter(half, -np.inf))) == 3
    assert len(g.grasp_cylinder_top(r, h, k=3, clearance=half)) == 1
    assert g.grasp_cylinder_top(r, h, k=3, clearance=np.nextafter(half, np.inf)) == []


@given(bad=st.sampled_from([np.nan, np.inf, -np.inf, -0.01, True]))
def test_invalid_clearance_raises_identifying_clearance(bad):
    # An explicit NaN/inf/negative/Boolean clearance is invalid across every public
    # specialized AND combined factory, and the message names `clearance` (#104).
    g = ParallelJawGripper(finger_length=0.08, max_aperture=0.30)
    calls = [
        lambda: g.grasp_cylinder_side(0.03, 0.12, clearance=bad),
        lambda: g.grasp_cylinder_top(0.03, 0.12, clearance=bad),
        lambda: g.grasp_cylinder_bottom(0.03, 0.12, clearance=bad),
        lambda: g.grasp_cylinder(0.03, 0.12, clearance=bad),
        lambda: g.grasp_box_top(0.05, 0.06, 0.07, clearance=bad),
        lambda: g.grasp_box_face_x(0.05, 0.06, 0.07, clearance=bad),
        lambda: g.grasp_box(0.05, 0.06, 0.07, clearance=bad),
        lambda: g.grasp_sphere(0.03, clearance=bad),
        lambda: g.grasp_torus_side(0.06, 0.02, clearance=bad),
        lambda: g.grasp_torus_span(0.06, 0.02, clearance=bad),
        lambda: g.grasp_torus(0.06, 0.02, clearance=bad),
    ]
    for call in calls:
        try:
            call()
            raise AssertionError(f"expected ValueError for clearance={bad!r}")
        except ValueError as e:
            assert "clearance" in str(e)


def test_clearance_zero_is_valid():
    # clearance=0 is a valid request (no raise); with a preshape wider than the object
    # it yields templates (a NaN/negative clearance would have raised instead) (#104).
    g = ParallelJawGripper(finger_length=0.08, max_aperture=0.30)
    templates = g.grasp_sphere(0.03, preshape=0.07, clearance=0.0)
    assert len(templates) > 0
    assert all(np.isfinite(t.preshape).all() for t in templates)


def test_insufficient_clearance_band_logs_reason_once(caplog):
    g = ParallelJawGripper(finger_length=0.08, max_aperture=0.30)
    with caplog.at_level(logging.DEBUG, logger="tsr.hands.base"):
        result = g.grasp_box_top(0.05, 0.06, 0.07, clearance=0.05)  # 0.05 > L/2 = 0.04
    assert result == []
    logs = [r.getMessage() for r in caplog.records if "empty feasible set" in r.getMessage()]
    assert len(logs) == 1
    assert "insufficient_clearance_band" in logs[0]


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
