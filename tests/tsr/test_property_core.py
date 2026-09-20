# SPDX-License-Identifier: BSD-2-Clause
# Authors: Siddhartha Srinivasa and contributors to TSR

"""Property-based tests for the pure-math core (TSR, TSRChain, utils).

These encode the core consistency contract from docs/ARCHITECTURE.md:
sample() ⇒ is_valid ⇒ contains ⇒ distance≈0, plus the SE(3) round-trips that
the C1 gimbal-lock fix made accurate everywhere.
"""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from tsr import TSR, TSRChain, wrap_to_interval
from tsr.core.utils import EPSILON, geodesic_distance, rotation_angle

from ._hypothesis_strategies import angles, any_rotation, near_gimbal_rotations, rotations, transforms, tsrs

# --------------------------------------------------------------------------
# SE(3) conversions
#
# Away from gimbal lock the round-trip is exact to ~1e-9 (the C1 fix). Within a
# few tens of microradians of pitch=±90° the rotation matrix rounds to exact
# gimbal lock and one roll/yaw DOF must be discarded, so the round-trip is only
# guaranteed to ~1e-4 there — still far inside the EPSILON=1e-3 tolerance the
# library uses for validity/containment.
# --------------------------------------------------------------------------


@given(R=rotations())
def test_rpy_rot_roundtrip_generic(R):
    R2 = TSR.rpy_to_rot(TSR.rot_to_rpy(R))
    np.testing.assert_allclose(R2, R, atol=1e-9)


@given(R=near_gimbal_rotations())
def test_rpy_rot_roundtrip_near_gimbal(R):
    R2 = TSR.rpy_to_rot(TSR.rot_to_rpy(R))
    np.testing.assert_allclose(R2, R, atol=1e-4)


@given(T=transforms())
def test_xyzrpy_trans_roundtrip(T):
    T2 = TSR.xyzrpy_to_trans(TSR.trans_to_xyzrpy(T))
    np.testing.assert_allclose(T2, T, atol=1e-4)


@given(R=any_rotation())
def test_rpy_to_rot_is_a_proper_rotation(R):
    R2 = TSR.rpy_to_rot(TSR.rot_to_rpy(R))
    np.testing.assert_allclose(R2 @ R2.T, np.eye(3), atol=1e-9)
    assert np.isclose(np.linalg.det(R2), 1.0, atol=1e-9)


# --------------------------------------------------------------------------
# utils
# --------------------------------------------------------------------------


@given(x=st.floats(-100, 100, allow_nan=False), lo=angles)
def test_wrap_to_interval_in_range_and_idempotent(x, lo):
    lower = np.array([lo])
    w = wrap_to_interval(np.array([x]), lower=lower)
    assert lower[0] - 1e-9 <= w[0] < lower[0] + 2 * np.pi + 1e-9
    np.testing.assert_allclose(wrap_to_interval(w, lower=lower), w, atol=1e-9)


@given(x=st.floats(-100, 100, allow_nan=False), lo=angles)
def test_wrap_to_interval_preserves_angle_mod_2pi(x, lo):
    w = wrap_to_interval(np.array([x]), lower=np.array([lo]))
    diff = (x - w[0]) / (2 * np.pi)
    np.testing.assert_allclose(diff, np.round(diff), atol=1e-9)


@given(R=any_rotation())
def test_rotation_angle_in_0_pi(R):
    a = rotation_angle(R)
    assert -1e-9 <= a <= np.pi + 1e-9


@given(A=transforms(), B=transforms())
def test_geodesic_distance_symmetry_and_nonneg(A, B):
    d_ab = geodesic_distance(A, B)
    d_ba = geodesic_distance(B, A)
    assert d_ab >= -1e-12
    np.testing.assert_allclose(d_ab, d_ba, atol=1e-9)


@given(A=transforms())
def test_geodesic_distance_zero_on_identity(A):
    # arccos((trace-1)/2) is ill-conditioned at angle 0 (slope -> inf), so a
    # ~1e-15 orthonormality error in A maps to a ~1e-7 angle. That's expected.
    np.testing.assert_allclose(geodesic_distance(A, A), 0.0, atol=1e-6)


# --------------------------------------------------------------------------
# Core consistency contract
# --------------------------------------------------------------------------


@given(tsr=tsrs(), data=st.data())
def test_sample_is_valid_and_contained(tsr, data):
    # A pose drawn from the TSR must satisfy is_valid, contains, and distance≈0.
    xyzrpy = tsr.sample_xyzrpy()
    assert all(tsr.is_valid(xyzrpy))
    pose = tsr.to_transform(xyzrpy)
    assert tsr.contains(pose)
    dist, _ = tsr.distance(pose)
    assert abs(dist) < EPSILON


@given(tsr=tsrs(), T=transforms())
def test_contains_iff_distance_zero(tsr, T):
    # contains() and distance() must agree for arbitrary probe poses.
    contained = tsr.contains(T)
    dist, _ = tsr.distance(T)
    if contained:
        assert abs(dist) < EPSILON
    else:
        assert dist > 0.0


@given(tsr=tsrs())
def test_tsr_dict_json_yaml_roundtrip(tsr):
    for revived in (
        TSR.from_dict(tsr.to_dict()),
        TSR.from_json(tsr.to_json()),
        TSR.from_yaml(tsr.to_yaml()),
    ):
        np.testing.assert_allclose(revived.T0_w, tsr.T0_w)
        np.testing.assert_allclose(revived.Tw_e, tsr.Tw_e)
        np.testing.assert_allclose(revived.Bw, tsr.Bw)


# --------------------------------------------------------------------------
# TSRChain
# --------------------------------------------------------------------------


@given(tsr=tsrs())
def test_single_chain_sample_is_contained(tsr):
    chain = TSRChain(TSRs=[tsr])
    pose = chain.sample()
    assert chain.contains(pose)


# Multi-TSR chains, non-identity frames and mixed fixed/free coords (#57). The
# multi-start solver must recognise every constructive sample as a member.
# Capped example count: the chain-distance solve is much heavier than the core.
@settings(max_examples=40)
@given(parts=st.lists(tsrs(), min_size=2, max_size=3))
def test_multi_chain_sample_is_contained(parts):
    chain = TSRChain(TSRs=parts)
    pose = chain.sample()
    assert chain.contains(pose)
    assert abs(chain.distance(pose)[0]) < EPSILON


@given(tsr=tsrs())
def test_chain_dict_roundtrip(tsr):
    chain = TSRChain(TSRs=[tsr])
    revived = TSRChain.from_dict(chain.to_dict())
    assert len(revived.TSRs) == 1
    np.testing.assert_allclose(revived.TSRs[0].Bw, tsr.Bw)
