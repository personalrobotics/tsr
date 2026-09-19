# SPDX-License-Identifier: BSD-2-Clause
# Authors: Siddhartha Srinivasa and contributors to TSR

"""Shared Hypothesis strategies for the tsr property tests.

The generators deliberately reach the awkward corners of SE(3):

* ``rotations`` is uniform over SO(3); ``near_gimbal_rotations`` forces pitch to
  within a few degrees of ±90° (the band that produced the C1 bug in
  ``docs/REVIEW.md``); ``any_rotation`` mixes the two.
* ``bw`` draws rotational bounds as two *independent* angles, so it produces both
  inner intervals (lo ≤ hi) and outer/wrapping intervals (lo > hi).

A CI-friendly ``tsr`` Hypothesis profile is registered and loaded on import.
"""

from __future__ import annotations

import numpy as np
from hypothesis import HealthCheck, settings
from hypothesis import strategies as st
from scipy.spatial.transform import Rotation

from tsr.core.tsr import TSR

settings.register_profile(
    "tsr",
    max_examples=200,
    deadline=None,  # SciPy optimiser paths (chain distance) are variable-latency.
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile("tsr")

# A finite, well-behaved angle in [-pi, pi].
angles = st.floats(min_value=-np.pi, max_value=np.pi, allow_nan=False, allow_infinity=False, width=64)

# Bounded translation coordinate.
coords = st.floats(min_value=-0.5, max_value=0.5, allow_nan=False, allow_infinity=False, width=64)


@st.composite
def rotations(draw: st.DrawFn) -> np.ndarray:
    """A rotation matrix drawn uniformly from SO(3) (Hypothesis-controlled seed)."""
    seed = draw(st.integers(min_value=0, max_value=2**32 - 1))
    return Rotation.random(random_state=seed).as_matrix()


@st.composite
def near_gimbal_rotations(draw: st.DrawFn) -> np.ndarray:
    """A rotation whose pitch is within ~3° of ±90° (the gimbal-lock band)."""
    roll = draw(angles)
    yaw = draw(angles)
    sign = draw(st.sampled_from([1.0, -1.0]))
    delta = draw(st.floats(min_value=0.0, max_value=0.05, allow_nan=False, width=64))
    pitch = sign * (np.pi / 2.0 - delta)
    return TSR.rpy_to_rot([roll, pitch, yaw])


def any_rotation() -> st.SearchStrategy:
    """SO(3) rotations, over-sampling the near-gimbal band."""
    return st.one_of(rotations(), near_gimbal_rotations())


@st.composite
def transforms(draw: st.DrawFn) -> np.ndarray:
    """A 4×4 SE(3) transform: any_rotation + bounded translation."""
    T = np.eye(4)
    T[:3, :3] = draw(any_rotation())
    T[:3, 3] = [draw(coords), draw(coords), draw(coords)]
    return T


@st.composite
def bw(draw: st.DrawFn) -> np.ndarray:
    """A valid (6,2) Bw. Translation rows are ordered [lo, lo+w]; rotational rows
    are two independent angles, so both inner and outer (wrapping) intervals occur.
    """
    Bw = np.zeros((6, 2))
    for i in range(3):
        lo = draw(st.floats(min_value=-0.2, max_value=0.2, allow_nan=False, width=64))
        width = draw(st.floats(min_value=0.0, max_value=0.3, allow_nan=False, width=64))
        Bw[i] = [lo, lo + width]
    for i in range(3, 6):
        Bw[i] = [draw(angles), draw(angles)]
    return Bw


@st.composite
def tsrs(draw: st.DrawFn) -> TSR:
    """A TSR with random frames and bounds."""
    return TSR(T0_w=draw(transforms()), Tw_e=draw(transforms()), Bw=draw(bw()))
