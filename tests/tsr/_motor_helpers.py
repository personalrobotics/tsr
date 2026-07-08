# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Test-only Motor helpers.

The ``tsr`` package no longer ships a transform-coercion module — its public
boundary is the polymorphic ``gafropy.Motor`` constructor and Motor methods.
The tests, however, assert against 4x4 matrices and the CGA split coordinates,
so these thin gafropy wrappers live here, in the test tree, purely to keep the
assertions readable. They are not part of the ``tsr`` public API.
"""

from __future__ import annotations

import numpy as np

from gafropy import Motor, Rotor


def as_motor(x) -> Motor:
    """Coerce a Motor / 4x4 matrix / 6-vector bivector log into a Motor."""
    return Motor(x) if isinstance(x, Motor) else Motor(np.asarray(x, dtype=float))


def to_matrix(m) -> np.ndarray:
    """4x4 transformation matrix of a Motor (or 4x4 passthrough)."""
    if isinstance(m, np.ndarray) and m.shape == (4, 4):
        return m
    return np.asarray(as_motor(m).to_transformation_matrix(), dtype=float)


def motor_from_log(v) -> Motor:
    """Reconstruct a Motor from its 6-vector bivector log (``Motor.exp``)."""
    v = np.asarray(v, dtype=float).reshape(6)
    return Motor.exp(*(float(x) for x in v))


def motor_split(m) -> tuple[np.ndarray, np.ndarray]:
    """Decompose a Motor into ``(translation 3-vec, rotor-bivector log 3-vec)``."""
    m = as_motor(m)
    t = m.get_translator()
    return (
        np.array([t.x(), t.y(), t.z()]),
        np.asarray(m.get_rotor().log().to_array(), dtype=float),
    )


def motor_from_split(trans, biv) -> Motor:
    """Inverse of :func:`motor_split`: ``Translator(t) * Rotor(exp(biv))``."""
    trans = np.asarray(trans, dtype=float).reshape(3)
    biv = np.asarray(biv, dtype=float).reshape(3)
    rotor = Rotor.exp(float(biv[0]), float(biv[1]), float(biv[2]))
    return Motor.from_translation_rotor(float(trans[0]), float(trans[1]), float(trans[2]), rotor)
