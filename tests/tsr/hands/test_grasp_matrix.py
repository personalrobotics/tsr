#!/usr/bin/env python
# SPDX-License-Identifier: BSD-2-Clause
# Authors: Siddhartha Srinivasa and contributors to TSR

"""Hypothesis property matrix over every primitive grasp factory (issue #73).

Every public ``grasp_*`` factory -- including the combined entry points and the
named grippers -- is driven through the independent #67 analytic oracle. Example
budgets scale with ``TSR_MATRIX_SCALE`` (default 1; the release gate uses 10); see
``docs/ARCHITECTURE.md`` ("Verification matrix").
"""

import unittest

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from .._hypothesis_strategies import transforms
from ._grasp_matrix import (
    FACTORIES,
    any_cases,
    apply_override,
    combined_failures,
    coverage_failures,
    equivariance_failures,
    grasp_cases,
    invalid_overrides_for,
    matrix_settings,
    mirror_family_failures,
    monotonicity_failures,
    oversize_dims,
    public_grasp_factories,
    serialization_failures,
    soundness_failures,
    symmetry_failures,
    uniqueness_failures,
)


class TestFactoryTable(unittest.TestCase):
    def test_every_public_grasp_factory_is_in_the_matrix(self):
        self.assertEqual(set(FACTORIES), public_grasp_factories())

    def test_combined_entry_points_list_existing_parts(self):
        for spec in FACTORIES.values():
            for part in spec.parts:
                self.assertEqual(FACTORIES[part].primitive, spec.primitive)


class TestOracleWitness(unittest.TestCase):
    """(1) Every returned pose has an analytic oracle witness."""

    @matrix_settings(300)
    @given(case=any_cases(), seed=st.integers(0, 2**32 - 1))
    def test_every_pose_has_a_witness(self, case, seed):
        templates = case.run()
        self.assertEqual(soundness_failures(case, templates, np.random.default_rng(seed)), [], case)


class TestEquivariance(unittest.TestCase):
    """(3) Templates are equivariant under an arbitrary reference transform."""

    @matrix_settings(150)
    @given(case=any_cases(), T=transforms())
    def test_reference_transform_maps_poses_and_keeps_the_witness(self, case, T):
        self.assertEqual(equivariance_failures(case, case.run(), T), [], case)


class TestSymmetry(unittest.TestCase):
    """(4) Primitive and opposite-face symmetries."""

    @matrix_settings(150)
    @given(case=any_cases(), theta=st.floats(-np.pi, np.pi))
    def test_symmetries_preserve_the_witness(self, case, theta):
        self.assertEqual(symmetry_failures(case, case.run(), theta), [], case)

    @matrix_settings(150)
    @given(case=any_cases())
    def test_mirror_families_have_equal_depths(self, case):
        self.assertEqual(mirror_family_failures(case, case.run()), [], case)


class TestSerialization(unittest.TestCase):
    """(7) Serialization and instantiation preserve the witness."""

    @matrix_settings(100)
    @given(case=any_cases())
    def test_round_trips_preserve_poses_and_witness(self, case):
        self.assertEqual(serialization_failures(case, case.run()), [], case)


class TestArguments(unittest.TestCase):
    """(2) Invalid arguments raise; physical infeasibility returns []."""

    @matrix_settings(200)
    @given(case=any_cases(), which=st.integers(0, 99))
    def test_invalid_arguments_raise_value_error(self, case, which):
        overrides = invalid_overrides_for(case.spec)
        name = sorted(overrides)[which % len(overrides)]
        kwargs = apply_override(case, overrides[name])
        with self.assertRaises(ValueError, msg=f"{case.factory} {name} {kwargs}"):
            getattr(case.gripper.build(), case.factory)(**kwargs)

    @matrix_settings(150)
    @given(case=grasp_cases())
    def test_oversize_object_returns_empty_without_raising(self, case):
        # No jaw opening within the aperture can straddle these dimensions.
        oversize = case.replace(dims=oversize_dims(case.spec.primitive, case.gripper.max_aperture))
        self.assertEqual(oversize.run(preshape=None), [], oversize)


class TestCoverageAndUniqueness(unittest.TestCase):
    """(6) Declared coverage, structured uniqueness, combined == parts."""

    @matrix_settings(200)
    @given(case=any_cases())
    def test_structured_keys_are_unique(self, case):
        self.assertEqual(uniqueness_failures(case, case.run()), [], case)

    @matrix_settings(200)
    @given(case=any_cases())
    def test_combined_entry_points_equal_their_parts(self, case):
        self.assertEqual(combined_failures(case, case.run()), [], case)

    @matrix_settings(150)
    @given(case=grasp_cases(), k=st.integers(1, 4))
    def test_declared_families_are_all_emitted(self, case, k):
        # A comfortably feasible regime: a cube-ish object well inside reach and aperture.
        A, L = case.gripper.max_aperture, case.gripper.finger_length
        size = min(A / 3.0, L / 2.0)
        dims = {
            "sphere": {"object_radius": size / 2},
            "cylinder": {"cylinder_radius": size / 2, "cylinder_height": 2 * size},
            "box": {"box_x": size, "box_y": size, "box_z": size},
            "torus": {"torus_radius": size, "tube_radius": size / 4},
        }[case.spec.primitive]
        comfortable = case.replace(dims=dims, options={"k": k, "clearance": size / 50})
        self.assertEqual(coverage_failures(comfortable, comfortable.run()), [], comfortable)


class TestMonotonicity(unittest.TestCase):
    """(5) More reach or a wider aperture never removes a feasible family."""

    @matrix_settings(150)
    @given(
        case=grasp_cases(named=False, explicit_clearance=True), attr=st.sampled_from(["finger_length", "max_aperture"])
    )
    def test_growing_the_gripper_keeps_every_family(self, case, attr):
        self.assertEqual(monotonicity_failures(case, attr), [], case)


if __name__ == "__main__":
    unittest.main()
