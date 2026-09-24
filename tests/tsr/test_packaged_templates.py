#!/usr/bin/env python
# SPDX-License-Identifier: BSD-2-Clause
# Authors: Siddhartha Srinivasa and contributors to TSR

"""Representation-level smoke test for the packaged YAML examples (issue #75)."""

from collections import Counter
from pathlib import Path

import numpy as np

from tsr import list_available_templates, load_package_template, load_package_templates_by_category


def _assert_se3(T: np.ndarray) -> None:
    assert T.shape == (4, 4)
    assert np.all(np.isfinite(T))
    np.testing.assert_allclose(T[3], [0.0, 0.0, 0.0, 1.0], atol=0.0)
    R = T[:3, :3]
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)
    np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-10)


def test_every_packaged_template_loads_and_instantiates() -> None:
    """The package promises valid TSR recipes, not unrecorded grasp semantics."""
    available = list_available_templates()
    assert available, "the package template inventory is unexpectedly empty"

    category_counts: Counter[str] = Counter()
    for index, relative in enumerate(available):
        parts = Path(relative).parts
        assert len(parts) == 2, f"packaged template must be category/name, got {relative!r}"
        category, name = parts
        category_counts[category] += 1

        template = load_package_template(category, name)
        _assert_se3(template.T_ref_tsr)
        _assert_se3(template.Tw_e)
        assert template.Bw.shape == (6, 2)
        assert np.all(np.isfinite(template.Bw))
        assert np.all(template.Bw[:3, 0] <= template.Bw[:3, 1])
        if template.preshape is not None:
            assert np.all(np.isfinite(template.preshape))

        tsr = template.instantiate(np.eye(4))
        pose = tsr.sample(rng=np.random.default_rng(index))
        _assert_se3(pose)
        assert tsr.contains(pose)
        distance, _ = tsr.distance(pose)
        assert distance == 0.0

    for category, expected in category_counts.items():
        assert len(load_package_templates_by_category(category)) == expected
