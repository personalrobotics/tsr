#!/usr/bin/env python
# SPDX-License-Identifier: BSD-2-Clause
# Authors: Siddhartha Srinivasa and contributors to TSR

"""The assurance boundary for packaged YAML templates (issue #75).

Packaged templates are hand-authored illustrative recipes, **not** analytically
certified primitive grasps: they specify relative pose constraints and deliberately
omit the primitive dimensions and gripper geometry the #67 oracle needs, so no
geometric-soundness claim is made for them. Their contract is representation
validity — every file loads through the public API, instantiates, and satisfies
``sample() ⇔ contains() ⇔ distance ≈ 0`` — and the user remains responsible for
geometric feasibility in the intended scene.

The suite is dynamic: files are discovered through ``list_available_templates``, so
adding or removing one needs no manifest update here.
"""

import unittest
from pathlib import Path

import numpy as np

from tsr import (
    TSRTemplate,
    get_package_templates,
    list_available_templates,
    load_package_template,
    load_package_templates_by_category,
)

POSES = (
    np.eye(4),
    np.array([[1.0, 0, 0, 0.4], [0, 1.0, 0, -0.2], [0, 0, 1.0, 0.75], [0, 0, 0, 1.0]]),
)


def _packaged():
    return [(Path(rel).parent.as_posix(), Path(rel).name) for rel in list_available_templates()]


class TestDiscovery(unittest.TestCase):
    def test_discovery_matches_the_files_on_disk(self):
        # Dynamic: no manifest to update when a template is added or removed.
        on_disk = sorted(
            p.relative_to(get_package_templates()).as_posix() for p in get_package_templates().rglob("*.yaml")
        )
        self.assertEqual(list_available_templates(), on_disk)
        self.assertTrue(on_disk, "no packaged templates were discovered")

    def test_every_category_loads_as_a_group(self):
        for category in sorted({category for category, _ in _packaged()}):
            with self.subTest(category=category):
                templates = load_package_templates_by_category(category)
                self.assertEqual(len(templates), sum(1 for c, _ in _packaged() if c == category))
                self.assertTrue(all(isinstance(t, TSRTemplate) for t in templates))


class TestRepresentationValidity(unittest.TestCase):
    """Every packaged file is a loadable, instantiable, self-consistent template."""

    def test_each_template_loads_instantiates_and_round_trips(self):
        for category, name in _packaged():
            with self.subTest(template=f"{category}/{name}"):
                t = load_package_template(category, name)
                self.assertIsInstance(t, TSRTemplate)
                self.assertTrue(t.task and t.subject and t.reference)
                for T in (t.T_ref_tsr, t.Tw_e):
                    self.assertTrue(np.all(np.isfinite(T)))
                    R = T[:3, :3]
                    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-9)
                    self.assertAlmostEqual(float(np.linalg.det(R)), 1.0, places=9)
                self.assertTrue(np.all(np.isfinite(t.Bw)))
                self.assertTrue(np.all(t.Bw[:, 0] <= t.Bw[:, 1]))
                self.assertEqual(TSRTemplate.from_dict(t.to_dict()).to_dict(), t.to_dict())

    def test_sampling_is_consistent_with_containment_at_several_poses(self):
        rng = np.random.default_rng(0)
        for category, name in _packaged():
            t = load_package_template(category, name)
            for T_ref_world in POSES:
                tsr = t.instantiate(T_ref_world)
                for _ in range(5):
                    pose = tsr.sample(rng=rng)
                    with self.subTest(template=f"{category}/{name}"):
                        self.assertTrue(np.all(np.isfinite(pose)))
                        self.assertTrue(tsr.contains(pose))
                        self.assertAlmostEqual(tsr.distance(pose)[0], 0.0, places=6)


class TestAssuranceBoundaryIsDocumented(unittest.TestCase):
    """The READMEs state what packaged templates do and do not guarantee."""

    ROOT = Path(__file__).resolve().parents[2]

    def test_packaged_readme_states_the_boundary_and_uses_real_signatures(self):
        readme = (get_package_templates() / "README.md").read_text()
        self.assertIn("representation", readme.lower())
        # The loader takes (category, name), and the grasp category directory is "grasps".
        self.assertIn('load_package_template("grasps", "screwdriver_grasp.yaml")', readme)
        self.assertIn('load_package_templates_by_category("grasps")', readme)
        self.assertNotIn('load_package_template("grasps/', readme)

    def test_public_readme_states_the_boundary(self):
        self.assertIn("illustrative", (self.ROOT / "README.md").read_text().lower())


if __name__ == "__main__":
    unittest.main()
