# SPDX-License-Identifier: BSD-2-Clause
# Authors: Siddhartha Srinivasa and contributors to TSR

"""Structured grasp-mode provenance (contract clause 8, issue #66).

Every grasp a factory returns must declare its mode via a machine-readable
``GraspProvenance`` record — tests read that record, never parse ``name`` — and
the record must round-trip serialization.
"""

import unittest

import numpy as np
import pytest

from tsr import GraspProvenance, ParallelJawGripper, TSRTemplate


class TestGraspProvenanceRecord(unittest.TestCase):
    def test_roundtrips_through_template_serialization(self):
        p = GraspProvenance(
            primitive="torus",
            mode="side",
            approach="tube",
            opening_axis="y",
            depth_index=1,
            depth_count=3,
            depth=0.021,
            variant="flip0",
            params={"minor_index": 2, "minor_angle": 0.5},
        )
        t = TSRTemplate(
            T_ref_tsr=np.eye(4),
            Tw_e=np.eye(4),
            Bw=np.zeros((6, 2)),
            task="grasp",
            subject="gripper",
            reference="torus",
            provenance=p,
        )
        for revived in (
            TSRTemplate.from_dict(t.to_dict()),
            TSRTemplate.from_json(t.to_json()),
            TSRTemplate.from_yaml(t.to_yaml()),
        ):
            self.assertEqual(revived.provenance, p)

    def test_rejects_unknown_primitive_and_mode(self):
        with pytest.raises(ValueError):
            GraspProvenance("blob", "side", "o", "y", 0, 1, 0.1)
        with pytest.raises(ValueError):
            GraspProvenance("box", "wiggle", "o", "y", 0, 1, 0.1)

    def test_rejects_depth_index_out_of_range(self):
        with pytest.raises(ValueError):
            GraspProvenance("box", "top", "+z", "x", 3, 2, 0.1)

    def test_non_grasp_templates_have_no_provenance(self):
        t = TSRTemplate(
            T_ref_tsr=np.eye(4),
            Tw_e=np.eye(4),
            Bw=np.zeros((6, 2)),
            task="place",
            subject="mug",
            reference="table",
        )
        self.assertIsNone(t.provenance)


# (method, kwargs, expected primitive, expected modes)
_CASES = [
    ("grasp_cylinder_side", dict(cylinder_radius=0.03, cylinder_height=0.12), "cylinder", {"side"}),
    ("grasp_cylinder_top", dict(cylinder_radius=0.03, cylinder_height=0.12), "cylinder", {"top"}),
    ("grasp_cylinder_bottom", dict(cylinder_radius=0.03, cylinder_height=0.12), "cylinder", {"bottom"}),
    ("grasp_box_top", dict(box_x=0.05, box_y=0.06, box_z=0.07), "box", {"top"}),
    ("grasp_box_bottom", dict(box_x=0.05, box_y=0.06, box_z=0.07), "box", {"bottom"}),
    ("grasp_box_face_x", dict(box_x=0.05, box_y=0.06, box_z=0.07), "box", {"face"}),
    ("grasp_box_face_y", dict(box_x=0.05, box_y=0.06, box_z=0.07), "box", {"face"}),
    ("grasp_sphere", dict(object_radius=0.03), "sphere", {"equatorial"}),
    ("grasp_torus_side", dict(torus_radius=0.06, tube_radius=0.02, n_minor=3), "torus", {"side"}),
    ("grasp_torus_span", dict(torus_radius=0.06, tube_radius=0.02), "torus", {"span"}),
]


class TestFactoriesEmitProvenance(unittest.TestCase):
    def setUp(self):
        # Aperture wide enough that every case (incl. torus span 2(R+r)) is feasible.
        self.gripper = ParallelJawGripper(finger_length=0.08, max_aperture=0.30)

    def test_every_factory_emits_consistent_provenance(self):
        for method, kwargs, primitive, modes in _CASES:
            templates = getattr(self.gripper, method)(**kwargs)
            self.assertTrue(templates, f"{method} unexpectedly returned []")
            seen_modes = set()
            for t in templates:
                p = t.provenance
                self.assertIsNotNone(p, f"{method}: template without provenance")
                self.assertEqual(p.primitive, primitive, method)
                self.assertIn(p.opening_axis, ("x", "y", "z"), method)
                # depth index is a valid position within the declared count
                self.assertTrue(0 <= p.depth_index < p.depth_count, f"{method}: {p}")
                seen_modes.add(p.mode)
            self.assertEqual(seen_modes, modes, f"{method}: modes {seen_modes} != {modes}")

    def test_depth_indices_cover_the_declared_count(self):
        # A single-orientation mode should enumerate depth_index 0..k-1 exactly.
        templates = self.gripper.grasp_cylinder_top(0.03, 0.12, k=4)
        idx = sorted(t.provenance.depth_index for t in templates)
        self.assertEqual(idx, [0, 1, 2, 3])
        self.assertTrue(all(t.provenance.depth_count == 4 for t in templates))


if __name__ == "__main__":
    unittest.main()
