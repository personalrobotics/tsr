# SPDX-License-Identifier: BSD-2-Clause
# Authors: Siddhartha Srinivasa and contributors to TSR

"""Structured grasp-mode provenance (contract clause 8; issues #66, #78-#81).

Every grasp a factory returns declares its mode via a machine-readable
``GraspProvenance`` record — tests read that record, never parse ``name``. These
tests check the *semantics and invariants* of the record (one frame/meaning per
field, closed vocabulary, immutability, lossless serialization), not merely that
strings belong to a broad set.
"""

import unittest

import numpy as np
import pytest

from tsr import GraspProvenance, ParallelJawGripper, TSRTemplate
from tsr.grasp_provenance import SPAN_AXES


class TestGraspProvenanceRecord(unittest.TestCase):
    def _valid(self, **overrides):
        base = dict(
            primitive="torus",
            mode="side",
            approach="tube",
            span_axis="tangential",
            depth_index=1,
            depth_count=3,
            depth=0.021,
            variant="flip0",
            params={"minor_index": 2, "minor_angle": 0.5},
        )
        base.update(overrides)
        return GraspProvenance(**base)

    def test_roundtrips_through_template_serialization(self):
        p = self._valid()
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

    def test_rejects_invalid_primitive_mode_pairs(self):
        # (#80) validated as pairs, not independent strings.
        with pytest.raises(ValueError):
            self._valid(primitive="box", mode="surface")
        with pytest.raises(ValueError):
            self._valid(primitive="sphere", mode="side")
        with pytest.raises(ValueError):
            self._valid(primitive="blob", mode="side")

    def test_rejects_malformed_scalar_fields(self):
        with pytest.raises(ValueError):
            self._valid(span_axis="w")  # not in SPAN_AXES
        with pytest.raises(ValueError):
            self._valid(approach="")  # empty
        with pytest.raises(ValueError):
            self._valid(depth_index=3, depth_count=2)  # out of range
        with pytest.raises(ValueError):
            self._valid(depth_index=True)  # bool is not an int here
        with pytest.raises(ValueError):
            self._valid(depth=-0.01)  # negative
        with pytest.raises(ValueError):
            self._valid(depth=float("inf"))  # non-finite

    def test_params_are_immutable_and_defensively_copied(self):
        d = {"minor_index": 0}
        p = self._valid(params=d)
        d["minor_index"] = 99  # mutate the caller's dict
        self.assertEqual(p.params["minor_index"], 0)  # record unaffected
        with self.assertRaises(TypeError):
            p.params["minor_index"] = 1  # record itself is read-only

    def test_params_reject_non_scalar_and_non_finite(self):
        with pytest.raises(ValueError):
            self._valid(params={"bad": [1, 2]})
        with pytest.raises(ValueError):
            self._valid(params={"bad": float("nan")})

    def test_from_dict_does_not_silently_narrow(self):
        d = self._valid().to_dict()
        d["depth_index"] = 0.5  # non-integral
        with pytest.raises(ValueError):
            GraspProvenance.from_dict(d)

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


# (method, kwargs, primitive, expected modes, expected span_axes)
_CASES = [
    ("grasp_cylinder_side", dict(cylinder_radius=0.03, cylinder_height=0.12), "cylinder", {"side"}, {"tangential"}),
    ("grasp_cylinder_top", dict(cylinder_radius=0.03, cylinder_height=0.12), "cylinder", {"top"}, {"diameter"}),
    ("grasp_cylinder_bottom", dict(cylinder_radius=0.03, cylinder_height=0.12), "cylinder", {"bottom"}, {"diameter"}),
    ("grasp_box_top", dict(box_x=0.05, box_y=0.06, box_z=0.07), "box", {"top"}, {"x", "y"}),
    ("grasp_box_bottom", dict(box_x=0.05, box_y=0.06, box_z=0.07), "box", {"bottom"}, {"x", "y"}),
    ("grasp_box_face_x", dict(box_x=0.05, box_y=0.06, box_z=0.07), "box", {"face"}, {"y", "z"}),
    ("grasp_box_face_y", dict(box_x=0.05, box_y=0.06, box_z=0.07), "box", {"face"}, {"x", "z"}),
    ("grasp_sphere", dict(object_radius=0.03), "sphere", {"surface"}, {"diameter"}),
    ("grasp_torus_side", dict(torus_radius=0.06, tube_radius=0.02, n_minor=3), "torus", {"side"}, {"tangential"}),
    ("grasp_torus_span", dict(torus_radius=0.06, tube_radius=0.02), "torus", {"span"}, {"diameter"}),
]


class TestFactoriesEmitProvenance(unittest.TestCase):
    def setUp(self):
        self.gripper = ParallelJawGripper(finger_length=0.08, max_aperture=0.30)

    def test_every_factory_emits_semantically_correct_provenance(self):
        for method, kwargs, primitive, modes, span_axes in _CASES:
            templates = getattr(self.gripper, method)(**kwargs)
            self.assertTrue(templates, f"{method} unexpectedly returned []")
            seen_modes, seen_spans = set(), set()
            for t in templates:
                p = t.provenance
                self.assertIsNotNone(p, f"{method}: template without provenance")
                self.assertEqual(p.primitive, primitive, method)
                self.assertIn(p.span_axis, SPAN_AXES, method)
                self.assertGreaterEqual(p.depth, 0.0, f"{method}: negative insertion depth")
                self.assertTrue(0 <= p.depth_index < p.depth_count, f"{method}: {p}")
                seen_modes.add(p.mode)
                seen_spans.add(p.span_axis)
            self.assertEqual(seen_modes, modes, f"{method}: modes {seen_modes} != {modes}")
            self.assertEqual(seen_spans, span_axes, f"{method}: span_axes {seen_spans} != {span_axes}")

    def test_box_approach_is_hand_occupied_side(self):
        # box top -> hand above (+z); face_x -> hand on +x/-x.
        self.assertEqual({t.provenance.approach for t in self.gripper.grasp_box_top(0.05, 0.06, 0.07)}, {"+z"})
        self.assertEqual({t.provenance.approach for t in self.gripper.grasp_box_face_x(0.05, 0.06, 0.07)}, {"+x", "-x"})

    def test_depth_count_is_actual_emitted_not_requested_k(self):
        # (#81) A thin gripper collapses the usable band to one depth even for k=5;
        # depth_count reports the actual emitted count (1), not the requested k.
        thin = ParallelJawGripper(finger_length=0.04, max_aperture=0.30)
        templates = thin.grasp_cylinder_side(0.03, 0.12, k=5, clearance=0.01)
        self.assertTrue(templates)
        self.assertTrue(
            all(t.provenance.depth_count == 1 for t in templates), "collapsed band should report depth_count=1"
        )
        self.assertTrue(all(t.provenance.depth_index == 0 for t in templates))

    def test_depth_indices_cover_the_declared_count_when_not_collapsed(self):
        templates = self.gripper.grasp_cylinder_top(0.03, 0.12, k=4)
        self.assertEqual(sorted(t.provenance.depth_index for t in templates), [0, 1, 2, 3])
        self.assertTrue(all(t.provenance.depth_count == 4 for t in templates))


if __name__ == "__main__":
    unittest.main()
