# SPDX-License-Identifier: BSD-2-Clause
# Authors: Siddhartha Srinivasa and contributors to TSR

"""Grasp provenance: the three-layer model (issues #66, #78-#84).

* ``GraspProvenance`` is a generic, extensible value object that validates only
  representation invariants (this file's first class).
* ``tsr.hands._conformance.validate_builtin_provenance`` owns the sstsr native
  vocabulary and cross-field relational checks (second class).
* Geometric truth is the oracle's job (#67), not provenance's — the oracle
  derives quantities from the pose and must not trust these recorded values.
"""

import unittest

import numpy as np
import pytest

from tsr import GraspProvenance, ParallelJawGripper, TSRTemplate
from tsr.hands._conformance import validate_builtin_provenance


def _record(**overrides):
    """A representation-valid record (built-in torus-side shape by default)."""
    base = dict(
        primitive="torus",
        mode="side",
        approach="tube",
        finger_orientation="tangential",
        depth=0.021,
        depth_index=1,
        depth_count=3,
        symmetry="flip0",
        metadata={"minor_index": 2, "minor_count": 3, "minor_angle": 0.5},
    )
    base.update(overrides)
    return GraspProvenance(**base)


class TestGraspProvenanceValueObject(unittest.TestCase):
    """Layer 1: generic representation invariants only — no closed vocabulary."""

    def test_roundtrips_losslessly(self):
        p = _record()
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

    def test_accepts_arbitrary_external_labels(self):
        # The value object is extensible: a third-party generator's vocabulary is
        # accepted. Only native conformance restricts to sstsr's built-ins.
        p = _record(primitive="cone", mode="wrap", approach="lateral", finger_orientation="helix", metadata={})
        self.assertEqual((p.primitive, p.mode), ("cone", "wrap"))
        # ...and native conformance does reject it.
        with pytest.raises(ValueError):
            validate_builtin_provenance(p)

    def test_rejects_empty_or_non_string_labels(self):
        for name in ("primitive", "mode", "approach", "finger_orientation"):
            with pytest.raises(ValueError):
                _record(**{name: ""})
            with pytest.raises(ValueError):
                _record(**{name: 5})
        with pytest.raises(ValueError):
            _record(symmetry=5)

    def test_depth_counter_and_depth_invariants(self):
        with pytest.raises(ValueError):
            _record(depth_index=3, depth_count=2)
        with pytest.raises(ValueError):
            _record(depth_index=True)
        with pytest.raises(ValueError):
            _record(depth=-0.01)
        with pytest.raises(ValueError):
            _record(depth=float("inf"))
        with pytest.raises(ValueError):
            _record(depth="0.1")  # not a real number
        with pytest.raises(ValueError):
            _record(depth=True)
        self.assertIsInstance(_record(depth=0).depth, float)  # int canonicalized

    def test_metadata_is_immutable_defensively_copied_and_scalar_only(self):
        d = {"minor_index": 0, "minor_count": 3, "minor_angle": 0.5}
        p = _record(metadata=d)
        d["minor_index"] = 99
        self.assertEqual(p.metadata["minor_index"], 0)
        with self.assertRaises(TypeError):
            p.metadata["minor_index"] = 1
        with pytest.raises(ValueError):
            _record(metadata={"bad": [1, 2]})
        with pytest.raises(ValueError):
            _record(metadata={"bad": float("nan")})

    def test_from_dict_does_not_silently_narrow(self):
        d = _record().to_dict()
        d["depth_index"] = 0.5
        with pytest.raises(ValueError):
            GraspProvenance.from_dict(d)

    def test_value_object_does_not_enforce_native_pairs(self):
        # (#84) the generic object does NOT know which (primitive, mode) exist.
        self.assertEqual(_record(primitive="box", mode="surface", metadata={}).mode, "surface")


# (method, kwargs, primitive, modes, finger_orientations)
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


class TestNativeConformance(unittest.TestCase):
    """Layer 2: sstsr-specific vocabulary + relational checks."""

    def setUp(self):
        self.gripper = ParallelJawGripper(finger_length=0.08, max_aperture=0.30)

    def test_every_builtin_template_is_conformant_and_roundtrips(self):
        for method, kwargs, primitive, modes, orientations in _CASES:
            templates = getattr(self.gripper, method)(**kwargs)
            self.assertTrue(templates, f"{method} unexpectedly returned []")
            seen_modes, seen_or = set(), set()
            for t in templates:
                validate_builtin_provenance(t.provenance)  # must not raise
                self.assertEqual(GraspProvenance.from_dict(t.provenance.to_dict()), t.provenance)
                self.assertEqual(t.provenance.primitive, primitive)
                self.assertGreaterEqual(t.provenance.depth, 0.0)
                seen_modes.add(t.provenance.mode)
                seen_or.add(t.provenance.finger_orientation)
            self.assertEqual(seen_modes, modes, method)
            self.assertEqual(seen_or, orientations, method)

    def test_relational_mistakes_are_caught(self):
        # box top: finger orientation must differ from the slide axis
        with pytest.raises(ValueError):
            validate_builtin_provenance(
                _record(
                    primitive="box",
                    mode="top",
                    approach="+z",
                    finger_orientation="x",
                    symmetry="",
                    metadata={"slide_axis": "x", "span": 0.05},
                )
            )
        # box face: finger orientation must lie in the face plane (not the normal)
        with pytest.raises(ValueError):
            validate_builtin_provenance(
                _record(
                    primitive="box",
                    mode="face",
                    approach="+x",
                    finger_orientation="x",
                    symmetry="",
                    metadata={"slide_axis": "y", "span": 0.05},
                )
            )
        # torus side: minor_index must be in range
        with pytest.raises(ValueError):
            validate_builtin_provenance(_record(metadata={"minor_index": 3, "minor_count": 3, "minor_angle": 0.0}))
        # missing required metadata
        with pytest.raises(ValueError):
            validate_builtin_provenance(_record(metadata={}))
        # non-built-in pair
        with pytest.raises(ValueError):
            validate_builtin_provenance(_record(primitive="box", mode="surface", finger_orientation="x", metadata={}))

    def test_box_approach_is_hand_occupied_side(self):
        self.assertEqual({t.provenance.approach for t in self.gripper.grasp_box_top(0.05, 0.06, 0.07)}, {"+z"})
        self.assertEqual({t.provenance.approach for t in self.gripper.grasp_box_face_x(0.05, 0.06, 0.07)}, {"+x", "-x"})

    def test_depth_count_is_actual_emitted_slots(self):
        thin = ParallelJawGripper(finger_length=0.04, max_aperture=0.30)
        collapsed = thin.grasp_cylinder_side(0.03, 0.12, k=5, clearance=0.01)
        self.assertTrue(all(t.provenance.depth_count == 1 for t in collapsed))
        # equality-boundary: 3 slots that coincide in depth (dedup deferred to #68)
        boundary = self.gripper.grasp_cylinder_top(0.03, 0.12, k=3, clearance=0.04)
        self.assertEqual(len(boundary), 3)
        self.assertTrue(all(t.provenance.depth_count == 3 for t in boundary))
        self.assertEqual(len({round(t.provenance.depth, 9) for t in boundary}), 1)

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


if __name__ == "__main__":
    unittest.main()
