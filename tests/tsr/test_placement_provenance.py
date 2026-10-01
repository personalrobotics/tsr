# SPDX-License-Identifier: BSD-2-Clause
# Authors: Siddhartha Srinivasa and contributors to TSR

"""Structured provenance for placement templates (issue #160).

Three layers, as on the grasp side (``tests/tsr/test_grasp_provenance.py``):

1. :class:`~tsr.placement_provenance.PlacementProvenance` validates *representation*
   only, so a third-party generator can use its own vocabulary;
2. ``tsr.placement._conformance`` owns the sstsr vocabulary and the relational checks;
3. the oracle derives truth from the pose and never trusts the record — which is why the
   geometric agreement checks live in ``tests/tsr/placement/_placement_matrix.py``
   (``provenance_failures``) rather than here.

The serialization tests deliberately go **through the filesystem**. A dict round-trip
passes while a record carrying a NumPy scalar dies on disk — ``yaml.dump`` writes a
``!!python/object/apply`` tag, ``safe_load`` refuses it, and
``load_templates_from_directory`` catches that and skips the file with a warning. So the
only test that can see that class of bug is one that saves and loads.
"""

from __future__ import annotations

import math
import unittest

import numpy as np
import pytest
import yaml

from tsr import GraspProvenance, PlacementProvenance, ProvenanceKindError, TSRTemplate, load_template, save_template
from tsr.io import load_templates_from_directory
from tsr.placement import StablePlacer
from tsr.placement._conformance import validate_builtin_placement_template, validate_builtin_provenance

TABLE = dict(table_x=0.60, table_y=0.60)


def _record(**overrides) -> PlacementProvenance:
    """A representation-valid record, so each negative test below is a one-key delta."""
    fields = dict(
        primitive="box",
        support_margin=0.05,
        com_height=0.15,
        footprint_radius=0.1118,
        equilibrium="stable",
        face_normal=(0.0, 0.0, -1.0),
    )
    fields.update(overrides)
    return PlacementProvenance(**fields)


def _grasp_record() -> GraspProvenance:
    return GraspProvenance(
        primitive="box", mode="top", approach="+z", finger_orientation="x", depth=0.01, depth_index=0, depth_count=1
    )


def _template(provenance=None, **overrides) -> TSRTemplate:
    fields = dict(
        T_ref_tsr=np.eye(4),
        Tw_e=np.eye(4),
        Bw=np.zeros((6, 2)),
        task="place",
        subject="box",
        reference="table",
        provenance=provenance,
    )
    fields.update(overrides)
    return TSRTemplate(**fields)


def _mesh_cube(half: float = 0.05) -> np.ndarray:
    return np.array([[sx * half, sy * half, sz * half] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])


# --------------------------------------------------------------------------- #
# Layer 1 — representation only
# --------------------------------------------------------------------------- #


class TestPlacementProvenanceValueObject(unittest.TestCase):
    def test_roundtrips_losslessly(self):
        record = _record(face_index=2, face_count=6, facet_count=4, metadata={"asset": "mug", "scale": 1.0})
        template = _template(record)
        for revived in (
            TSRTemplate.from_dict(template.to_dict()),
            TSRTemplate.from_json(template.to_json()),
            TSRTemplate.from_yaml(template.to_yaml()),
        ):
            self.assertEqual(revived.provenance, record)

    def test_accepts_arbitrary_external_labels(self):
        """The layer boundary: a third-party generator's vocabulary is representable, and
        native conformance is what rejects it. If this collapses into one closed
        vocabulary, the public type stops being usable outside sstsr."""
        foreign = _record(primitive="wedge", equilibrium="metastable", face_normal=(0.0, 0.0, -3.0))
        self.assertEqual(foreign.primitive, "wedge")
        with pytest.raises(ValueError):
            validate_builtin_provenance(foreign)

    def test_numpy_scalars_are_canonicalised(self):
        """Not cosmetic: ``np.float64`` passes ``isinstance(x, float)`` and then
        serializes to a tag ``safe_load`` refuses."""
        record = _record(support_margin=np.float64(0.05), face_normal=np.array([0.0, 0.0, -1.0]))
        self.assertIs(type(record.support_margin), float)
        self.assertIs(type(record.face_normal[0]), float)
        self.assertNotIn("python/object", yaml.dump(record.to_dict()))

    def test_metadata_is_frozen_and_copied(self):
        source = {"asset": "mug"}
        record = _record(metadata=source)
        source["asset"] = "bowl"
        self.assertEqual(record.metadata["asset"], "mug")
        with pytest.raises(TypeError):
            record.metadata["asset"] = "bowl"

    def test_a_face_normal_must_be_a_direction(self):
        with pytest.raises(ValueError):
            _record(face_normal=(0.0, 0.0, 0.0))
        with pytest.raises(ValueError):
            _record(face_normal=(0.0, 0.0))
        with pytest.raises(ValueError):
            _record(face_normal=(0.0, 0.0, math.nan))

    def test_lengths_must_be_finite_and_non_negative(self):
        for name in ("support_margin", "com_height", "footprint_radius"):
            for bad in (-1e-9, math.nan, math.inf, True):
                with pytest.raises(ValueError):
                    _record(**{name: bad})

    def test_labels_must_be_non_empty_strings(self):
        for name in ("primitive", "equilibrium"):
            for bad in ("", None, 3):
                with pytest.raises(ValueError):
                    _record(**{name: bad})

    def test_the_mesh_counters_are_present_together(self):
        with pytest.raises(ValueError):
            _record(face_index=0)  # without face_count/facet_count
        with pytest.raises(ValueError):
            _record(face_index=6, face_count=6, facet_count=2)  # out of range
        with pytest.raises(ValueError):
            _record(face_index=True, face_count=6, facet_count=2)  # bool is not an index
        with pytest.raises(ValueError):
            _record(face_index=0, face_count=6, facet_count=0)


# --------------------------------------------------------------------------- #
# The wire discriminator
# --------------------------------------------------------------------------- #


class TestWireDiscriminator(unittest.TestCase):
    def test_a_legacy_grasp_block_without_a_kind_still_loads(self):
        """Back-compat for every record sstsr wrote before #160."""
        block = _grasp_record().to_dict()
        del block["kind"]
        revived = TSRTemplate.from_dict({**_template().to_dict(), "provenance": block})
        self.assertEqual(revived.provenance, _grasp_record())

    def test_the_two_record_types_reject_each_others_dicts(self):
        """The only route to a silent misread is a ``from_dict`` that defaults a missing
        key, so both directions must refuse rather than narrow."""
        with pytest.raises(ValueError):
            PlacementProvenance.from_dict(_grasp_record().to_dict())
        with pytest.raises(ValueError):
            GraspProvenance.from_dict(_record().to_dict())

    def test_an_unknown_kind_is_not_silently_skipped(self):
        """A record from a newer version must fail loudly. ``ProvenanceKindError`` is
        deliberately outside the ``(YAMLError, KeyError, ValueError, TypeError)`` tuple
        ``load_templates_from_directory`` swallows, or a version-skewed directory would
        come back empty."""
        with pytest.raises(ProvenanceKindError):
            TSRTemplate.from_dict({**_template().to_dict(), "provenance": {"kind": "from-the-future"}})
        assert not isinstance(ProvenanceKindError("x"), (ValueError, KeyError, TypeError))


def test_an_unknown_kind_propagates_out_of_a_directory_load(tmp_path):
    document = _template(_record()).to_dict()
    document["provenance"]["kind"] = "from-the-future"
    (tmp_path / "future.yaml").write_text(yaml.dump(document))
    with pytest.raises(ProvenanceKindError):
        load_templates_from_directory(tmp_path)
    with pytest.raises(ProvenanceKindError):
        load_template(tmp_path / "future.yaml")


def test_a_saved_placement_template_survives_a_directory_load(tmp_path):
    """The test that catches the whole ``io``-swallow class: a missing ``kind``, a leaked
    NumPy scalar, a dropped key. A dict round-trip passes while this one fails."""
    placer = StablePlacer(**TABLE)
    vertices = _mesh_cube()
    emitted = (
        placer.place_cylinder(0.04, 0.12)
        + placer.place_box(0.20, 0.10, 0.30)
        + placer.place_sphere(0.05)
        + placer.place_torus(0.05, 0.012)
        + placer.place_mesh(vertices, vertices.mean(axis=0))
    )
    for index, template in enumerate(emitted):
        save_template(template, tmp_path / f"{index:02d}.yaml")

    loaded = load_templates_from_directory(tmp_path)
    assert len(loaded) == len(emitted), "a template was skipped on load"
    for original, revived in zip(emitted, loaded):
        assert revived.provenance == original.provenance
        validate_builtin_placement_template(revived)


# --------------------------------------------------------------------------- #
# Layer 2 — native conformance
# --------------------------------------------------------------------------- #

#: (factory, kwargs, primitive, expected face normals, expected equilibria). The normal
#: and equilibrium sets are asserted to be **equal** to what the factory emits, not a
#: superset: a subset assertion would pass a factory that emitted nothing.
_CASES = [
    (
        "place_cylinder",
        dict(cylinder_radius=0.04, cylinder_height=0.12),
        "cylinder",
        {(0, 0, -1), (0, 0, 1)},
        {"stable"},
    ),
    (
        "place_box",
        dict(lx=0.20, ly=0.10, lz=0.30),
        "box",
        {(0, 0, -1), (0, 0, 1), (0, -1, 0), (0, 1, 0), (-1, 0, 0), (1, 0, 0)},
        {"stable"},
    ),
    ("place_sphere", dict(radius=0.05), "sphere", set(), {"neutral"}),
    ("place_torus", dict(major_radius=0.05, minor_radius=0.012), "torus", {(0, 0, -1), (0, 0, 1)}, {"stable"}),
]


class TestNativeConformance(unittest.TestCase):
    def test_every_builtin_primitive_template_is_conformant(self):
        placer = StablePlacer(**TABLE)
        for factory, kwargs, primitive, normals, equilibria in _CASES:
            templates = getattr(placer, factory)(**kwargs)
            self.assertTrue(templates, factory)
            for t in templates:
                validate_builtin_placement_template(t)
                self.assertEqual(t.provenance.primitive, primitive)
            emitted_normals = {
                tuple(int(round(c)) for c in t.provenance.face_normal)
                for t in templates
                if t.provenance.face_normal is not None
            }
            self.assertEqual(emitted_normals, normals, factory)
            self.assertEqual({t.provenance.equilibrium for t in templates}, equilibria, factory)

    def test_every_builtin_mesh_template_is_conformant(self):
        placer = StablePlacer(**TABLE)
        vertices = _mesh_cube()
        templates = placer.place_mesh(vertices, vertices.mean(axis=0))
        self.assertEqual(len(templates), 6)
        for t in templates:
            validate_builtin_placement_template(t)
        self.assertEqual({t.provenance.face_count for t in templates}, {6})
        self.assertEqual({t.provenance.face_index for t in templates}, set(range(6)))

    def test_relational_mistakes_are_caught(self):
        import dataclasses

        base = _record()
        for broken, why in (
            (dataclasses.replace(base, primitive="sphere"), "a sphere with a face normal"),
            (dataclasses.replace(base, face_normal=None), "a box without a face normal"),
            (dataclasses.replace(base, face_normal=(0.0, 0.0, -2.0)), "a non-unit normal"),
            (dataclasses.replace(base, equilibrium="neutral"), "neutral with a positive support margin"),
            (dataclasses.replace(base, support_margin=0.0), "a zero support margin called stable"),
            (dataclasses.replace(base, face_index=0, face_count=6, facet_count=2), "mesh counters on a box"),
            (dataclasses.replace(base, com_height=0.0), "a zero lever arm"),
            (dataclasses.replace(base, primitive="mesh"), "a mesh without face counters"),
        ):
            with self.subTest(why), pytest.raises(ValueError):
                validate_builtin_provenance(broken)

    def test_a_reported_angle_that_contradicts_its_components_is_caught(self):
        """The identity ``stability_margin == atan2(support_margin, com_height)`` is what
        makes the headline number auditable, so a template whose angle disagrees with its
        own record must be rejected."""
        record = _record()
        honest = math.atan2(record.support_margin, record.com_height)
        validate_builtin_placement_template(_template(record, variant="-z", stability_margin=honest))
        with pytest.raises(ValueError):
            validate_builtin_placement_template(_template(record, variant="-z", stability_margin=honest * 1.01))

    def test_a_variant_that_contradicts_the_recorded_face_is_caught(self):
        record = _record(face_normal=(0.0, 0.0, -1.0))
        honest = math.atan2(record.support_margin, record.com_height)
        with pytest.raises(ValueError):
            validate_builtin_placement_template(_template(record, variant="+x", stability_margin=honest))

    def test_mesh_facet_counts_account_for_the_hull(self):
        """The #152 merge count must be real. The expectation comes from scipy, not from
        our own grouping code: a cube's every face is stable, so the facets merged across
        all six faces are exactly the hull's triangles."""
        from scipy.spatial import ConvexHull

        placer = StablePlacer(**TABLE)
        for vertices in (_mesh_cube(), _mesh_cube(0.03)):
            templates = placer.place_mesh(vertices, vertices.mean(axis=0))
            self.assertEqual(len(templates), 6)
            self.assertEqual(sum(t.provenance.facet_count for t in templates), len(ConvexHull(vertices).simplices))

    def test_a_tessellated_side_merges_exactly_two_facets(self):
        """Each flat side of an n-gon prism is one quad, i.e. two triangles — so a
        hardcoded or inflated facet count shows up here."""
        placer = StablePlacer(**TABLE)
        n = 24
        angles = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
        rim = np.stack([0.04 * np.cos(angles), 0.04 * np.sin(angles)], axis=-1)
        vertices = np.vstack([np.c_[rim, np.full(n, z)] for z in (-0.06, 0.06)])
        sides = [t for t in placer.place_mesh(vertices, np.zeros(3)) if abs(t.Tw_e[2, 2]) < 0.9]
        self.assertEqual(len(sides), n)
        self.assertEqual({t.provenance.facet_count for t in sides}, {2})


if __name__ == "__main__":
    unittest.main()
