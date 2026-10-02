#!/usr/bin/env python
# SPDX-License-Identifier: BSD-2-Clause
# Authors: Siddhartha Srinivasa and contributors to TSR

"""The experimental Viser backend (issue #77).

Only the pure geometry and sampling helpers are tested here: **no server is started**
and no browser is involved, so the default suite neither requires the optional extra
nor opens a port. The interactive behaviour this backend exists to evaluate is judged
by a human, not asserted here — visualization is diagnostic, never part of the
template-correctness argument.
"""

import unittest

import numpy as np

from tsr.hands import ParallelJawGripper

viser_backend = __import__("importlib").util.find_spec("viser")
if viser_backend is not None:
    from tsr.viser import (
        PRIMITIVE_SPECS,
        _generate,
        filter_templates,
        free_coordinates,
        gripper_segments,
        mode_of,
        sample_poses,
    )
    from tsr.viser import _wxyz as wxyz

GRIPPER = ParallelJawGripper(finger_length=0.08, max_aperture=0.14)


@unittest.skipIf(viser_backend is None, "optional extra 'viser' is not installed")
class TestSampling(unittest.TestCase):
    def setUp(self):
        self.templates = GRIPPER.grasp_cylinder_side(0.03, 0.12)

    def test_seed_makes_the_view_reproducible(self):
        a = sample_poses(self.templates, 3, seed=7)
        b = sample_poses(self.templates, 3, seed=7)
        self.assertEqual(len(a), 3 * len(self.templates))
        for pa, pb in zip(a, b):
            np.testing.assert_array_equal(pa, pb)

    def test_an_explicit_rng_is_honoured(self):
        poses = sample_poses(self.templates, 2, rng=np.random.default_rng(1))
        other = sample_poses(self.templates, 2, rng=np.random.default_rng(2))
        self.assertTrue(any(not np.array_equal(p, q) for p, q in zip(poses, other)))

    def test_sampled_poses_lie_in_their_template(self):
        # The viewer must show poses the TSR actually admits, at the requested frame.
        T = np.eye(4)
        T[:3, 3] = [0.4, -0.2, 0.1]
        for template in self.templates:
            tsr = template.instantiate(T)
            for pose in sample_poses([template], 4, T_ref_world=T, seed=3):
                self.assertTrue(tsr.contains(pose))


@unittest.skipIf(viser_backend is None, "optional extra 'viser' is not installed")
class TestFreeCoordinates(unittest.TestCase):
    """The sliders are driven by the template's own non-degenerate Bw rows."""

    def test_cylinder_side_is_free_in_z_and_yaw(self):
        template = GRIPPER.grasp_cylinder_side(0.03, 0.12)[0]
        free = free_coordinates(template)
        self.assertEqual([row for row, _, _, _ in free], [2, 5])
        self.assertEqual([label for _, label, _, _ in free], ["z [m]", "yaw [rad]"])
        for row, _label, lo, hi in free:
            self.assertEqual((lo, hi), (template.Bw[row, 0], template.Bw[row, 1]))
            self.assertLess(lo, hi)

    def test_cylinder_top_is_free_in_yaw_only(self):
        template = GRIPPER.grasp_cylinder_top(0.03, 0.12)[0]
        self.assertEqual([label for _, label, _, _ in free_coordinates(template)], ["yaw [rad]"])

    def test_degenerate_rows_are_never_offered(self):
        # A zero-width row is not a freedom; putting a slider on it would be a lie.
        for template in GRIPPER.grasp_box_top(0.05, 0.05, 0.05):
            for row, _label, lo, hi in free_coordinates(template):
                self.assertGreater(hi, lo, row)


@unittest.skipIf(viser_backend is None, "optional extra 'viser' is not installed")
class TestDepthAndModeIsolation(unittest.TestCase):
    """Isolating one depth or mode is the studio's main debugging move.

    ``mode_of`` is deliberately coarser than the depth family of ARCHITECTURE.md, which
    also fixes ``symmetry`` and ``metadata``; the viewer control wants the coarse
    grouping, so it does not reuse the word *family* for it.
    """

    def setUp(self):
        self.templates = GRIPPER.grasp_cylinder(0.03, 0.12, k=3)

    def test_a_depth_selects_that_level_in_every_family(self):
        # depth_index is per-FAMILY, so one index covers different metre depths: a cap
        # grasp's first depth is not a side grasp's first depth. Selecting an index must
        # therefore keep one level from each family, not one distance.
        for index in range(3):
            selected = filter_templates(self.templates, depth_index=index)
            self.assertTrue(selected)
            self.assertEqual({t.provenance.depth_index for t in selected}, {index})
            self.assertEqual(
                {mode_of(t.provenance) for t in selected},
                {mode_of(t.provenance) for t in self.templates},
            )

    def test_depths_partition_the_templates(self):
        by_depth = [filter_templates(self.templates, depth_index=i) for i in range(3)]
        self.assertEqual(sum(len(part) for part in by_depth), len(self.templates))

    def test_a_mode_selects_only_that_mode(self):
        selected = filter_templates(self.templates, mode="side/radial/tangential")
        self.assertTrue(selected)
        self.assertEqual({mode_of(t.provenance) for t in selected}, {"side/radial/tangential"})

    def test_a_mode_is_coarser_than_a_depth_family(self):
        # One mode can hold several depth families: a cylinder side grasp has two roll
        # variants, and each is its own depth family.
        side = filter_templates(self.templates, mode="side/radial/tangential")
        depth_families = {(t.provenance.symmetry, t.provenance.depth_index) for t in side}
        self.assertEqual({symmetry for symmetry, _ in depth_families}, {"roll0", "rollpi"})

    def test_the_two_filters_compose(self):
        selected = filter_templates(self.templates, depth_index=1, mode="side/radial/tangential")
        self.assertTrue(selected)
        self.assertEqual(
            {(mode_of(t.provenance), t.provenance.depth_index) for t in selected}, {("side/radial/tangential", 1)}
        )

    def test_no_filter_is_the_identity(self):
        self.assertEqual(filter_templates(self.templates), self.templates)


@unittest.skipIf(viser_backend is None, "optional extra 'viser' is not installed")
class TestRenderedAperture(unittest.TestCase):
    """The jaws are drawn at the width the TEMPLATE claims, or not at all.

    The drawn opening is the commanded preshape (span + clearance). Substituting the
    gripper's max_aperture for a template that carries no preshape would show a far
    coarser grasp than intended -- a silent misrepresentation in a tool whose whole
    purpose is to be believed.
    """

    def test_the_drawn_width_is_the_templates_preshape(self):
        for factory, dims, span in (
            ("grasp_cylinder_side", (0.03, 0.12), 0.06),
            ("grasp_torus_side", (0.05, 0.012), 0.024),
            ("grasp_torus_span", (0.05, 0.012), 2 * (0.05 + 0.012)),
        ):
            with self.subTest(factory=factory):
                template = getattr(GRIPPER, factory)(*dims, clearance=0.006)[0]
                preshape = float(template.preshape[0])
                self.assertAlmostEqual(preshape, span + 0.006, places=9)
                crossbar = gripper_segments(GRIPPER.finger_length, preshape)[0]
                self.assertAlmostEqual(crossbar[1][1] - crossbar[0][1], preshape, places=12)
                self.assertNotAlmostEqual(preshape, GRIPPER.max_aperture, places=3)

    def test_each_box_orientation_is_drawn_at_its_own_span(self):
        widths = {
            (t.provenance.finger_orientation, round(t.provenance.metadata["span"], 3)): float(t.preshape[0])
            for t in GRIPPER.grasp_box_top(0.05, 0.07, 0.10, clearance=0.006)
        }
        self.assertEqual(sorted(widths), [("x", 0.05), ("y", 0.07)])
        for (_orientation, span), preshape in widths.items():
            self.assertAlmostEqual(preshape, span + 0.006, places=9)


@unittest.skipIf(viser_backend is None, "optional extra 'viser' is not installed")
class TestStudioDiagnostics(unittest.TestCase):
    """The studio reports WHY a request produced nothing.

    The library's contract separates two failures: an exception means the request was
    invalid, an empty list means the feasible set is empty and the reason is logged.
    A debugging tool that collapses them, or that shows an unexplained empty scene, is
    worse than useless -- so the distinction is asserted here rather than assumed.
    """

    def test_a_feasible_request_reports_no_problem(self):
        templates, diagnostics = _generate(GRIPPER, "grasp_cylinder", dict(cylinder_radius=0.03, cylinder_height=0.12))
        self.assertTrue(templates)
        self.assertEqual(diagnostics, [])

    def test_each_infeasible_request_names_its_reason(self):
        cases = {
            "exceeds_aperture": (
                ParallelJawGripper(0.08, 0.05),
                "grasp_cylinder_side",
                dict(cylinder_radius=0.04, cylinder_height=0.12),
            ),
            "finger_too_short": (ParallelJawGripper(0.02, 0.30), "grasp_sphere", dict(object_radius=0.06)),
            "insufficient_clearance_band": (
                ParallelJawGripper(0.08, 0.30),
                "grasp_cylinder_top",
                dict(cylinder_radius=0.03, cylinder_height=0.12, clearance=0.05),
            ),
        }
        for reason, (gripper, factory, kwargs) in cases.items():
            with self.subTest(reason=reason):
                templates, diagnostics = _generate(gripper, factory, kwargs)
                self.assertEqual(templates, [])
                self.assertTrue(any(reason in message for message in diagnostics), diagnostics)

    def test_an_invalid_request_is_reported_as_invalid_not_as_empty(self):
        templates, diagnostics = _generate(GRIPPER, "grasp_cylinder", dict(cylinder_radius=-0.03, cylinder_height=0.12))
        self.assertEqual(templates, [])
        self.assertTrue(diagnostics[0].startswith("invalid request:"), diagnostics)
        self.assertIn("cylinder_radius", diagnostics[0])

    def test_diagnostic_capture_leaves_logging_untouched(self):
        # The studio raises the log level to collect reasons; it must put it back, or a
        # session slowly fills with debug output from an unrelated tool.
        import logging

        logger = logging.getLogger("tsr.hands")
        before_level, before_handlers = logger.level, list(logger.handlers)
        _generate(ParallelJawGripper(0.02, 0.30), "grasp_sphere", dict(object_radius=0.06))
        self.assertEqual(logger.level, before_level)
        self.assertEqual(logger.handlers, before_handlers)

    def test_every_listed_factory_accepts_its_primitive_arguments(self):
        # The studio builds its sliders from this table, so a name that does not match
        # the factory signature would surface as a TypeError mid-session.
        for kind, spec in PRIMITIVE_SPECS.items():
            kwargs = {arg: initial for arg, _lo, _hi, initial in spec["dims"]}
            for factory in spec["factories"]:
                with self.subTest(factory=factory):
                    templates, diagnostics = _generate(GRIPPER, factory, kwargs)
                    self.assertFalse(any(d.startswith("invalid request") for d in diagnostics), (factory, diagnostics))
                    self.assertTrue(templates, f"{factory} produced nothing at the studio's default dimensions")


@unittest.skipIf(viser_backend is None, "optional extra 'viser' is not installed")
class TestGeometry(unittest.TestCase):
    def test_jaw_segments_follow_the_library_frame_convention(self):
        segments = gripper_segments(finger_length=0.08, aperture=0.06)
        self.assertEqual(segments.shape, (4, 2, 3))
        crossbar, finger_neg, finger_pos, stick = segments
        # Fingers extend along +z (approach) from the palm plane at z = 0.
        np.testing.assert_allclose([finger_neg[0][2], finger_pos[0][2]], [0.0, 0.0])
        np.testing.assert_allclose([finger_neg[1][2], finger_pos[1][2]], [0.08, 0.08])
        # The crossbar spans the opening along y, at the requested aperture.
        np.testing.assert_allclose(crossbar[:, 1], [-0.03, 0.03])
        self.assertLess(stick[1][2], 0.0)  # the approach stick sits behind the palm

    def test_quaternion_matches_the_rotation(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
            if np.linalg.det(q) < 0:
                q[:, 0] *= -1.0
            w, x, y, z = wxyz(q)
            # Rebuild the matrix from the quaternion and compare.
            R = np.array(
                [
                    [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                    [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                    [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
                ]
            )
            np.testing.assert_allclose(R, q, atol=1e-12)


if __name__ == "__main__":
    unittest.main()


class TestPlacementTemplatesInTheViewer(unittest.TestCase):
    """A placement collection goes through the viewer's provenance-reading paths (#167).

    The module advertised placement templates as supported input while every site that read
    provenance assumed a ``GraspProvenance``, so ``explore_templates(placer.place_box(...))``
    raised ``AttributeError``. These cover each of those paths with a placement collection,
    and keep the grasp labels unchanged.
    """

    @classmethod
    def setUpClass(cls):
        from tsr.placement import StablePlacer

        placer = StablePlacer(table_x=0.6, table_y=0.4)
        cls.placements = placer.place_box(lx=0.1, ly=0.08, lz=0.06)
        cls.mesh_placements = placer.place_mesh(
            vertices=np.array([[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]], dtype=float),
            com=np.array([0.025, 0.025, 0.025]),
        )
        cls.grasps = ParallelJawGripper(finger_length=0.08, max_aperture=0.14).grasp_cylinder_side(0.015, 0.12)

    def test_a_placement_is_labelled_from_its_own_record(self):
        from tsr.viser import _template_label

        for index, template in enumerate(self.placements):
            label = _template_label(index, template)
            # its own vocabulary: the resting face and the tipping angle, not a depth family
            self.assertIn(template.variant, label)
            self.assertIn(template.provenance.equilibrium, label)
            self.assertNotIn("depth", label)

    def test_a_grasp_label_is_unchanged(self):
        from tsr.viser import _template_label

        label = _template_label(0, self.grasps[0])
        self.assertIn("depth", label)
        self.assertIn(self.grasps[0].provenance.mode, label)

    def test_placements_group_by_resting_face(self):
        """An analytic box carries no ``face_index``, so grouping on that alone would
        collapse all six faces into one entry; the face normal names them."""
        from tsr.viser import mode_of

        groups = {mode_of(t.provenance) for t in self.placements}
        self.assertEqual(len(groups), len(self.placements))
        mesh_groups = {mode_of(t.provenance) for t in self.mesh_placements}
        self.assertEqual(len(mesh_groups), len(self.mesh_placements))

    def test_filtering_a_placement_collection(self):
        from tsr.viser import filter_templates, mode_of

        one = mode_of(self.placements[0].provenance)
        self.assertEqual(len(filter_templates(self.placements, mode=one)), 1)
        # depth_index is a grasp idea, so no placement can match it -- and it must not raise
        self.assertEqual(filter_templates(self.placements, depth_index=0), [])
        self.assertEqual(len(filter_templates(self.placements)), len(self.placements))

    def test_sampling_and_free_coordinates_work_on_a_placement(self):
        from tsr.viser import free_coordinates, sample_poses

        poses = sample_poses(self.placements, 2, rng=np.random.default_rng(0))
        self.assertEqual(len(poses), 2 * len(self.placements))
        for template in self.placements:
            free_coordinates(template)  # must not raise on a record with no depth family

    def test_a_mixed_collection_is_refused_naming_the_mixture(self):
        """The controls differ per kind, so a mixed collection cannot be drawn honestly.
        Refusing it names what it holds, rather than showing half the controls inert."""
        from tsr.viser import collection_kind

        self.assertEqual(collection_kind(self.placements), "placement")
        self.assertEqual(collection_kind(self.grasps), "grasp")
        with self.assertRaises(ValueError) as caught:
            collection_kind(list(self.placements) + list(self.grasps))
        message = str(caught.exception)
        self.assertIn("grasp", message)
        self.assertIn("placement", message)
        self.assertIn(str(len(self.placements)), message)

    def test_an_empty_collection_has_no_kind_and_does_not_raise(self):
        from tsr.viser import collection_kind

        self.assertEqual(collection_kind([]), "")

    def test_a_record_declaring_no_kind_is_described_rather_than_refused(self):
        """A third-party generator's record should still draw. Dispatching on the wire
        discriminator rather than on the class is what makes that possible (#160)."""
        import dataclasses

        from tsr.viser import _template_label, collection_kind, mode_of

        class ThirdParty:
            pass

        template = dataclasses.replace(self.placements[0], provenance=ThirdParty())
        self.assertEqual(collection_kind([template]), "")
        self.assertIn("ThirdParty", mode_of(template.provenance))
        _template_label(0, template)  # must not raise
