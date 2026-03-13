"""Tests for TablePlacer."""
import unittest
import numpy as np
from numpy import pi

from tsr.placement import TablePlacer
from tsr.template import TSRTemplate

TX = 0.30
TY = 0.20


def _valid_se3(T):
    R = T[:3, :3]
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-9)
    np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-9)


def _check_common(test, templates, subject, reference, count=None):
    """Assert count, types, task/subject/reference, Bw shape."""
    if count is not None:
        test.assertEqual(len(templates), count)
    for t in templates:
        test.assertIsInstance(t, TSRTemplate)
        test.assertEqual(t.task, "place")
        test.assertEqual(t.subject, subject)
        test.assertEqual(t.reference, reference)
        test.assertEqual(t.Bw.shape, (6, 2))
        _valid_se3(t.Tw_e)
        _valid_se3(t.T_ref_tsr)


def _check_bw_standard(test, t):
    """xy = table extents, z/roll/pitch fixed, yaw = [-π, π]."""
    np.testing.assert_allclose(t.Bw[0], [-TX, TX])
    np.testing.assert_allclose(t.Bw[1], [-TY, TY])
    test.assertEqual(t.Bw[2, 0], t.Bw[2, 1])   # z fixed
    test.assertEqual(t.Bw[3, 0], t.Bw[3, 1])   # roll fixed
    test.assertEqual(t.Bw[4, 0], t.Bw[4, 1])   # pitch fixed
    np.testing.assert_allclose(t.Bw[5], [-pi, pi])


class TestTablePlacerCylinder(unittest.TestCase):

    def setUp(self):
        self.placer = TablePlacer(table_x=TX, table_y=TY)

    def test_returns_two_templates(self):
        _check_common(self, self.placer.place_cylinder(0.04, 0.12), "object", "table", 2)

    def test_zero_radius_raises(self):
        with self.assertRaises(ValueError):
            self.placer.place_cylinder(0.0, 0.12)

    def test_zero_height_raises(self):
        with self.assertRaises(ValueError):
            self.placer.place_cylinder(0.04, 0.0)

    def test_subject_reference_forwarded(self):
        t = self.placer.place_cylinder(0.04, 0.12, subject="mug")[0]
        self.assertEqual(t.subject, "mug")
        self.assertEqual(t.reference, "table")

    def test_com_height_is_half_height(self):
        H = 0.12
        for t in self.placer.place_cylinder(0.04, H):
            np.testing.assert_allclose(t.Tw_e[2, 3], H / 2)

    def test_neg_z_face_has_identity_rotation(self):
        t = self.placer.place_cylinder(0.04, 0.12)[0]
        self.assertEqual(t.variant, "-z")
        np.testing.assert_allclose(t.Tw_e[:3, :3], np.eye(3), atol=1e-10)

    def test_bw_standard_all_templates(self):
        for t in self.placer.place_cylinder(0.04, 0.12):
            _check_bw_standard(self, t)

    def test_instantiate_and_sample(self):
        table_pose = np.eye(4)
        table_pose[2, 3] = 0.75
        t = self.placer.place_cylinder(0.04, 0.12)[0]
        pose = t.instantiate(table_pose).sample()
        self.assertEqual(pose.shape, (4, 4))
        np.testing.assert_allclose(pose[3], [0, 0, 0, 1], atol=1e-10)
        _valid_se3(pose)
        # COM z ≈ table_z + height/2 when Bw[xy] = 0
        tsr = t.instantiate(table_pose)
        pose_at_zero = tsr.sample()  # any sample; just check it's reachable
        self.assertGreater(pose_at_zero[2, 3], 0.75)  # above table


class TestTablePlacerBox(unittest.TestCase):

    def setUp(self):
        self.placer = TablePlacer(table_x=TX, table_y=TY)

    def test_always_returns_six_templates(self):
        # All 6 faces are returned regardless of dimension symmetry.
        _check_common(self, self.placer.place_box(0.10, 0.08, 0.06), "object", "table", 6)

    def test_equal_dims_still_six_templates(self):
        _check_common(self, self.placer.place_box(0.10, 0.10, 0.06), "object", "table", 6)

    def test_cube_still_six_templates(self):
        _check_common(self, self.placer.place_box(0.10, 0.10, 0.10), "object", "table", 6)

    def test_zero_dim_raises(self):
        with self.assertRaises(ValueError):
            self.placer.place_box(0.0, 0.08, 0.06)

    def test_com_heights_match_half_extents(self):
        # Each unique height appears twice (once per face in each opposing pair).
        LX, LY, LZ = 0.10, 0.08, 0.06
        heights = sorted(t.Tw_e[2, 3] for t in self.placer.place_box(LX, LY, LZ))
        expected = sorted([LX/2, LX/2, LY/2, LY/2, LZ/2, LZ/2])
        np.testing.assert_allclose(heights, expected, atol=1e-10)

    def test_bw_standard_all_templates(self):
        for t in self.placer.place_box(0.10, 0.08, 0.06):
            _check_bw_standard(self, t)

    def test_tw_e_valid_se3_all_templates(self):
        for t in self.placer.place_box(0.10, 0.08, 0.06):
            _valid_se3(t.Tw_e)

    def test_instantiate_and_sample(self):
        table_pose = np.eye(4)
        for t in self.placer.place_box(0.10, 0.08, 0.06):
            pose = t.instantiate(table_pose).sample()
            self.assertEqual(pose.shape, (4, 4))
            np.testing.assert_allclose(pose[3], [0, 0, 0, 1], atol=1e-10)
            _valid_se3(pose)


class TestTablePlacerSphere(unittest.TestCase):

    def setUp(self):
        self.placer = TablePlacer(table_x=TX, table_y=TY)

    def test_returns_one_template(self):
        _check_common(self, self.placer.place_sphere(0.05), "object", "table", 1)

    def test_zero_radius_raises(self):
        with self.assertRaises(ValueError):
            self.placer.place_sphere(0.0)

    def test_com_height_equals_radius(self):
        r = 0.05
        t = self.placer.place_sphere(r)[0]
        np.testing.assert_allclose(t.Tw_e[2, 3], r)

    def test_all_orientations_free(self):
        t = self.placer.place_sphere(0.05)[0]
        np.testing.assert_allclose(t.Bw[3], [-pi, pi])
        np.testing.assert_allclose(t.Bw[4], [-pi, pi])
        np.testing.assert_allclose(t.Bw[5], [-pi, pi])

    def test_tw_e_identity_rotation(self):
        t = self.placer.place_sphere(0.05)[0]
        np.testing.assert_allclose(t.Tw_e[:3, :3], np.eye(3), atol=1e-10)


class TestTablePlacerTorus(unittest.TestCase):

    def setUp(self):
        self.placer = TablePlacer(table_x=TX, table_y=TY)

    def test_returns_two_templates(self):
        _check_common(self, self.placer.place_torus(0.05, 0.01), "object", "table", 2)

    def test_com_height_equals_minor_radius(self):
        r = 0.015
        for t in self.placer.place_torus(0.05, r):
            np.testing.assert_allclose(t.Tw_e[2, 3], r)

    def test_minor_geq_major_raises(self):
        with self.assertRaises(ValueError):
            self.placer.place_torus(0.05, 0.05)

    def test_zero_major_raises(self):
        with self.assertRaises(ValueError):
            self.placer.place_torus(0.0, 0.01)

    def test_neg_z_face_has_identity_rotation(self):
        t = self.placer.place_torus(0.05, 0.01)[0]
        self.assertEqual(t.variant, "-z")
        np.testing.assert_allclose(t.Tw_e[:3, :3], np.eye(3), atol=1e-10)

    def test_bw_standard_all_templates(self):
        for t in self.placer.place_torus(0.05, 0.01):
            _check_bw_standard(self, t)


class TestTablePlacerMesh(unittest.TestCase):

    def setUp(self):
        self.placer = TablePlacer(table_x=TX, table_y=TY)
        L = 0.05
        self.cube_verts = np.array([
            [-L, -L, -L], [L, -L, -L], [L, L, -L], [-L, L, -L],
            [-L, -L,  L], [L, -L,  L], [L, L,  L], [-L, L,  L],
        ])
        self.L = L
        self.cube_com = np.zeros(3)

    def test_cube_returns_six_templates(self):
        templates = self.placer.place_mesh(self.cube_verts, self.cube_com)
        self.assertEqual(len(templates), 6)

    def test_all_templates_valid(self):
        _check_common(
            self,
            self.placer.place_mesh(self.cube_verts, self.cube_com, subject="box"),
            "box", "table",
        )

    def test_cube_com_heights_equal_half_side(self):
        for t in self.placer.place_mesh(self.cube_verts, self.cube_com):
            np.testing.assert_allclose(t.Tw_e[2, 3], self.L, atol=1e-10)

    def test_bw_standard_all_templates(self):
        for t in self.placer.place_mesh(self.cube_verts, self.cube_com):
            _check_bw_standard(self, t)

    def test_invalid_vertices_shape_raises(self):
        with self.assertRaises(ValueError):
            self.placer.place_mesh(np.ones((4, 2)), self.cube_com)

    def test_invalid_com_shape_raises(self):
        with self.assertRaises(ValueError):
            self.placer.place_mesh(self.cube_verts, np.ones(4))

    def test_instantiate_and_sample(self):
        table_pose = np.eye(4)
        for t in self.placer.place_mesh(self.cube_verts, self.cube_com):
            pose = t.instantiate(table_pose).sample()
            self.assertEqual(pose.shape, (4, 4))
            np.testing.assert_allclose(pose[3], [0, 0, 0, 1], atol=1e-10)
            _valid_se3(pose)

    def test_off_center_com_cube_still_six_templates(self):
        # For a cube the face squares are large enough that any interior COM
        # projection lands inside every face (cube-specific, not a general rule).
        L = 0.05
        verts = np.array([
            [-L, -L, -L], [L, -L, -L], [L, L, -L], [-L, L, -L],
            [-L, -L,  L], [L, -L,  L], [L, L,  L], [-L, L,  L],
        ])
        com = np.array([L * 0.5, L * 0.5, L * 0.5])
        templates = self.placer.place_mesh(verts, com)
        self.assertEqual(len(templates), 6)

    def test_unstable_face_filtered(self):
        # Elongated tetrahedron: tiny base triangle + apex very far in x.
        # The COM lands outside the tiny base, so that face is not returned.
        verts = np.array([
            [0.0,   0.0,  0.0],
            [0.01,  0.0,  0.0],
            [0.005, 0.01, 0.0],
            [100.0, 0.0,  0.001],  # apex far in x
        ])
        com = verts.mean(axis=0)   # centroid ≈ (25, 0.0025, 0.00025)
        templates = self.placer.place_mesh(verts, com)
        # Base face (z≈0) has COM projection at x≈25, outside the 0.01-wide triangle.
        self.assertLess(len(templates), 4)


class TestTablePlacerInit(unittest.TestCase):

    def test_negative_table_x_raises(self):
        with self.assertRaises(ValueError):
            TablePlacer(table_x=-0.3, table_y=0.2)

    def test_negative_table_y_raises(self):
        with self.assertRaises(ValueError):
            TablePlacer(table_x=0.3, table_y=-0.2)

    def test_custom_reference(self):
        placer = TablePlacer(table_x=0.3, table_y=0.2, reference="countertop")
        t = placer.place_cylinder(0.04, 0.12)[0]
        self.assertEqual(t.reference, "countertop")

    def test_table_extents_stored(self):
        placer = TablePlacer(table_x=0.3, table_y=0.2)
        self.assertAlmostEqual(placer.table_x, 0.3)
        self.assertAlmostEqual(placer.table_y, 0.2)


class TestNewAPI(unittest.TestCase):
    """Tests for stability_margin, sample(), min_margin_deg, and __repr__."""

    def setUp(self):
        self.placer = TablePlacer(table_x=TX, table_y=TY)
        L = 0.05
        verts = np.array([
            [-L, -L, -L], [L, -L, -L], [L, L, -L], [-L, L, -L],
            [-L, -L,  L], [L, -L,  L], [L, L,  L], [-L, L,  L],
        ])
        self.cube_verts = verts
        self.cube_com = np.zeros(3)

    # -- stability_margin --------------------------------------------------

    def test_place_mesh_stability_margin_set(self):
        tmpls = self.placer.place_mesh(self.cube_verts, self.cube_com)
        for t in tmpls:
            self.assertIsNotNone(t.stability_margin)
            self.assertGreater(t.stability_margin, 0.0)

    def test_place_mesh_sorted_descending(self):
        tmpls = self.placer.place_mesh(self.cube_verts, self.cube_com)
        margins = [t.stability_margin for t in tmpls]
        self.assertEqual(margins, sorted(margins, reverse=True))

    def test_primitive_stability_margin_none(self):
        for t in self.placer.place_cylinder(0.04, 0.12):
            self.assertIsNone(t.stability_margin)
        for t in self.placer.place_box(0.08, 0.06, 0.18):
            self.assertIsNone(t.stability_margin)
        for t in self.placer.place_sphere(0.04):
            self.assertIsNone(t.stability_margin)
        for t in self.placer.place_torus(0.05, 0.01):
            self.assertIsNone(t.stability_margin)

    # -- min_margin_deg ----------------------------------------------------

    def test_min_margin_deg_filters(self):
        all_tmpls = self.placer.place_mesh(self.cube_verts, self.cube_com)
        best = all_tmpls[0]
        threshold_deg = np.degrees(best.stability_margin) - 1.0
        filtered = self.placer.place_mesh(self.cube_verts, self.cube_com,
                                          min_margin_deg=threshold_deg)
        self.assertLessEqual(len(filtered), len(all_tmpls))
        for t in filtered:
            self.assertGreaterEqual(np.degrees(t.stability_margin), threshold_deg - 1e-9)

    def test_min_margin_deg_zero_returns_all(self):
        all_tmpls = self.placer.place_mesh(self.cube_verts, self.cube_com)
        filtered  = self.placer.place_mesh(self.cube_verts, self.cube_com,
                                           min_margin_deg=0.0)
        self.assertEqual(len(filtered), len(all_tmpls))

    def test_min_margin_deg_high_returns_empty(self):
        tmpls = self.placer.place_mesh(self.cube_verts, self.cube_com,
                                       min_margin_deg=90.0)
        self.assertEqual(len(tmpls), 0)

    # -- sample() shorthand ------------------------------------------------

    def test_sample_returns_valid_se3(self):
        table_pose = np.eye(4)
        table_pose[2, 3] = 0.75
        for t in self.placer.place_mesh(self.cube_verts, self.cube_com):
            pose = t.sample(table_pose)
            _valid_se3(pose)

    def test_sample_matches_instantiate_sample(self):
        table_pose = np.eye(4)
        table_pose[2, 3] = 0.75
        t = self.placer.place_cylinder(0.04, 0.12)[0]
        # Both paths should produce valid SE(3) poses (values differ due to sampling)
        _valid_se3(t.sample(table_pose))
        _valid_se3(t.instantiate(table_pose).sample())

    # -- __repr__ ----------------------------------------------------------

    def test_template_repr_contains_task_and_subject(self):
        t = self.placer.place_cylinder(0.04, 0.12, subject="mug")[0]
        r = repr(t)
        self.assertIn("task='place'", r)
        self.assertIn("subject='mug'", r)

    def test_template_repr_shows_margin_for_mesh(self):
        t = self.placer.place_mesh(self.cube_verts, self.cube_com)[0]
        r = repr(t)
        self.assertIn("margin=", r)
        self.assertIn("°", r)

    def test_tsr_repr_shows_free_dofs(self):
        from tsr.tsr import TSR
        tsr = TSR()
        tsr.Bw[0] = [-1.0, 1.0]  # x free
        tsr.Bw[5] = [-np.pi, np.pi]  # yaw free
        r = repr(tsr)
        self.assertIn("x", r)
        self.assertIn("yaw", r)

    # -- stability_margin round-trips through serialization ----------------

    def test_stability_margin_survives_json_roundtrip(self):
        import json
        t = self.placer.place_mesh(self.cube_verts, self.cube_com)[0]
        t2 = TSRTemplate.from_json(t.to_json())
        self.assertAlmostEqual(t.stability_margin, t2.stability_margin)

    def test_stability_margin_none_survives_json_roundtrip(self):
        import json
        t = self.placer.place_cylinder(0.04, 0.12)[0]
        t2 = TSRTemplate.from_json(t.to_json())
        self.assertIsNone(t2.stability_margin)


if __name__ == "__main__":
    unittest.main()
