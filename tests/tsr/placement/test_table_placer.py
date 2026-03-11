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

    def test_returns_one_template(self):
        _check_common(self, self.placer.place_cylinder(0.04, 0.12), "object", "table", 1)

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
        t = self.placer.place_cylinder(0.04, H)[0]
        np.testing.assert_allclose(t.Tw_e[2, 3], H / 2)

    def test_tw_e_identity_rotation(self):
        t = self.placer.place_cylinder(0.04, 0.12)[0]
        np.testing.assert_allclose(t.Tw_e[:3, :3], np.eye(3), atol=1e-10)

    def test_bw_standard(self):
        _check_bw_standard(self, self.placer.place_cylinder(0.04, 0.12)[0])

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

    def test_distinct_dims_returns_three_templates(self):
        _check_common(self, self.placer.place_box(0.10, 0.08, 0.06), "object", "table", 3)

    def test_two_equal_dims_deduplicates(self):
        # lx == ly → only 2 unique heights (lx/2 = ly/2 and lz/2)
        _check_common(self, self.placer.place_box(0.10, 0.10, 0.06), "object", "table", 2)

    def test_cube_deduplicates_to_one(self):
        _check_common(self, self.placer.place_box(0.10, 0.10, 0.10), "object", "table", 1)

    def test_zero_dim_raises(self):
        with self.assertRaises(ValueError):
            self.placer.place_box(0.0, 0.08, 0.06)

    def test_com_heights_match_half_extents(self):
        LX, LY, LZ = 0.10, 0.08, 0.06
        heights = sorted(t.Tw_e[2, 3] for t in self.placer.place_box(LX, LY, LZ))
        np.testing.assert_allclose(heights, sorted([LX / 2, LY / 2, LZ / 2]), atol=1e-10)

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

    def test_returns_one_template(self):
        _check_common(self, self.placer.place_torus(0.05, 0.01), "object", "table", 1)

    def test_com_height_equals_minor_radius(self):
        r = 0.015
        t = self.placer.place_torus(0.05, r)[0]
        np.testing.assert_allclose(t.Tw_e[2, 3], r)

    def test_minor_geq_major_raises(self):
        with self.assertRaises(ValueError):
            self.placer.place_torus(0.05, 0.05)

    def test_zero_major_raises(self):
        with self.assertRaises(ValueError):
            self.placer.place_torus(0.0, 0.01)

    def test_tw_e_identity_rotation(self):
        t = self.placer.place_torus(0.05, 0.01)[0]
        np.testing.assert_allclose(t.Tw_e[:3, :3], np.eye(3), atol=1e-10)

    def test_bw_standard(self):
        _check_bw_standard(self, self.placer.place_torus(0.05, 0.01)[0])


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


if __name__ == "__main__":
    unittest.main()
