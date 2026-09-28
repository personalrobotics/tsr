# SPDX-License-Identifier: BSD-2-Clause
# Authors: Siddhartha Srinivasa and contributors to TSR

"""``stability_margin`` is the physical tipping angle (issues #151, #153, #156).

Clause 4 of the placement contract. A tipping angle is a property of the object and the
pose, so it cannot depend on how the object frame happens to be oriented, on the units
the vertices are expressed in, or on which entry point the caller used. The defects here:

* #151 -- the COM-to-edge distance was measured in a projection that dropped the
  normal's dominant axis, foreshortening it by up to ``1/sqrt(3)``: margins were
  under-reported by up to 40% and changed when the same rigid object was re-expressed in
  a rotated frame;
* #153 -- containment was decided on a cross product (an area) against an absolute
  ``1e-10``, so below ~1e-5 m every edge test was skipped and the helper fell through to
  "inside", fabricating stable poses; a COM exactly on a support edge was admitted as a
  rest;
* #156 -- the primitive factories reported no margin and offered no ``min_margin_deg``.

The expected angles here are written independently of the generator: closed forms for the
primitives, and the oracle's ``tipping_angle``, which reads only the posed geometry, for
meshes.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from tsr.placement import StablePlacer

from .._hypothesis_strategies import rotations
from ._placement_matrix import cases, margin_failures, matrix_settings, ordering_failures
from ._placement_oracle import Mesh, tipping_angle

TABLE = dict(table_x=0.60, table_y=0.60)


def _box_vertices(lx, ly, lz):
    return np.array([[sx * lx, sy * ly, sz * lz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) / 2.0


def _analytic_box_angles(lx, ly, lz):
    """Tipping angles of a uniform box, from first principles.

    Resting on the face normal to axis ``i``, the centre of mass is ``l_i/2`` up and the
    nearest edge of the support rectangle is half the *smaller* of the other two extents
    away, so the angle is ``arctan(min(others)/l_i)``.
    """
    L = np.array([lx, ly, lz], dtype=float)
    return sorted({np.arctan2(min(np.delete(L, i)) / 2.0, L[i] / 2.0) for i in range(3)})


def _mesh_margins(vertices, com, **kwargs):
    placer = StablePlacer(**TABLE)
    return [t.stability_margin for t in placer.place_mesh(vertices, com, **kwargs)]


# --------------------------------------------------------------------------- #
# #151 -- the angle, and its independence from the object frame
# --------------------------------------------------------------------------- #


def test_box_margins_match_the_analytic_tipping_angles():
    lx, ly, lz = 0.20, 0.10, 0.30
    expected = _analytic_box_angles(lx, ly, lz)  # 18.435, 26.565, 63.435 degrees
    got = sorted(set(np.round(_mesh_margins(_box_vertices(lx, ly, lz), np.zeros(3)), 12)))
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-12)


def test_a_tilted_face_reports_its_true_tipping_angle():
    """A regular octahedron, whose every face is tilted with respect to the object frame.

    This is the shape that makes the defect visible without a random rotation: the axis
    dropped by the old projection carried ``1/sqrt(3)`` of each face normal, so all eight
    faces were reported at 22.21° instead of 35.26°. An axis-aligned box hides it
    completely — the dropped axis is exactly the one that contributes nothing — which is
    why a suite built on cubes never saw it.

    For a face of ``x + y + z = a``: the centre of mass is ``a/sqrt(3)`` below it, the
    face is equilateral with side ``a·sqrt(2)`` so its incircle radius is ``a/sqrt(6)``,
    and the tipping angle is ``arctan(sqrt(3)/sqrt(6)) = arctan(1/sqrt(2))``.
    """
    a = 0.10
    octahedron = np.array([[s * a if i == j else 0.0 for j in range(3)] for i in range(3) for s in (-1, 1)])
    expected = np.arctan(1.0 / np.sqrt(2.0))  # 35.264 degrees
    margins = _mesh_margins(octahedron, np.zeros(3))
    assert len(margins) == 8
    np.testing.assert_allclose(margins, expected, rtol=0, atol=1e-12)


@given(Q=rotations())
def test_margins_are_invariant_under_a_rigid_rotation_of_the_object(Q):
    """The regression for the reported worst case: in one rotated frame a true 18.435°
    face was reported as 11.586°, a 37% under-report. A tipping angle is physical, so
    re-expressing the same rigid body cannot change it."""
    lx, ly, lz = 0.20, 0.10, 0.30
    expected = sorted(np.repeat(_analytic_box_angles(lx, ly, lz), 2))  # opposite faces agree
    verts = _box_vertices(lx, ly, lz)
    got = sorted(_mesh_margins(verts @ Q.T, Q @ np.zeros(3)))
    # 1e-6 rad: the residual is the hull's own normals, measured at 4.2e-7 deg (7e-9 rad)
    # over 3000 random rotations -- five orders below the 40% error this guards against.
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-6)


def test_margins_rank_faces_by_the_angle_the_object_actually_tips_at():
    """The ordering claim, judged by the oracle rather than by the sorted-by-itself test
    the suite used to make."""
    verts = _box_vertices(0.20, 0.10, 0.30)
    placer = StablePlacer(**TABLE)
    templates = placer.place_mesh(verts, np.zeros(3))
    prim = Mesh(verts, np.zeros(3))
    reported, oracle = [], []
    for t in templates:
        reported.append(t.stability_margin)
        oracle.append(tipping_angle(prim, t.instantiate(np.eye(4)).to_transform(np.zeros(6))))
    np.testing.assert_allclose(reported, oracle, rtol=0, atol=1e-12)
    assert reported == sorted(reported, reverse=True)


@matrix_settings(150)
@given(case=cases(fit="fits"))
def test_every_factory_reports_the_pose_it_tips_at(case):
    templates = case.call()
    assert not margin_failures(case, templates)
    assert not ordering_failures(case, templates)


# --------------------------------------------------------------------------- #
# #153 -- scale invariance, and failing closed
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("scale", [1e-6, 1e-4, 1e-2, 1.0, 1e2, 1e3])
def test_the_stable_pose_set_does_not_depend_on_the_unit(scale):
    """The same mesh in different units is the same mesh."""
    verts = _box_vertices(0.20, 0.10, 0.30) * scale
    placer = StablePlacer(table_x=1.0 * scale, table_y=1.0 * scale)
    margins = sorted(np.round([t.stability_margin for t in placer.place_mesh(verts, np.zeros(3))], 9))
    expected = sorted(np.round(np.repeat(_analytic_box_angles(0.20, 0.10, 0.30), 2), 9))
    np.testing.assert_allclose(margins, expected, rtol=0, atol=1e-9)


@pytest.mark.parametrize("scale", [1.0, 1e-4, 1e-5, 1e-6])
def test_a_com_outside_the_support_polygon_is_never_admitted_at_any_scale(scale):
    """An obtuse tetrahedron whose centroid provably projects outside its ``z = 0``
    facet. Below 1e-5 m the old absolute tolerance skipped every edge test and the
    containment helper returned "inside", fabricating an 86° margin for the pose that
    tips — the most stable face in the ranking it produced."""
    tet = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.2, 0.15, 0.0], [0.9, 0.1, 1.0]]) * scale
    placer = StablePlacer(table_x=2.0 * scale, table_y=2.0 * scale)
    templates = placer.place_mesh(tet, tet.mean(axis=0))
    resting_on_the_bottom = [t for t in templates if abs(t.Tw_e[2, 2] + 1.0) < 1e-9]
    assert resting_on_the_bottom == []
    prim = Mesh(tet, tet.mean(axis=0))
    for t in templates:  # and whatever is returned really does rest stably
        assert tipping_angle(prim, t.instantiate(np.eye(4)).to_transform(np.zeros(6))) > 0.0


def test_a_com_on_the_support_edge_is_a_tipping_case_not_a_rest():
    """A critical equilibrium is excluded, and the boundary is probed with ``nextafter``
    on both sides. The old code admitted the exact-edge case with a 0° margin, which the
    default ``min_margin_deg=0`` then passed through as a stable placement."""
    verts = _box_vertices(0.10, 0.10, 0.10)
    placer = StablePlacer(**TABLE)
    edge = 0.05  # the COM directly above the +y edge of the bottom face

    def z_faces(com_y):
        return [t for t in placer.place_mesh(verts, np.array([0.0, com_y, 0.0])) if abs(t.Tw_e[2, 2] - 1.0) < 1e-9]

    assert len(z_faces(np.nextafter(edge, 0.0))) == 0
    assert len(z_faces(edge)) == 0
    assert len(z_faces(np.nextafter(edge, 1.0))) == 0
    inside = z_faces(0.049)  # just inside: the -z face is stable again
    assert len(inside) == 1 and inside[0].stability_margin > 0.0


# --------------------------------------------------------------------------- #
# #156 -- the primitives describe themselves the same way
# --------------------------------------------------------------------------- #


def test_primitive_margins_match_their_closed_forms():
    placer = StablePlacer(**TABLE)
    r, h = 0.04, 0.12
    for t in placer.place_cylinder(r, h):  # tips about a tangent to the rim
        np.testing.assert_allclose(t.stability_margin, np.arctan2(r, h / 2.0), atol=1e-12)

    R, minor = 0.05, 0.012
    for t in placer.place_torus(R, minor):  # tips about a tangent to the contact circle
        np.testing.assert_allclose(t.stability_margin, np.arctan2(R, minor), atol=1e-12)

    lx, ly, lz = 0.20, 0.10, 0.30
    expected = {
        "-z": np.arctan2(min(lx, ly) / 2, lz / 2),
        "+z": np.arctan2(min(lx, ly) / 2, lz / 2),
        "-y": np.arctan2(min(lx, lz) / 2, ly / 2),
        "+y": np.arctan2(min(lx, lz) / 2, ly / 2),
        "-x": np.arctan2(min(ly, lz) / 2, lx / 2),
        "+x": np.arctan2(min(ly, lz) / 2, lx / 2),
    }
    for t in placer.place_box(lx, ly, lz):
        np.testing.assert_allclose(t.stability_margin, expected[t.variant], atol=1e-12)

    # A sphere is neutrally stable: it rolls rather than tipping.
    assert placer.place_sphere(0.05)[0].stability_margin == 0.0


@given(
    lx=st.floats(0.01, 0.2),
    ly=st.floats(0.01, 0.2),
    lz=st.floats(0.01, 0.2),
)
def test_a_box_reports_the_same_margins_through_either_entry_point(lx, ly, lz):
    """The cross-check the suite never had: ``place_box`` and ``place_mesh`` describe the
    same solid, so they must agree — which also keeps the closed forms above honest."""
    placer = StablePlacer(**TABLE)
    by_primitive = sorted(t.stability_margin for t in placer.place_box(lx, ly, lz))
    by_mesh = sorted(t.stability_margin for t in placer.place_mesh(_box_vertices(lx, ly, lz), np.zeros(3)))
    np.testing.assert_allclose(by_primitive, by_mesh, rtol=1e-9, atol=1e-12)


def test_min_margin_deg_filters_primitives_as_it_filters_meshes(caplog):
    """The needle standing on its end: in equilibrium, but at 1.15°. A caller asking for
    5° gets the four side faces from either entry point."""
    placer = StablePlacer(**TABLE)
    needle = dict(lx=0.01, ly=0.01, lz=0.50)
    knife_edge = np.degrees(np.arctan2(0.005, 0.25))
    assert 1.1 < knife_edge < 1.2

    unfiltered = placer.place_box(**needle)
    assert len(unfiltered) == 6
    assert sorted(np.round(np.degrees([t.stability_margin for t in unfiltered]), 2)) == [
        1.15,
        1.15,
        45.0,
        45.0,
        45.0,
        45.0,
    ]

    filtered = placer.place_box(**needle, min_margin_deg=5.0)
    assert sorted(t.variant for t in filtered) == ["+x", "+y", "-x", "-y"]
    by_mesh = placer.place_mesh(
        _box_vertices(needle["lx"], needle["ly"], needle["lz"]), np.zeros(3), min_margin_deg=5.0
    )
    np.testing.assert_allclose(
        sorted(t.stability_margin for t in filtered),
        sorted(t.stability_margin for t in by_mesh),
        rtol=1e-9,
    )

    with caplog.at_level(logging.DEBUG, logger="tsr.placement.stable_placer"):
        assert placer.place_box(**needle, min_margin_deg=60.0) == []
    messages = [r.getMessage() for r in caplog.records if "empty feasible set" in r.getMessage()]
    assert len(messages) == 1 and "below_min_margin" in messages[0]


def test_a_positive_threshold_rejects_a_sphere(caplog):
    placer = StablePlacer(**TABLE)
    assert len(placer.place_sphere(0.05, min_margin_deg=0.0)) == 1
    with caplog.at_level(logging.DEBUG, logger="tsr.placement.stable_placer"):
        assert placer.place_sphere(0.05, min_margin_deg=1e-6) == []
    assert any("below_min_margin" in r.getMessage() for r in caplog.records)
