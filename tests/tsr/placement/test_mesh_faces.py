# SPDX-License-Identifier: BSD-2-Clause
# Authors: Siddhartha Srinivasa and contributors to TSR

"""``place_mesh`` describes the solid, not its triangulation (issue #152).

Two claims that pull in opposite directions, and both have to hold.

**A face is a face however it was triangulated.** Hull facets that lie in one plane are
one resting face. They are grouped by proximity, because the facets of a single face
agree only to floating-point precision — and a mesh loaded from a file carries float32
vertices, which puts that error around 1e-7.

**A face is not a face merely because it is nearly parallel to its neighbour.** A
tessellated curved surface has genuinely distinct facets, and collapsing them would
invent a flat face the caller never supplied. `place_mesh` describes the polyhedron it
was given; the library cannot know that a 48-gon prism was meant to be a cylinder.

What reconciles the two with `place_cylinder`, which documents that resting on the curved
side is not stable, is that the tessellation's margins converge to the truth: a cylinder
cut into ``n`` sides rests on each at exactly ``180/n`` degrees, which tends to 0 — the
right answer for an ideal cylinder, whose contact is a line and which rolls rather than
tipping. So ``min_margin_deg`` above ``180/n`` is how a caller says "this is a curved
surface", and the two entry points then agree.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tsr.placement import StablePlacer

TABLE = dict(table_x=0.60, table_y=0.60)


def _box(lx, ly, lz):
    return np.array([[sx * lx, sy * ly, sz * lz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) / 2.0


def _tessellated_cylinder(radius, height, n):
    a = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    rim = np.stack([radius * np.cos(a), radius * np.sin(a)], axis=-1)
    return np.vstack([np.c_[rim, np.full(n, z)] for z in (-height / 2, height / 2)])


def _margins(vertices, com=None, **kwargs):
    com = np.zeros(3) if com is None else com
    return sorted(np.degrees([t.stability_margin for t in StablePlacer(**TABLE).place_mesh(vertices, com, **kwargs)]))


# --------------------------------------------------------------------------- #
# One plane is one face
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("seed", [0, 1, 7, 23])
def test_a_mesh_with_float32_vertices_rests_exactly_as_its_float64_twin(seed):
    """The defect this fix exists for, and the common case rather than a corner one:
    STL, OBJ, glTF and MuJoCo all store float32, so this is what loading an asset gives.

    Grouping facets by an 8-decimal rounding of the hull normal split each of the box's
    six faces into two single-triangle "faces", neither of which can support the centre
    of mass. Released 3.0.0 answered with 12 templates all claiming a 0° margin — every
    one admitted by the default filter; with containment failing closed (#153) it would
    instead answer with no placements at all.
    """
    lx, ly, lz = 0.20, 0.10, 0.30
    Q = Rotation.random(random_state=seed).as_matrix()
    exact = sorted(
        np.repeat(
            [np.degrees(np.arctan2(min(np.delete([lx, ly, lz], i)) / 2, [lx, ly, lz][i] / 2)) for i in range(3)], 2
        )
    )

    float32 = (_box(lx, ly, lz).astype(np.float32) @ Q.T.astype(np.float32)).astype(np.float64)
    got = _margins(float32, float32.mean(axis=0))
    assert len(got) == 6
    np.testing.assert_allclose(got, exact, rtol=0, atol=1e-3)


def test_a_face_split_into_many_facets_is_still_one_face():
    """A face triangulated finely, by adding points along its edges, must not be read as
    many small faces: the support polygon is the whole face."""
    lx, ly, lz = 0.20, 0.10, 0.30
    extra = np.array([[x, y, lz / 2] for x in np.linspace(-lx / 2, lx / 2, 5) for y in (-ly / 2, ly / 2)])
    dense = np.vstack([_box(lx, ly, lz), extra])
    np.testing.assert_allclose(_margins(dense), _margins(_box(lx, ly, lz)), rtol=0, atol=1e-12)


# --------------------------------------------------------------------------- #
# Nearly-parallel is not the same plane
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n", [8, 12, 24, 48, 96])
def test_a_tessellated_curved_surface_keeps_its_facets(n):
    """Each side facet is a real resting face of the prism the caller supplied, and its
    margin is exactly ``180/n`` — a number that goes to zero as the tessellation
    refines, which is the ideal cylinder's true neutral equilibrium."""
    verts = _tessellated_cylinder(0.04, 0.12, n)
    placer = StablePlacer(**TABLE)
    sides = [t for t in placer.place_mesh(verts, np.zeros(3)) if abs(t.Tw_e[2, 2]) < 0.9]
    assert len(sides) == n
    np.testing.assert_allclose(np.degrees([t.stability_margin for t in sides]), 180.0 / n, rtol=1e-6)


def test_filtering_a_tessellated_cylinder_agrees_with_place_cylinder():
    """The cross-check the two entry points never had. Above the tessellation's own
    ``180/n``, ``place_mesh`` returns exactly what ``place_cylinder`` does: the two caps,
    at the same height, with the same tipping angle."""
    r, h, n = 0.04, 0.12, 48
    placer = StablePlacer(**TABLE)
    by_mesh = placer.place_mesh(_tessellated_cylinder(r, h, n), np.zeros(3), min_margin_deg=1.1 * 180.0 / n)
    by_primitive = placer.place_cylinder(r, h)

    assert len(by_mesh) == len(by_primitive) == 2
    np.testing.assert_allclose(
        sorted(t.T_ref_tsr[2, 3] for t in by_mesh),
        sorted(t.T_ref_tsr[2, 3] for t in by_primitive),
        atol=1e-12,
    )
    # The cap's support polygon is the tessellated n-gon inscribed in the true disc, so
    # its margin is the primitive's scaled by the apothem ratio cos(pi/n).
    np.testing.assert_allclose(
        sorted(np.tan([t.stability_margin for t in by_mesh])),
        np.sort(np.tan([t.stability_margin for t in by_primitive])) * np.cos(np.pi / n),
        rtol=1e-9,
    )


def test_merging_never_collapses_two_faces_of_a_real_polyhedron():
    """The guard on the merge tolerance: a shallow wedge whose two upper faces meet at
    0.01 rad — four orders above the tolerance — stays two faces."""
    half_angle = 0.005
    apex = np.tan(half_angle) * 0.10
    wedge = np.array(
        [
            [-0.10, -0.05, 0.0],
            [0.10, -0.05, 0.0],
            [0.10, 0.05, 0.0],
            [-0.10, 0.05, 0.0],
            [0.0, -0.05, apex],
            [0.0, 0.05, apex],
        ]
    )
    placer = StablePlacer(**TABLE)
    normals = {
        tuple(np.round(t.Tw_e[:3, :3].T @ np.array([0.0, 0.0, -1.0]), 6))
        for t in placer.place_mesh(wedge, wedge.mean(axis=0))
    }
    assert len(normals) == len(placer.place_mesh(wedge, wedge.mean(axis=0)))
