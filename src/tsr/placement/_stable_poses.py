# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Stable-pose detection helpers for tsr.placement."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterator, Tuple

import numpy as np
from gafropy import Rotor, Vector
from scipy.spatial import ConvexHull


def _rotor_to_align(a: np.ndarray, b: np.ndarray) -> Rotor:
    """Return a gafropy ``Rotor`` R such that ``R a = b`` (both unit vectors).

    Pure CGA via ``Vector.get_rotor``; the antiparallel case (no unique shortest
    arc) is a 180° rotation about an axis perpendicular to ``a``, expressed as a
    rotor exponential of the perpendicular bivector.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if float(np.dot(a, b)) < -1.0 + 1e-9:
        perp = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        perp = perp - np.dot(perp, a) * a
        perp /= np.linalg.norm(perp)
        # 180° about ``perp``: bivector generator = pi * (Hodge dual of perp).
        # dual of [x,y,z] in [e12,e13,e23] coords is [z, -y, x].
        biv = np.pi * np.array([perp[2], -perp[1], perp[0]])
        return Rotor.exp(float(biv[0]), float(biv[1]), float(biv[2]))
    return Vector(float(a[0]), float(a[1]), float(a[2])).get_rotor(
        Vector(float(b[0]), float(b[1]), float(b[2]))
    )


def _dist_point_to_segment_2d(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom < 1e-20:
        return float(np.linalg.norm(p - a))
    t = float(np.clip(np.dot(p - a, ab) / denom, 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def _point_in_convex_polygon_2d(p: np.ndarray, poly: np.ndarray) -> bool:
    """True if point p is inside or on the boundary of convex polygon poly.

    poly: (M, 2) vertices in any consistent winding order.
    """
    n = len(poly)
    sign = None
    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]
        cross = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
        if abs(cross) < 1e-10:
            continue  # point on this edge; check remaining edges
        if sign is None:
            sign = cross > 0
        elif (cross > 0) != sign:
            return False
    return True


def _polygon_edge_dist_2d(p: np.ndarray, poly: np.ndarray) -> float:
    """Min distance from point p to any edge of polygon poly (2D)."""
    n = len(poly)
    return min(_dist_point_to_segment_2d(p, poly[i], poly[(i + 1) % n]) for i in range(n))


def stable_poses_mesh(
    vertices: np.ndarray,
    com: np.ndarray,
) -> Iterator[Tuple[np.ndarray, float, float]]:
    """Detect stable resting poses of a rigid body via convex hull + COM projection.

    Groups co-planar hull facets into faces, then for each face checks whether
    the COM projects onto the face polygon.  Works for non-convex objects:
    the support polygon is the convex hull of the contact points.

    Args:
        vertices: (N, 3) array of object vertices in the object frame.
        com:      (3,) center of mass in the same frame.

    Yields:
        (rotor, com_height, stability_margin) for each stable face:
        - rotor (gafropy.Rotor): rotation s.t. face outward-normal → -z (face
          rests on table).
        - com_height (float): perpendicular distance from COM to face / table height.
        - stability_margin (float): arctan(d_min / com_height) in radians.
    """
    vertices = np.asarray(vertices, dtype=float)
    com = np.asarray(com, dtype=float)
    hull = ConvexHull(vertices)

    # Group triangles that share the same outward normal into one face.
    face_groups: dict = defaultdict(list)
    for i, simplex in enumerate(hull.simplices):
        n_key = tuple(np.round(hull.equations[i, :3], 8))
        face_groups[n_key].append(i)

    _neg_z = np.array([0.0, 0.0, -1.0])

    for n_key, simplex_indices in face_groups.items():
        n = np.array(n_key, dtype=float)
        norm = np.linalg.norm(n)
        if norm < 1e-12:
            continue
        n /= norm

        d = float(hull.equations[simplex_indices[0], 3])

        # COM height above this face (positive = COM on the interior side).
        com_height = -(np.dot(n, com) + d)
        if com_height < 1e-10:
            continue  # degenerate or COM outside hull

        # Collect all unique vertex indices for this face.
        verts_idx = set()
        for si in simplex_indices:
            verts_idx.update(hull.simplices[si])
        face_verts = vertices[sorted(verts_idx)]  # (M, 3)

        # Project COM onto face plane.
        v0 = face_verts[0]
        p3 = com - np.dot(com - v0, n) * n  # 3D projection onto face

        # Project face and point onto the best 2D plane (drop dominant axis of n).
        i0 = int(np.argmax(np.abs(n)))
        ax = [i for i in range(3) if i != i0]
        pts_2d = face_verts[:, ax]  # (M, 2)
        p_2d = np.array([p3[ax[0]], p3[ax[1]]])

        # Sort polygon vertices by angle from centroid (face is convex).
        center_2d = pts_2d.mean(axis=0)
        angles = np.arctan2(pts_2d[:, 1] - center_2d[1], pts_2d[:, 0] - center_2d[0])
        poly = pts_2d[np.argsort(angles)]

        if not _point_in_convex_polygon_2d(p_2d, poly):
            continue

        d_min = _polygon_edge_dist_2d(p_2d, poly)
        stability_margin = float(np.arctan2(d_min, com_height))

        rotor = _rotor_to_align(n, _neg_z)
        yield rotor, com_height, stability_margin
