# SPDX-License-Identifier: BSD-2-Clause
# Authors: Siddhartha Srinivasa and contributors to TSR

"""Stable-pose detection helpers for tsr.placement."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterator, Tuple

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from ..core.utils import length_atol


def _rotation_to_align(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return rotation matrix R such that R @ a = b (both unit vectors)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = float(np.linalg.norm(v))
    if s < 1e-12:
        return np.eye(3) if c > 0 else _rotation_180_perp(a)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1.0 - c) / (s * s)


def _rotation_180_perp(a: np.ndarray) -> np.ndarray:
    """180° rotation around an axis perpendicular to unit vector a."""
    perp = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    perp = perp - np.dot(perp, a) * a
    perp /= np.linalg.norm(perp)
    return 2.0 * np.outer(perp, perp) - np.eye(3)


def _plane_basis(n: np.ndarray) -> np.ndarray:
    """Orthonormal 3x2 basis of the plane with unit normal ``n``.

    The face's in-plane distances must be measured in an **orthonormal** basis. Dropping
    the normal's dominant axis instead -- the projection this replaces -- foreshortens
    them by ``|n_max| >= 1/sqrt(3)``, so the reported tipping angle depended on how the
    object frame happened to be oriented and was under-reported by up to 40% (#151).
    """
    seed = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = seed - np.dot(seed, n) * n
    u /= np.linalg.norm(u)
    return np.stack([u, np.cross(n, u)], axis=-1)


def _inward_distance(point: np.ndarray, polygon: np.ndarray) -> float:
    """Signed distance from ``point`` to the boundary of a convex polygon, in metres.

    Positive inside, negative outside, and it is a **length** rather than the twice-area
    cross product the previous containment test compared against an absolute constant:
    that made the verdict unit-dependent, and below ~1e-5 m every edge test was skipped
    and the helper fell through to "inside", fabricating stable poses (#153).

    ``polygon`` is given in counter-clockwise order.
    """
    edges = np.roll(polygon, -1, axis=0) - polygon
    lengths = np.linalg.norm(edges, axis=1)
    keep = lengths > 0.0
    if not np.any(keep):
        return -np.inf  # degenerate support polygon: fail closed
    inward = np.stack([-edges[keep, 1], edges[keep, 0]], axis=-1) / lengths[keep, None]
    return float(np.einsum("ij,ij->i", inward, point - polygon[keep]).min())


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
        (R, origin_height, stability_margin) for each stable face:
        - R (3×3): rotation s.t. face outward-normal → -z (face rests on table).
        - origin_height (float): height of the OBJECT-FRAME ORIGIN above the
          table when this face rests on it (= perpendicular distance from the
          origin to the face plane). Placing the origin here makes the resting
          face sit at z=0 regardless of where the COM is. Equals the COM height
          only when the COM lies on the face normal through the origin.
        - stability_margin (float): the tipping angle ``arctan(d_min / com_height)`` in
          radians, where ``d_min`` is the in-plane distance from the projected COM to
          the nearest support-polygon edge and ``com_height`` the perpendicular
          COM-to-face distance. That is the angle the body must rotate about the nearest
          support edge before the COM passes over it, so it is a property of the object
          and the pose: invariant under rigid motion and under a change of object frame.

    A face is stable only when the projected COM is inside the support polygon by more
    than the scale-aware length tolerance. A COM exactly on an edge is a critical
    equilibrium, not a rest, and an indeterminate test rejects the face (#153).
    """
    vertices = np.asarray(vertices, dtype=float)
    com = np.asarray(com, dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"vertices must have shape (N, 3), got {vertices.shape}")
    if vertices.shape[0] < 4:
        raise ValueError(f"need at least 4 vertices for a 3D convex hull, got {vertices.shape[0]}")
    try:
        hull = ConvexHull(vertices)
    except QhullError as e:
        raise ValueError(
            "could not build a 3D convex hull from the given vertices; they may be "
            "coplanar, collinear, or otherwise degenerate"
        ) from e

    # Group triangles that share the same outward normal into one face.
    face_groups: dict = defaultdict(list)
    for i, simplex in enumerate(hull.simplices):
        n_key = tuple(np.round(hull.equations[i, :3], 8))
        face_groups[n_key].append(i)

    _neg_z = np.array([0.0, 0.0, -1.0])
    atol = length_atol(float(np.ptp(vertices, axis=0).max()))

    for n_key, simplex_indices in face_groups.items():
        n = np.array(n_key, dtype=float)
        norm = np.linalg.norm(n)
        if norm < 1e-12:
            continue
        n /= norm

        d = float(hull.equations[simplex_indices[0], 3])

        # COM height above this face (positive = COM on the interior side).
        # Used for the stability lever arm (the margin), NOT for placement.
        com_height = -(np.dot(n, com) + d)
        if com_height <= atol:
            continue  # degenerate, or the COM lies on or outside this face

        # Height of the object-frame ORIGIN above this face. Placing the origin
        # here makes the resting face sit at table z=0 for any COM (see C2).
        origin_height = -d

        # Collect all unique vertex indices for this face.
        verts_idx = set()
        for si in simplex_indices:
            verts_idx.update(hull.simplices[si])
        face_verts = vertices[sorted(verts_idx)]  # (M, 3)

        # Express the face and the COM's projection in an orthonormal basis of the face
        # plane, so in-plane distances are true distances.
        basis = _plane_basis(n)
        v0 = face_verts[0]
        pts_2d = (face_verts - v0) @ basis  # (M, 2)
        p_2d = (com - v0) @ basis  # the projection drops the normal component

        # Order the (convex) face counter-clockwise about its centroid.
        centred = pts_2d - pts_2d.mean(axis=0)
        poly = pts_2d[np.argsort(np.arctan2(centred[:, 1], centred[:, 0]))]

        d_min = _inward_distance(p_2d, poly)
        if d_min <= atol:
            continue  # outside the support polygon, or on its edge: a tipping case
        stability_margin = float(np.arctan2(d_min, com_height))

        R = _rotation_to_align(n, _neg_z)
        yield R, origin_height, stability_margin
