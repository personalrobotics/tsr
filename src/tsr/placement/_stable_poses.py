# SPDX-License-Identifier: BSD-2-Clause
# Authors: Siddhartha Srinivasa and contributors to TSR

"""Stable-pose detection helpers for tsr.placement."""

from __future__ import annotations

import itertools
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


#: Tolerance on a unit face normal when deciding whether two facets are coplanar. For
#: unit vectors this is the angle between them in radians. It has to absorb the error in
#: normals computed from float32 vertices (~1e-7, and every common mesh format -- STL,
#: OBJ, glTF, MuJoCo -- stores float32), while staying far below the angle between
#: genuinely distinct faces: a cylinder tessellated into ``n`` sides separates its
#: neighbours by ``2*pi/n``, which stays 600x above this tolerance even at n = 10000.
_NORMAL_ATOL = 1e-6

#: The 81 cells adjacent to a cell in the 4D plane grid, including the cell itself.
_NEIGHBOURS = np.array(list(itertools.product((-1, 0, 1), repeat=4)))


def _group_coplanar_facets(hull: ConvexHull, atol: float) -> Iterator[Tuple[np.ndarray, list]]:
    """Group hull facets that lie in the same plane, and yield ``(plane, facet indices)``.

    A convex solid's face is triangulated into several facets whose planes agree only to
    floating-point precision, so they must be grouped by *proximity*. Rounding the normal
    to a fixed number of decimals and bucketing on the result -- what this replaces --
    fragments a face whenever its facets straddle a bucket boundary. On a rotated box
    with float32 vertices that splits all six faces into twelve single-triangle "faces",
    each too small to support the centre of mass, so the box has no stable placement at
    all (#152).

    Facets are indexed in a 4D grid over ``(n, d)`` whose cells are one tolerance wide,
    and each facet joins a group whose *representative* plane is within tolerance,
    searching the 81 surrounding cells. Matching against the representative rather than
    any member bounds the group to one tolerance, so groups cannot drift by chaining, and
    searching the neighbours removes the boundary sensitivity that caused the defect.
    """
    cell = np.array([_NORMAL_ATOL, _NORMAL_ATOL, _NORMAL_ATOL, atol])
    groups: list = []  # (representative plane, [facet indices])
    index: dict = {}  # grid cell -> group id
    for i in range(len(hull.simplices)):
        plane = hull.equations[i, :4]
        key = np.floor(plane / cell).astype(np.int64)
        gid = None
        for candidate in (index.get(tuple(k)) for k in key + _NEIGHBOURS):
            if candidate is None:
                continue
            delta = np.abs(groups[candidate][0] - plane)
            if delta[:3].max() <= _NORMAL_ATOL and delta[3] <= atol:
                gid = candidate
                break
        if gid is None:
            gid = len(groups)
            groups.append((plane, []))
            index[tuple(key)] = gid
        groups[gid][1].append(i)

    for _, indices in groups:
        # The mean plane of the group: for a mesh whose vertices are not exactly
        # coplanar there is no exact face plane, and the mean minimises the residual.
        planes = hull.equations[indices, :4]
        n = planes[:, :3].mean(axis=0)
        norm = np.linalg.norm(n)
        if norm < 1e-12:
            continue
        yield np.append(n / norm, planes[:, 3].mean()), indices


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

    _neg_z = np.array([0.0, 0.0, -1.0])
    atol = length_atol(float(np.ptp(vertices, axis=0).max()))

    for plane, simplex_indices in _group_coplanar_facets(hull, atol):
        n, d = plane[:3], float(plane[3])

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

        # The support polygon is the 2D hull of the contact points, which is what a
        # non-convex object rests on. Taking the hull rather than sorting the points by
        # angle also tolerates a merged face whose points are not exactly coplanar.
        try:
            poly = pts_2d[ConvexHull(pts_2d).vertices]  # counter-clockwise
        except QhullError:
            continue  # collinear contact: a line or point support cannot hold a rest

        d_min = _inward_distance(p_2d, poly)
        if d_min <= atol:
            continue  # outside the support polygon, or on its edge: a tipping case
        stability_margin = float(np.arctan2(d_min, com_height))

        R = _rotation_to_align(n, _neg_z)
        yield R, origin_height, stability_margin
