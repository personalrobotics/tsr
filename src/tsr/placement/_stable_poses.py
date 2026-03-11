"""Stable-pose detection helpers for tsr.placement."""
from __future__ import annotations

from collections import defaultdict
from typing import Iterator, Tuple

import numpy as np
from scipy.spatial import ConvexHull


def _rotation_to_align(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return rotation matrix R such that R @ a = b (both unit vectors)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = float(np.linalg.norm(v))
    if s < 1e-12:
        return np.eye(3) if c > 0 else _rotation_180_perp(a)
    vx = np.array([[0, -v[2], v[1]],
                   [v[2],  0, -v[0]],
                   [-v[1], v[0],  0]])
    return np.eye(3) + vx + vx @ vx * (1.0 - c) / (s * s)


def _rotation_180_perp(a: np.ndarray) -> np.ndarray:
    """180° rotation around an axis perpendicular to unit vector a."""
    perp = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    perp = perp - np.dot(perp, a) * a
    perp /= np.linalg.norm(perp)
    return 2.0 * np.outer(perp, perp) - np.eye(3)


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
    return min(
        _dist_point_to_segment_2d(p, poly[i], poly[(i + 1) % n])
        for i in range(n)
    )


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
        (R, com_height, stability_margin) for each stable face:
        - R (3×3): rotation s.t. face outward-normal → -z (face rests on table).
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
        pts_2d = face_verts[:, ax]          # (M, 2)
        p_2d = np.array([p3[ax[0]], p3[ax[1]]])

        # Sort polygon vertices by angle from centroid (face is convex).
        center_2d = pts_2d.mean(axis=0)
        angles = np.arctan2(pts_2d[:, 1] - center_2d[1],
                            pts_2d[:, 0] - center_2d[0])
        poly = pts_2d[np.argsort(angles)]

        if not _point_in_convex_polygon_2d(p_2d, poly):
            continue

        d_min = _polygon_edge_dist_2d(p_2d, poly)
        stability_margin = float(np.arctan2(d_min, com_height))

        R = _rotation_to_align(n, _neg_z)
        yield R, com_height, stability_margin
