# SPDX-License-Identifier: BSD-2-Clause
# Authors: Siddhartha Srinivasa and contributors to TSR

"""Experimental interactive viewer for grasp templates, backed by Viser (issue #77).

This is an **evaluation** backend, not a replacement for :mod:`tsr.viz`: it is under
assessment for whether an interactive browser view is a better way to inspect grasp
poses and the continuous freedoms of a TSR than the existing static PyVista renderer.
Install it with ``pip install "sstsr[viser]"``.

Visualization is explanatory and diagnostic. It is never part of the correctness
argument for a template — that is the analytic oracle's job (docs/ARCHITECTURE.md).

Nothing here is imported by core sstsr, and importing this module does not start a
server. Server lifetime is the caller's: pass one in, or take the one returned and
``stop()`` it.

Usage::

    from tsr.hands import ParallelJawGripper
    from tsr.viser import show_templates

    gripper = ParallelJawGripper(finger_length=0.08, max_aperture=0.14)
    templates = gripper.grasp_cylinder_side(0.03, 0.12)
    server = show_templates(templates, cylinder=(0.03, 0.12), gripper=gripper, seed=0)
    server.sleep_forever()  # Ctrl-C to stop
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from .template import TSRTemplate

try:
    import viser as _viser
except ImportError as e:  # pragma: no cover - exercised by the extra being absent
    raise ImportError(
        f"The experimental Viser viewer requires optional dependencies: {e}\n"
        'Install with: pip install "sstsr[viser]"  (or: uv sync --extra viser)'
    ) from e

__all__ = ["show_templates", "sample_poses", "gripper_segments"]

_AXIS_COLORS = ((255, 80, 80), (80, 220, 80), (90, 140, 255))  # x, y, z_EE = approach


def _wxyz(R: np.ndarray) -> np.ndarray:
    """Rotation matrix → ``(w, x, y, z)`` quaternion, Viser's orientation convention."""
    m = np.asarray(R, dtype=float)
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        q = np.array([0.25 / s, (m[2, 1] - m[1, 2]) * s, (m[0, 2] - m[2, 0]) * s, (m[1, 0] - m[0, 1]) * s])
    else:
        i = int(np.argmax([m[0, 0], m[1, 1], m[2, 2]]))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * np.sqrt(max(1e-12, 1.0 + m[i, i] - m[j, j] - m[k, k]))
        q = np.empty(4)
        q[0] = (m[k, j] - m[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (m[j, i] + m[i, j]) / s
        q[1 + k] = (m[k, i] + m[i, k]) / s
    return q / np.linalg.norm(q)


def sample_poses(
    templates: Sequence[TSRTemplate],
    n_per_template: int = 4,
    *,
    T_ref_world: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> List[np.ndarray]:
    """Reproducible end-effector poses from ``templates``.

    Each template is instantiated at ``T_ref_world`` (identity by default) and sampled
    ``n_per_template`` times. Sampling is driven by an explicit ``rng`` or ``seed``, so
    a view can be reproduced exactly; with neither, it is nondeterministic by design.
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    reference = np.eye(4) if T_ref_world is None else np.asarray(T_ref_world, dtype=float)
    poses: List[np.ndarray] = []
    for template in templates:
        tsr = template.instantiate(reference)
        poses.extend(tsr.sample(rng=rng) for _ in range(n_per_template))
    return poses


def gripper_segments(finger_length: float, aperture: float) -> np.ndarray:
    """Line segments of an idealized parallel jaw, in the end-effector frame.

    The library convention (docs/ARCHITECTURE.md): ``z_EE`` approaches the object,
    ``y_EE`` is the finger-opening direction, ``x_EE = y_EE × z_EE``. Returns an
    ``(S, 2, 3)`` array of segment endpoints: the palm crossbar, the two fingers, and
    a short approach stick behind the palm.
    """
    half, fl = aperture / 2.0, finger_length
    return np.array(
        [
            [[0.0, -half, 0.0], [0.0, half, 0.0]],  # palm crossbar
            [[0.0, -half, 0.0], [0.0, -half, fl]],  # finger, -y side
            [[0.0, half, 0.0], [0.0, half, fl]],  # finger, +y side
            [[0.0, 0.0, 0.0], [0.0, 0.0, -0.4 * fl]],  # approach stick behind the palm
        ],
        dtype=float,
    )


def _posed_segments(segments: np.ndarray, poses: Sequence[np.ndarray]) -> np.ndarray:
    """Broadcast ``(S, 2, 3)`` local segments through ``poses`` → ``(N*S, 2, 3)`` world."""
    out = np.empty((len(poses) * len(segments), 2, 3), dtype=float)
    for i, pose in enumerate(poses):
        R, t = pose[:3, :3], pose[:3, 3]
        out[i * len(segments) : (i + 1) * len(segments)] = segments @ R.T + t
    return out


def show_templates(
    templates: Sequence[TSRTemplate],
    *,
    cylinder: Optional[Tuple[float, float]] = None,
    gripper=None,
    n_per_template: int = 4,
    seed: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    server: Optional["_viser.ViserServer"] = None,
    port: int = 8080,
    axes_length: float = 0.02,
    name: str = "grasp",
) -> "_viser.ViserServer":
    """Show a reference object, sampled end-effector poses, and the gripper geometry.

    Args:
        templates: Templates to sample. They are used read-only; nothing here is
            attached to ``TSRTemplate`` or to the grasp factories.
        cylinder: ``(radius, height)`` of the reference cylinder to draw, in the
            template's reference frame (the cylinder convention is ``z ∈ [0, height]``).
        gripper: Any object exposing ``finger_length`` / ``max_aperture``; the jaw
            drawing uses the template's own preshape for the opening when present.
        n_per_template: Poses sampled per template.
        seed / rng: Reproducible sampling; see :func:`sample_poses`.
        server: An existing server to draw into. When omitted a new one is created on
            ``port`` and **returned** — the caller owns it and should ``stop()`` it.
        axes_length: Length of the per-pose end-effector axes [m].
        name: Scene-node prefix, so several calls can share one server.

    Returns:
        The server that was drawn into (created here when ``server`` is None).
    """
    poses = sample_poses(templates, n_per_template, seed=seed, rng=rng)
    server = _viser.ViserServer(port=port) if server is None else server
    scene = server.scene

    if cylinder is not None:
        radius, height = cylinder
        # sstsr cylinders span z ∈ [0, height]; Viser's cylinder is centred on its origin.
        scene.add_cylinder(
            f"/{name}/reference",
            radius=radius,
            height=height,
            position=(0.0, 0.0, height / 2.0),
            color=(170, 170, 178),
            opacity=0.55,
        )
    scene.add_frame(f"/{name}/reference_frame", axes_length=max(axes_length * 2, 0.04), axes_radius=0.0015)

    if poses:
        positions = np.array([p[:3, 3] for p in poses])
        orientations = np.array([_wxyz(p[:3, :3]) for p in poses])
        # One batched node for every pose's canonical EE axes: x red, y green, z blue,
        # so approach (z) and jaw opening (y) are readable at a glance.
        scene.add_batched_axes(
            f"/{name}/ee_axes",
            batched_wxyzs=orientations,
            batched_positions=positions,
            axes_length=axes_length,
            axes_radius=axes_length / 12.0,
        )
        if gripper is not None:
            aperture = float(templates[0].preshape[0]) if templates[0].preshape is not None else gripper.max_aperture
            segments = gripper_segments(gripper.finger_length, aperture)
            # A single line-segments node for every jaw, rather than one node per pose:
            # the browser then has one object to update instead of N.
            scene.add_line_segments(
                f"/{name}/jaws",
                points=_posed_segments(segments, poses),
                colors=np.broadcast_to(np.array([60, 60, 70]), (len(poses) * len(segments), 2, 3)),
                thickness=1.6,
                thickness_units="screen",
            )
    return server
