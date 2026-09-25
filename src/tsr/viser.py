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

__all__ = [
    "show_templates",
    "explore_templates",
    "free_coordinates",
    "sample_poses",
    "gripper_segments",
    "add_primitive",
    "torus_mesh",
]

# Bw row order, with the unit each coordinate is measured in.
_COORDS = (("x", "m"), ("y", "m"), ("z", "m"), ("roll", "rad"), ("pitch", "rad"), ("yaw", "rad"))

_AXIS_COLORS = ((255, 80, 80), (80, 220, 80), (90, 140, 255))  # x, y, z_EE = approach

# Jaws are coloured by depth index (shallow -> deep), so the discrete depth levels of a
# family separate visually instead of merging into one grey thicket.
_DEPTH_COLORS = np.array([(13, 8, 135), (126, 3, 168), (204, 71, 120), (248, 149, 64), (240, 249, 33)], dtype=float)


def _depth_color(index: int, count: int) -> np.ndarray:
    if count <= 1:
        return _DEPTH_COLORS[0]
    t = (index / (count - 1)) * (len(_DEPTH_COLORS) - 1)
    lo = int(np.floor(t))
    hi = min(lo + 1, len(_DEPTH_COLORS) - 1)
    return _DEPTH_COLORS[lo] + (t - lo) * (_DEPTH_COLORS[hi] - _DEPTH_COLORS[lo])


def torus_mesh(major_radius: float, minor_radius: float, *, major_segments: int = 64, minor_segments: int = 24):
    """``(vertices, faces)`` for a torus about +z, centred at the origin.

    Viser has native cylinder, box and sphere primitives but no torus, so this is the
    one primitive sstsr has to tessellate itself. The convention matches the grasp
    factories: ``R`` is the ring radius, ``r`` the tube radius.
    """
    u = np.linspace(0.0, 2.0 * np.pi, major_segments, endpoint=False)
    v = np.linspace(0.0, 2.0 * np.pi, minor_segments, endpoint=False)
    uu, vv = np.meshgrid(u, v, indexing="ij")
    ring = major_radius + minor_radius * np.cos(vv)
    vertices = np.stack([ring * np.cos(uu), ring * np.sin(uu), minor_radius * np.sin(vv)], axis=-1)
    vertices = vertices.reshape(-1, 3)

    i, j = np.meshgrid(np.arange(major_segments), np.arange(minor_segments), indexing="ij")
    a = (i * minor_segments + j).ravel()
    b = (((i + 1) % major_segments) * minor_segments + j).ravel()
    c = (((i + 1) % major_segments) * minor_segments + (j + 1) % minor_segments).ravel()
    d = (i * minor_segments + (j + 1) % minor_segments).ravel()
    faces = np.concatenate([np.stack([a, b, c], axis=-1), np.stack([a, c, d], axis=-1)])
    return vertices, faces.astype(np.uint32)


def add_primitive(
    scene,
    name: str,
    kind: str,
    dims: Sequence[float],
    *,
    pose: Optional[np.ndarray] = None,
    centered: bool = False,
    color: Tuple[int, int, int] = (170, 170, 178),
    opacity: Optional[float] = None,
):
    """Draw one sstsr primitive, in **its own** frame convention.

    The frames are the ones sstsr uses, not Viser's, so a caller never has to remember
    which is which.

    The two conventions in the library differ, and confusing them shifts an object by
    half its height:

    * **grasp factories** put a cylinder and a box at ``z ∈ [0, height]``, centred in x
      and y (``centered=False``, the default);
    * **placement templates** (:class:`~tsr.placement.StablePlacer`) give the pose of
      the object's **centre**, so pass ``centered=True`` when drawing a placed object.

    A sphere and a torus are centred either way.

    Args:
        kind: ``"cylinder"`` (radius, height), ``"box"`` (x, y, z), ``"sphere"``
            (radius) or ``"torus"`` (major, minor).
        pose: 4x4 placement of the primitive's frame; identity by default.
        centered: Whether ``pose`` locates the object's centre (placements) rather than
            the factory frame (grasps).
    """
    pose = np.eye(4) if pose is None else np.asarray(pose, dtype=float)
    shared = dict(wxyz=_wxyz(pose[:3, :3]), color=color, opacity=opacity)

    def placed(local_offset):
        return pose[:3, 3] + pose[:3, :3] @ np.asarray(local_offset, dtype=float)

    if kind == "cylinder":
        radius, height = dims
        rise = 0.0 if centered else height / 2
        return scene.add_cylinder(name, radius=radius, height=height, position=placed((0, 0, rise)), **shared)
    if kind == "box":
        bx, by, bz = dims
        rise = 0.0 if centered else bz / 2
        return scene.add_box(name, dimensions=(bx, by, bz), position=placed((0, 0, rise)), **shared)
    if kind == "sphere":
        return scene.add_icosphere(name, radius=dims[0], position=placed((0, 0, 0)), **shared)
    if kind == "torus":
        vertices, faces = torus_mesh(dims[0], dims[1])
        return scene.add_mesh_simple(name, vertices=vertices, faces=faces, position=placed((0, 0, 0)), **shared)
    raise ValueError(f"unknown primitive {kind!r}; expected cylinder, box, sphere or torus")


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
    n_per_template: int = 2,
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


def free_coordinates(template: TSRTemplate) -> List[Tuple[int, str, float, float]]:
    """The template's non-degenerate Bw rows as ``(row, label, lo, hi)``.

    These are the continuous freedoms a template actually has — the coordinates worth
    putting a slider on. A cylinder side grasp, for example, is free in ``z`` (slide
    along the axis) and ``yaw`` (all the way around) and fixed in the other four.
    """
    out = []
    for row, (name, unit) in enumerate(_COORDS):
        lo, hi = float(template.Bw[row, 0]), float(template.Bw[row, 1])
        if hi > lo:
            out.append((row, f"{name} [{unit}]", lo, hi))
    return out


def _template_label(index: int, template: TSRTemplate) -> str:
    p = template.provenance
    return f"{index}: {p.mode}/{p.approach} depth {p.depth * 1000:.0f}mm ({p.depth_index + 1}/{p.depth_count})"


def explore_templates(
    templates: Sequence[TSRTemplate],
    *,
    cylinder: Optional[Tuple[float, float]] = None,
    gripper=None,
    T_ref_world: Optional[np.ndarray] = None,
    server: Optional["_viser.ViserServer"] = None,
    port: int = 8080,
    axes_length: float = 0.03,
    name: str = "explore",
) -> "_viser.ViserServer":
    """Scrub one grasp through a template's continuous freedoms with GUI sliders.

    Where :func:`show_templates` draws a static cloud of sampled poses, this drives a
    **single** grasp from one slider per free Bw coordinate, plus a selector for which
    template in the family to inspect. Moving a slider re-evaluates
    ``tsr.to_transform(ξ)`` and updates the node transforms only — the jaw geometry is
    uploaded once per template, not per frame.

    The sampled cloud is available too, behind a checkbox, so a family's represented
    set and one concrete pose within it can be compared directly.
    """
    reference = np.eye(4) if T_ref_world is None else np.asarray(T_ref_world, dtype=float)
    server = _viser.ViserServer(port=port) if server is None else server
    scene, gui = server.scene, server.gui

    _draw_reference(scene, name, cylinder, axes_length=max(axes_length, 0.04))

    ee_axes = scene.add_frame(f"/{name}/ee", axes_length=axes_length, axes_radius=axes_length / 18.0)
    jaw = None  # rebuilt whenever the selected template changes (its preshape may differ)
    sliders: List = []
    folder = None

    selector = gui.add_dropdown(
        "Template", tuple(_template_label(i, t) for i, t in enumerate(templates)), initial_value=None
    )
    show_cloud = gui.add_checkbox("Show sampled poses", False)
    readout = gui.add_text("Pose", initial_value="", disabled=True)

    # Drawn once and toggled by the checkbox, so the represented set and one concrete
    # pose within it can be compared without rebuilding anything.
    cloud = _draw_pose_cloud(
        scene,
        templates,
        f"{name}/cloud",
        gripper=gripper,
        n_per_template=2,
        seed=0,
        rng=None,
        axes_length=axes_length / 2.5,
    )

    def _cloud_visible(visible: bool) -> None:
        for handle in cloud:
            handle.visible = visible

    def _selected() -> TSRTemplate:
        return templates[int(selector.value.split(":")[0])]

    def _apply(_=None) -> None:
        template = _selected()
        xi = (template.Bw[:, 0] + template.Bw[:, 1]) / 2.0
        for (row, _label, _lo, _hi), slider in zip(free_coordinates(template), sliders):
            xi[row] = slider.value
        pose = template.instantiate(reference).to_transform(xi)
        ee_axes.position, ee_axes.wxyz = pose[:3, 3], _wxyz(pose[:3, :3])
        if jaw is not None:
            jaw.position, jaw.wxyz = pose[:3, 3], _wxyz(pose[:3, :3])
        free = ", ".join(f"{lab.split(' ')[0]}={xi[row]:+.3f}" for row, lab, _, _ in free_coordinates(template))
        readout.value = free or "(no continuous freedom)"

    def _rebuild(_=None) -> None:
        nonlocal jaw, folder, sliders
        template = _selected()
        if folder is not None:
            folder.remove()
        sliders = []
        folder = gui.add_folder("TSR coordinates")
        with folder:
            for row, label, lo, hi in free_coordinates(template):
                slider = gui.add_slider(label, min=lo, max=hi, step=(hi - lo) / 200.0, initial_value=(lo + hi) / 2.0)
                slider.on_update(_apply)
                sliders.append(slider)
        if gripper is not None:
            aperture = float(template.preshape[0]) if template.preshape is not None else gripper.max_aperture
            if jaw is not None:
                jaw.remove()
            jaw = scene.add_line_segments(
                f"/{name}/jaw",
                points=gripper_segments(gripper.finger_length, aperture),
                colors=np.broadcast_to(np.array([20, 20, 30], dtype=np.uint8), (4, 2, 3)),
                thickness=0.003,
                thickness_units="world",
            )
        _apply()

    selector.on_update(_rebuild)
    show_cloud.on_update(lambda _: _cloud_visible(show_cloud.value))
    _rebuild()
    _cloud_visible(False)
    return server


def show_templates(
    templates: Sequence[TSRTemplate],
    *,
    cylinder: Optional[Tuple[float, float]] = None,
    gripper=None,
    n_per_template: int = 2,
    seed: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    server: Optional["_viser.ViserServer"] = None,
    port: int = 8080,
    axes_length: float = 0.014,
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
    server = _viser.ViserServer(port=port) if server is None else server
    _draw_reference(server.scene, name, cylinder, axes_length=max(axes_length * 2, 0.04))
    _draw_pose_cloud(
        server.scene,
        templates,
        name,
        gripper=gripper,
        n_per_template=n_per_template,
        seed=seed,
        rng=rng,
        axes_length=axes_length,
    )
    return server


def _draw_reference(scene, name: str, cylinder: Optional[Tuple[float, float]], *, axes_length: float) -> None:
    """The reference object and its frame, in the template's reference frame."""
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
    scene.add_frame(f"/{name}/reference_frame", axes_length=axes_length, axes_radius=0.0015)


def _draw_pose_cloud(
    scene,
    templates: Sequence[TSRTemplate],
    name: str,
    *,
    gripper,
    n_per_template: int,
    seed: Optional[int],
    rng: Optional[np.random.Generator],
    axes_length: float,
) -> List:
    """Sampled poses as **two** scene nodes, and return their handles.

    Batching matters: one node carries every pose's axes and one carries every jaw, so
    the browser updates two objects rather than N, and a caller (the explorer) can show
    or hide the whole cloud with two assignments.
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    # Keep each pose's originating template, so jaws can be coloured by depth.
    per_template = [sample_poses([t], n_per_template, rng=rng) for t in templates]
    poses = [pose for group in per_template for pose in group]
    if not poses:
        return []

    handles = [
        # x red, y green, z blue, so approach (z) and jaw opening (y) read at a glance.
        scene.add_batched_axes(
            f"/{name}/ee_axes",
            batched_wxyzs=np.array([_wxyz(p[:3, :3]) for p in poses]),
            batched_positions=np.array([p[:3, 3] for p in poses]),
            axes_length=axes_length,
            axes_radius=axes_length / 22.0,
        )
    ]
    if gripper is not None:
        points, colors = [], []
        for template, group in zip(templates, per_template):
            aperture = float(template.preshape[0]) if template.preshape is not None else gripper.max_aperture
            segments = gripper_segments(gripper.finger_length, aperture)
            points.append(_posed_segments(segments, group))
            color = _depth_color(template.provenance.depth_index, template.provenance.depth_count)
            colors.append(np.broadcast_to(color, (len(group) * len(segments), 2, 3)))
        # World-space thickness keeps the jaws readable as solid geometry, not hairlines.
        handles.append(
            scene.add_line_segments(
                f"/{name}/jaws",
                points=np.concatenate(points),
                colors=np.concatenate(colors).astype(np.uint8),
                thickness=0.0022,
                thickness_units="world",
            )
        )
    return handles
