"""TSR visualization using PyVista.

Generic visualizer for the TSR pattern: one reference object + N sampled
subject poses. Decoupled from any specific hand geometry.

Requires the viz optional dependency:
    uv sync --extra viz

Usage::

    from tsr.viz import TSRVisualizer, parallel_jaw_renderer, cylinder_renderer

    visualizer = TSRVisualizer()
    visualizer.render(
        reference_renderer=cylinder_renderer(radius=0.04, height=0.12),
        subject_renderer=parallel_jaw_renderer(finger_length=0.055, half_aperture=0.052),
        poses=poses,
        out="assets/tsr_viz.png",
    )
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Sequence

import numpy as np

try:
    import pyvista as pv
    import matplotlib.cm as cm
    from PIL import Image
except ImportError as e:
    raise ImportError(
        f"TSR visualization requires optional dependencies: {e}\n"
        "Install with: uv sync --extra viz"
    ) from e

# Type alias: a renderer draws one object into a Plotter at a given pose/color.
# reference_renderer: (pl) -> None
# subject_renderer:   (pl, pose_4x4, color) -> None
ReferenceRenderer = Callable[["pv.Plotter"], None]
SubjectRenderer   = Callable[["pv.Plotter", np.ndarray, tuple], None]


class TSRVisualizer:
    """Renders a TSR: one reference object + N sampled subject poses → PNG.

    Args:
        window_size: Output resolution (width, height) in pixels.
        background:  Background hex color.
        camera_dist: Distance from camera to scene center.
        camera_az:   Camera azimuth angle in degrees.
        camera_el:   Camera elevation angle in degrees.
        title:       Optional two-line title overlaid on the image.
        focus:       Scene center for camera look-at (default: origin).
        crop_pad:    Pixels of dark padding to keep when cropping margins.
    """

    def __init__(
        self,
        window_size: tuple = (1600, 1000),
        background: str = "#0d1117",
        camera_dist: float = 0.55,
        camera_az: float = 215.,
        camera_el: float = 25.,
        title: str = "",
        focus: tuple = (0., 0., 0.),
        crop_pad: int = 24,
    ):
        self.window_size = window_size
        self.background  = background
        self.camera_dist = camera_dist
        self.camera_az   = camera_az
        self.camera_el   = camera_el
        self.title       = title
        self.focus       = focus
        self.crop_pad    = crop_pad

    def render(
        self,
        reference_renderer: ReferenceRenderer,
        subject_renderer: SubjectRenderer,
        poses: Sequence[np.ndarray],
        out: str | Path,
        colors: Sequence[tuple] | None = None,
    ) -> None:
        """Render reference + subjects and save to out.

        Args:
            reference_renderer: Callable (pl) -> None. Draws the reference
                object (e.g. a mug cylinder) into the plotter.
            subject_renderer:   Callable (pl, pose_4x4, color) -> None. Draws
                one subject (e.g. a gripper) at the given pose.
            poses:  Sequence of 4x4 SE(3) subject poses to visualize.
            out:    Output PNG path.
            colors: Optional per-pose RGB colors. If None, colors are sampled
                uniformly from the plasma colormap. Pass a list of repeated
                colors to give all poses from the same TSR the same color.
        """
        out = Path(out)
        out.parent.mkdir(exist_ok=True)

        pl = pv.Plotter(off_screen=True, window_size=self.window_size)
        pl.set_background(self.background)

        reference_renderer(pl)

        if colors is None:
            colors = _plasma_colors(len(poses))
        for pose, col in zip(poses, colors):
            subject_renderer(pl, pose, col)

        D  = self.camera_dist
        az = np.radians(self.camera_az)
        el = np.radians(self.camera_el)
        pl.camera_position = [
            (D * np.cos(az) * np.cos(el),
             D * np.sin(az) * np.cos(el),
             D * np.sin(el)),
            self.focus,
            (0., 0., 1.),
        ]

        if self.title:
            pl.add_title(self.title, font_size=11, color="#c9d1d9", font="courier")

        pl.screenshot(str(out), transparent_background=False)
        pl.close()

        _crop_margins(out, self.background, self.crop_pad)


# ── Reference renderers ───────────────────────────────────────────────────────

def cylinder_renderer(
    radius: float,
    height: float,
    color: str = "#1c5f99",
    rim_color: str = "#5aaaf0",
    offset: tuple = (0., 0., 0.),
) -> ReferenceRenderer:
    """Renderer for a solid cylinder (mug, can, bottle, etc.)."""
    ox, oy, oz = offset

    def render(pl: pv.Plotter) -> None:
        cyl = pv.Cylinder(
            center=(ox, oy, oz + height / 2.), direction=(0., 0., 1.),
            radius=radius, height=height, resolution=60, capping=True,
        )
        pl.add_mesh(cyl, color=color, opacity=1.0, smooth_shading=True,
                    lighting=True, specular=0.4, diffuse=0.8, ambient=0.15)

        for z_r in (oz, oz + height):
            th  = np.linspace(0., 2. * np.pi, 120, endpoint=True)
            pts = np.column_stack([radius * np.cos(th) + ox,
                                   radius * np.sin(th) + oy,
                                   np.full(120, z_r)])
            pl.add_mesh(pv.Spline(pts, n_points=120).tube(radius=0.0008),
                        color=rim_color, opacity=0.85, smooth_shading=True, lighting=True)

        for r in np.linspace(radius * 0.3, radius * 0.88, 4):
            th  = np.linspace(0., 2. * np.pi, 80, endpoint=True)
            pts = np.column_stack([r * np.cos(th) + ox,
                                   r * np.sin(th) + oy,
                                   np.full(80, oz + height)])
            pl.add_mesh(pv.Spline(pts, n_points=80).tube(radius=0.0004),
                        color=rim_color, opacity=0.22, lighting=False)

    return render


def box_renderer(
    box_x: float,
    box_y: float,
    box_z: float,
    color: str = "#1c5f99",
    edge_color: str = "#5aaaf0",
    offset: tuple = (0., 0., 0.),
) -> ReferenceRenderer:
    """Renderer for a solid box (cereal box, book, brick, etc.).

    Box coordinate convention: centered in x/y, bottom face at z=0, top at z=box_z.
    """
    ox, oy, oz = offset

    def render(pl: pv.Plotter) -> None:
        box = pv.Box(bounds=(
            ox - box_x / 2., ox + box_x / 2.,
            oy - box_y / 2., oy + box_y / 2.,
            oz,              oz + box_z,
        ))
        pl.add_mesh(box, color=color, opacity=1.0, smooth_shading=True,
                    lighting=True, specular=0.4, diffuse=0.8, ambient=0.15)

        hx, hy, hz = box_x / 2., box_y / 2., box_z
        corners = [
            (-hx, -hy), ( hx, -hy), ( hx,  hy), (-hx,  hy), (-hx, -hy),
        ]
        for z_f in (oz, oz + hz):
            pts = np.array([(ox + x, oy + y, z_f) for x, y in corners])
            pl.add_mesh(pv.Spline(pts, n_points=len(corners)).tube(radius=0.0008),
                        color=edge_color, opacity=0.85, smooth_shading=True, lighting=True)
        for x, y in ((-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)):
            pts = np.array([(ox + x, oy + y, oz), (ox + x, oy + y, oz + hz)])
            pl.add_mesh(pv.Spline(pts, n_points=2).tube(radius=0.0008),
                        color=edge_color, opacity=0.85, smooth_shading=True, lighting=True)

    return render


# ── Subject renderers ─────────────────────────────────────────────────────────

def parallel_jaw_renderer(
    finger_length: float,
    half_aperture: float,
    tube_radius: float = 0.0015,
) -> SubjectRenderer:
    """Renderer for a parallel jaw gripper (GraspGen-style U+stick wireframe).

    Gripper frame convention:
        z = approach (toward object), y = finger opening, x = palm normal

    Args:
        finger_length:  Distance from palm to fingertip [m].
        half_aperture:  Half of jaw opening [m] (= object_radius + safety_gap).
        tube_radius:    Tube thickness for rendering [m].
    """
    FL, APT = finger_length, half_aperture
    segments = [
        (np.array([0., -APT, 0.]), np.array([0.,  APT, 0.])),  # palm crossbar
        (np.array([0., -APT, 0.]), np.array([0., -APT, FL])),  # left finger
        (np.array([0.,  APT, 0.]), np.array([0.,  APT, FL])),  # right finger
        (np.array([0.,  0.,  0.]), np.array([0.,  0., -FL])),  # approach stick
    ]

    def render(pl: pv.Plotter, pose_4x4: np.ndarray, color: tuple,
               opacity: float = 0.92) -> None:
        R, t = pose_4x4[:3, :3], pose_4x4[:3, 3]
        for p_l, q_l in segments:
            p_w, q_w = t + R @ p_l, t + R @ q_l
            pts  = np.vstack([p_w, (p_w + q_w) / 2, q_w])
            tube = pv.Spline(pts, n_points=3).tube(radius=tube_radius, n_sides=8)
            pl.add_mesh(tube, color=color, opacity=opacity,
                        smooth_shading=True, lighting=True)

    return render


# ── Helpers ───────────────────────────────────────────────────────────────────

def plasma_colors(n: int, lo: float = 0.05, hi: float = 0.92) -> List[tuple]:
    """N RGB colors sampled uniformly from the plasma colormap."""
    return [cm.plasma(v)[:3] for v in np.linspace(lo, hi, n)]


# Keep private alias for internal use
_plasma_colors = plasma_colors


def _crop_margins(path: Path, background_hex: str, pad: int) -> None:
    """Crop dark background margins from a PNG, keeping pad pixels of border."""
    r = int(background_hex[1:3], 16)
    g = int(background_hex[3:5], 16)
    b = int(background_hex[5:7], 16)
    bg  = np.array([r, g, b])

    img  = Image.open(path).convert("RGB")
    arr  = np.array(img)
    mask = np.any(np.abs(arr.astype(int) - bg) > 12, axis=2)
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    r0, r1 = max(rows[0] - pad, 0), min(rows[-1] + pad, arr.shape[0])
    c0, c1 = max(cols[0] - pad, 0), min(cols[-1] + pad, arr.shape[1])
    img.crop((c0, r0, c1, r1)).save(path)
