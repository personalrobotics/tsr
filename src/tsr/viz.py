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
    from matplotlib.colors import ListedColormap
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
        parallel_projection: bool = False,
    ):
        self.window_size         = window_size
        self.background          = background
        self.camera_dist         = camera_dist
        self.camera_az           = camera_az
        self.camera_el           = camera_el
        self.title               = title
        self.focus               = focus
        self.crop_pad            = crop_pad
        self.parallel_projection = parallel_projection

    def render(
        self,
        reference_renderer: ReferenceRenderer,
        subject_renderer: SubjectRenderer,
        poses: Sequence[np.ndarray],
        out: str | Path,
        colors: Sequence[tuple] | None = None,
    ) -> None:
        """Render one subject renderer at multiple poses and save to out."""
        if colors is None:
            colors = _plasma_colors(len(poses))
        subjects = [(subject_renderer, pose, col)
                    for pose, col in zip(poses, colors)]
        self.render_multi(reference_renderer, subjects, out)

    def render_multi(
        self,
        reference_renderer: ReferenceRenderer,
        subjects: Sequence[tuple],
        out: str | Path,
    ) -> None:
        """Render multiple (renderer, pose, color) subjects and save to out.

        Args:
            reference_renderer: Callable (pl) -> None. Draws the reference
                object (e.g. a table surface) into the plotter.
            subjects: Sequence of (subject_renderer, pose_4x4, color) triples.
                Each triple draws one object at its pose with the given color.
            out: Output PNG path.
        """
        out = Path(out)
        out.parent.mkdir(exist_ok=True)

        pl = pv.Plotter(off_screen=True, window_size=self.window_size)
        pl.set_background(self.background)
        pl.hide_axes()

        reference_renderer(pl)

        for renderer, pose, color in subjects:
            renderer(pl, pose, color)

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

        if self.parallel_projection:
            pl.enable_parallel_projection()

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


def sphere_renderer(
    radius: float,
    color: str = "#1c5f99",
    rim_color: str = "#5aaaf0",
    offset: tuple = (0., 0., 0.),
) -> ReferenceRenderer:
    """Renderer for a solid sphere.

    Sphere coordinate convention: center at origin (offset shifts the center).
    """
    ox, oy, oz = offset

    def render(pl: pv.Plotter) -> None:
        sph = pv.Sphere(radius=radius, center=(ox, oy, oz), theta_resolution=60,
                        phi_resolution=60)
        pl.add_mesh(sph, color=color, opacity=1.0, smooth_shading=True,
                    lighting=True, specular=0.6, diffuse=0.8, ambient=0.15)

        # Equatorial ring
        th  = np.linspace(0., 2. * np.pi, 120, endpoint=True)
        pts = np.column_stack([radius * np.cos(th) + ox,
                               radius * np.sin(th) + oy,
                               np.full(120, oz)])
        pl.add_mesh(pv.Spline(pts, n_points=120).tube(radius=0.0008),
                    color=rim_color, opacity=0.85, smooth_shading=True, lighting=True)

    return render


def torus_renderer(
    torus_radius: float,
    tube_radius: float,
    color: str = "#1c5f99",
    rim_color: str = "#5aaaf0",
    offset: tuple = (0., 0., 0.),
) -> ReferenceRenderer:
    """Renderer for a solid torus (ring/donut shape).

    Torus coordinate convention: center at origin, axis along +z.
    torus_radius R: distance from center to tube center.
    tube_radius  r: cross-section radius of the tube.
    """
    ox, oy, oz = offset

    def render(pl: pv.Plotter) -> None:
        # Build torus surface via parametric sweep
        N_phi, N_theta = 80, 40
        phi   = np.linspace(0., 2. * np.pi, N_phi,   endpoint=False)  # around ring
        theta = np.linspace(0., 2. * np.pi, N_theta, endpoint=False)  # around tube
        PHI, THETA = np.meshgrid(phi, theta, indexing='ij')

        X = (torus_radius + tube_radius * np.cos(THETA)) * np.cos(PHI) + ox
        Y = (torus_radius + tube_radius * np.cos(THETA)) * np.sin(PHI) + oy
        Z = tube_radius * np.sin(THETA) + oz

        # Build StructuredGrid
        pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        grid = pv.StructuredGrid()
        grid.points = pts
        grid.dimensions = (N_phi, N_theta, 1)
        surface = grid.extract_surface()
        pl.add_mesh(surface, color=color, opacity=1.0, smooth_shading=True,
                    lighting=True, specular=0.4, diffuse=0.8, ambient=0.15)

        # Outer and inner rim rings
        th = np.linspace(0., 2. * np.pi, 120, endpoint=True)
        for r_ring in (torus_radius + tube_radius, torus_radius - tube_radius):
            pts_ring = np.column_stack([r_ring * np.cos(th) + ox,
                                        r_ring * np.sin(th) + oy,
                                        np.full(120, oz)])
            pl.add_mesh(pv.Spline(pts_ring, n_points=120).tube(radius=0.0008),
                        color=rim_color, opacity=0.85, smooth_shading=True, lighting=True)

    return render


# ── Placement renderers ───────────────────────────────────────────────────────

def table_surface_renderer(
    table_x: float,
    table_y: float,
    color: str = "#2d333b",
    grid_color: str = "#3d4755",
    grid_spacing: float = 0.05,
) -> ReferenceRenderer:
    """Renderer for a flat table surface with a subtle grid (reference for placement TSRs).

    table_x, table_y: half-extents of the table surface [m].
    grid_spacing: distance between grid lines [m].
    """
    def render(pl: pv.Plotter) -> None:
        # Flat plane at z=0 — no slab, avoids bottom/side face artifacts.
        top = pv.Plane(
            center=(0., 0., 0.), direction=(0., 0., 1.),
            i_size=2 * table_x, j_size=2 * table_y,
            i_resolution=1, j_resolution=1,
        )
        pl.add_mesh(top, color=color, opacity=1.0,
                    lighting=True, specular=0.05, diffuse=0.8, ambient=0.3)

        EPS = 0.0001  # lift grid lines above surface to avoid z-fighting

        # Grid lines along x (constant y)
        for y in np.arange(-table_y, table_y + 1e-9, grid_spacing):
            pts = np.array([(-table_x, y, EPS), (table_x, y, EPS)])
            pl.add_mesh(pv.Spline(pts, n_points=2).tube(radius=0.0005),
                        color=grid_color, opacity=0.6, lighting=False)

        # Grid lines along y (constant x)
        for x in np.arange(-table_x, table_x + 1e-9, grid_spacing):
            pts = np.array([(x, -table_y, EPS), (x, table_y, EPS)])
            pl.add_mesh(pv.Spline(pts, n_points=2).tube(radius=0.0005),
                        color=grid_color, opacity=0.6, lighting=False)

        # Border
        corners = [(-table_x, -table_y), (table_x, -table_y),
                   (table_x,  table_y),  (-table_x,  table_y),
                   (-table_x, -table_y)]
        pts = np.array([(x, y, EPS) for x, y in corners])
        pl.add_mesh(pv.Spline(pts, n_points=len(corners)).tube(radius=0.002),
                    color=grid_color, opacity=0.8, lighting=False)

    return render


def placed_box_renderer(
    lx: float,
    ly: float,
    lz: float,
) -> SubjectRenderer:
    """Subject renderer for a placed box with per-face coloring.

    Each of the 6 faces renders in a distinct color; the ``color`` argument is
    unused (the palette is fixed).  Face order: -z, +z, -y, +y, -x, +x.
    """
    _FACE_COLORS = [
        (0.122, 0.467, 0.706),  # blue   – -z (bottom)
        (1.000, 0.498, 0.055),  # orange – +z (top)
        (0.173, 0.627, 0.173),  # green  – -y
        (0.839, 0.153, 0.157),  # red    – +y
        (0.580, 0.404, 0.741),  # purple – -x
        (0.549, 0.337, 0.294),  # brown  – +x
    ]
    _cmap = ListedColormap(_FACE_COLORS)

    # Pre-compute which cell → which face ID (stable across equal-bounds boxes).
    _box0 = pv.Box(bounds=(-lx/2, lx/2, -ly/2, ly/2, -lz/2, lz/2))
    _normals = _box0.compute_normals(cell_normals=True,
                                     point_normals=False).cell_data["Normals"]
    _AX_OFFSET = {2: 0, 1: 2, 0: 4}  # axis → base face-id  (-=+0, +=+1)
    _fids = np.array([
        _AX_OFFSET[int(np.argmax(np.abs(n)))] + (1 if n[int(np.argmax(np.abs(n)))] > 0 else 0)
        for n in _normals
    ], dtype=float)

    def render(pl: pv.Plotter, pose_4x4: np.ndarray, _color: tuple) -> None:
        R, t = pose_4x4[:3, :3], pose_4x4[:3, 3]
        box = pv.Box(bounds=(-lx/2, lx/2, -ly/2, ly/2, -lz/2, lz/2))
        box.points = (R @ box.points.T).T + t
        box.cell_data["face_id"] = _fids
        pl.add_mesh(box, scalars="face_id", cmap=_cmap, clim=(-0.5, 5.5),
                    n_colors=6, show_scalar_bar=False,
                    smooth_shading=True, lighting=True,
                    specular=0.4, diffuse=0.8, ambient=0.15,
                    show_edges=True, edge_color="#e0e0e0", line_width=1.2)

    return render


def placed_cylinder_renderer(
    radius: float,
    height: float,
) -> SubjectRenderer:
    """Subject renderer for a placed cylinder with per-region coloring.

    Bottom cap, curved body, and top cap each render in a distinct color.
    The ``color`` argument is unused (the palette is fixed).
    """
    _COLORS = [
        (0.620, 0.620, 0.620),  # gray   – curved body
        (1.000, 0.498, 0.055),  # orange – top cap  (+z)
        (0.122, 0.467, 0.706),  # blue   – bottom cap (-z)
    ]
    _cmap = ListedColormap(_COLORS)

    def render(pl: pv.Plotter, pose_4x4: np.ndarray, _color: tuple) -> None:
        R, t = pose_4x4[:3, :3], pose_4x4[:3, 3]
        axis = R @ np.array([0., 0., 1.])
        cyl = pv.Cylinder(center=t, direction=axis,
                          radius=radius, height=height, resolution=60, capping=True)
        cyl = cyl.compute_normals(cell_normals=True, point_normals=False)
        dot = cyl.cell_data["Normals"] @ axis
        cyl.cell_data["region"] = np.where(dot > 0.8, 1.0,
                                           np.where(dot < -0.8, 2.0, 0.0))
        pl.add_mesh(cyl, scalars="region", cmap=_cmap, clim=(-0.5, 2.5),
                    n_colors=3, show_scalar_bar=False,
                    smooth_shading=True, lighting=True,
                    specular=0.4, diffuse=0.8, ambient=0.15,
                    show_edges=True, edge_color="#e0e0e0", line_width=0.8)

    return render


def placed_sphere_renderer(radius: float) -> SubjectRenderer:
    """Subject renderer for a placed sphere.

    Sphere frame: origin at center.
    """
    def render(pl: pv.Plotter, pose_4x4: np.ndarray, color: tuple) -> None:
        t = pose_4x4[:3, 3]
        sph = pv.Sphere(radius=radius, center=t,
                        theta_resolution=40, phi_resolution=40)
        pl.add_mesh(sph, color=color, opacity=1.0, smooth_shading=True,
                    lighting=True, specular=0.5, diffuse=0.8, ambient=0.15)

    return render


def placed_torus_renderer(major_radius: float, minor_radius: float) -> SubjectRenderer:
    """Subject renderer for a placed torus with per-half coloring.

    The upper half (local z ≥ 0) and lower half (local z < 0) render in
    distinct colors.  The ``color`` argument is unused (the palette is fixed).
    """
    N_phi, N_theta = 60, 24
    _COLORS = [
        (0.839, 0.153, 0.157),  # red   – bottom half (z < 0)
        (0.173, 0.627, 0.173),  # green – top half    (z ≥ 0)
    ]
    _cmap = ListedColormap(_COLORS)

    def render(pl: pv.Plotter, pose_4x4: np.ndarray, _color: tuple) -> None:
        R, t = pose_4x4[:3, :3], pose_4x4[:3, 3]
        phi   = np.linspace(0., 2. * np.pi, N_phi,   endpoint=False)
        theta = np.linspace(0., 2. * np.pi, N_theta, endpoint=False)
        PHI, THETA = np.meshgrid(phi, theta, indexing='ij')
        X = (major_radius + minor_radius * np.cos(THETA)) * np.cos(PHI)
        Y = (major_radius + minor_radius * np.cos(THETA)) * np.sin(PHI)
        Z = minor_radius * np.sin(THETA)

        # Build surface in local frame to classify top/bottom half by z.
        grid = pv.StructuredGrid()
        grid.points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        grid.dimensions = (N_phi, N_theta, 1)
        surface = grid.extract_surface()
        centers_z = surface.cell_centers().points[:, 2]
        surface.cell_data["half"] = (centers_z >= 0).astype(float)

        # Apply pose transform to points only (cell data stays valid).
        surface.points = (R @ surface.points.T).T + t
        pl.add_mesh(surface, scalars="half", cmap=_cmap, clim=(-0.5, 1.5),
                    n_colors=2, show_scalar_bar=False,
                    smooth_shading=True, lighting=True,
                    specular=0.4, diffuse=0.8, ambient=0.15)

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
    if rows.size == 0 or cols.size == 0:
        return  # nothing to crop — image is uniform background
    r0, r1 = max(rows[0] - pad, 0), min(rows[-1] + pad, arr.shape[0])
    c0, c1 = max(cols[0] - pad, 0), min(cols[-1] + pad, arr.shape[1])
    img.crop((c0, r0, c1, r1)).save(path)
