"""mesh_placements.py — Stable placement TSRs for non-convex meshes.

Three shapes that demonstrate ``place_mesh`` on arbitrary vertex clouds:
  L-shape  – asymmetric; unstable faces filtered by COM-projection test
  T-shape  – symmetric bar atop a centred stem; tests marginal stability
  Mug      – cylinder body + box handle, COM shifted laterally off the axis

Usage::

    uv run python examples/mesh_placements.py
    uv sync --extra viz && uv run python examples/mesh_placements.py
"""
import numpy as np

from tsr.placement import TablePlacer

placer     = TablePlacer(table_x=0.60, table_y=0.40)
table_pose = np.eye(4)
table_pose[2, 3] = 0.75

print("=== Mesh Placement Demo ===\n")


# ── helpers ──────────────────────────────────────────────────────────────────

def _extrude_xz(xz_pts, depth):
    """Extrude a 2D xz silhouette into 3D by adding y=0 and y=depth layers."""
    return np.array([[x, y, z] for y in [0.0, depth] for x, z in xz_pts])


def _weighted_com(*parts):
    """Volume-weighted COM from (centre_3d, volume) pairs."""
    total = sum(v for _, v in parts)
    return sum(c * v for c, v in parts) / total


def _margin(tmpl):
    return tmpl.name.split("margin ")[1].rstrip(")")


# ── L-shape ──────────────────────────────────────────────────────────────────
#
#   lz_total │ ┌──────┐
#            │ │ stem │
#   lz_base  │ │      ├──────────────────┐
#            │ │      │   base arm       │
#            └─┴──────┴──────────────────┘
#              0    lx_stem           lx_total

L_LX, L_LX_STEM = 0.10, 0.04
L_LY             = 0.04
L_LZ, L_LZ_BASE  = 0.12, 0.04

l_verts = _extrude_xz([
    (0,          0),         (L_LX,       0),
    (L_LX,       L_LZ_BASE), (L_LX_STEM,  L_LZ_BASE),
    (L_LX_STEM,  L_LZ),      (0,          L_LZ),
], L_LY)

l_com = _weighted_com(
    (np.array([L_LX_STEM / 2,            L_LY / 2, L_LZ / 2]),
     L_LX_STEM * L_LY * L_LZ),
    (np.array([(L_LX_STEM + L_LX) / 2,   L_LY / 2, L_LZ_BASE / 2]),
     (L_LX - L_LX_STEM) * L_LY * L_LZ_BASE),
)

l_tmpls = placer.place_mesh(l_verts, l_com, subject="L-shape")
print(f"L-shape  → {len(l_tmpls)} stable pose(s)  "
      f"(COM in natural frame: {l_com.round(3)})")
for t in l_tmpls:
    pose = t.instantiate(table_pose).sample()
    print(f"  [{t.variant:6s}]  COM z = {pose[2,3]:.3f} m  margin = {_margin(t)}")
print()


# ── T-shape ───────────────────────────────────────────────────────────────────
#
#   ┌─────────────────────────┐   lz_bar
#   └──────────┐  ┌───────────┘
#              │  │               lz_stem
#              └──┘
#   0        sx0  sx1           lx_bar

T_LX_BAR, T_LX_STEM = 0.12, 0.04
T_LY                 = 0.04
T_LZ_STEM, T_LZ_BAR  = 0.08, 0.04

T_SX0 = (T_LX_BAR - T_LX_STEM) / 2   # 0.04
T_SX1 = T_SX0 + T_LX_STEM             # 0.08

t_verts = _extrude_xz([
    (T_SX0,    0),                     (T_SX1,     0),
    (T_SX1,    T_LZ_STEM),             (T_LX_BAR,  T_LZ_STEM),
    (T_LX_BAR, T_LZ_STEM + T_LZ_BAR), (0,         T_LZ_STEM + T_LZ_BAR),
    (0,        T_LZ_STEM),             (T_SX0,     T_LZ_STEM),
], T_LY)

t_com = _weighted_com(
    (np.array([T_LX_BAR / 2, T_LY / 2, T_LZ_STEM / 2]),
     T_LX_STEM * T_LY * T_LZ_STEM),
    (np.array([T_LX_BAR / 2, T_LY / 2, T_LZ_STEM + T_LZ_BAR / 2]),
     T_LX_BAR  * T_LY * T_LZ_BAR),
)

t_tmpls = placer.place_mesh(t_verts, t_com, subject="T-shape")
print(f"T-shape  → {len(t_tmpls)} stable pose(s)  "
      f"(COM in natural frame: {t_com.round(3)})")
for t in t_tmpls:
    pose = t.instantiate(table_pose).sample()
    print(f"  [{t.variant:6s}]  COM z = {pose[2,3]:.3f} m  margin = {_margin(t)}")
print()


# ── Mug (cylinder body + box handle) ─────────────────────────────────────────
#
#   Top view:          Side view:
#     ┌───┐              ┌─────────┐
#     │   ├──┐           │   cyl   │
#     │   │  │  handle   └─────────┘
#     │   ├──┘
#     └───┘
#
MUG_R, MUG_H         = 0.04, 0.10
HDL_LX, HDL_LY, HDL_LZ = 0.06, 0.02, 0.06   # handle: extends in +x, spans ±y, ±z

ang = np.linspace(0, 2 * np.pi, 40, endpoint=False)
mug_verts = np.vstack([
    # top and bottom circles of the cylinder
    np.column_stack([MUG_R * np.cos(ang), MUG_R * np.sin(ang), np.full(40,  MUG_H / 2)]),
    np.column_stack([MUG_R * np.cos(ang), MUG_R * np.sin(ang), np.full(40, -MUG_H / 2)]),
    # 8 corners of the handle box
    np.array([[MUG_R + dx * HDL_LX, dy * HDL_LY / 2, dz * HDL_LZ / 2]
              for dx in (0, 1) for dy in (-1, 1) for dz in (-1, 1)]),
])

mug_com = _weighted_com(
    (np.zeros(3),                           np.pi * MUG_R**2 * MUG_H),
    (np.array([MUG_R + HDL_LX / 2, 0., 0.]), HDL_LX * HDL_LY * HDL_LZ),
)

mug_tmpls = placer.place_mesh(mug_verts, mug_com, subject="mug")
print(f"Mug      → {len(mug_tmpls)} stable pose(s)  "
      f"(COM in natural frame: {mug_com.round(4)})")
for t in mug_tmpls:
    pose = t.instantiate(table_pose).sample()
    print(f"  [{t.variant:6s}]  COM z = {pose[2,3]:.3f} m  margin = {_margin(t)}")
# The cylinder discretisation creates many tiny hull facets on the curved side
# (sub-3° margins).  Filter to physically meaningful poses for visualisation.
_MIN_MARGIN_DEG = 5.0
mug_tmpls_viz = [t for t in mug_tmpls if float(_margin(t).rstrip("°")) >= _MIN_MARGIN_DEG]
print(f"         → {len(mug_tmpls_viz)} pose(s) with margin ≥ {_MIN_MARGIN_DEG}° "
      f"shown in visualisation")


# ── Visualization ─────────────────────────────────────────────────────────────
try:
    import pyvista as pv
    import matplotlib.cm as cm
    from tsr.viz import TSRVisualizer, table_surface_renderer

    TX, TY = 0.60, 0.40

    def _pose(x, y, tmpl):
        T = np.eye(4); T[0, 3] = x; T[1, 3] = y
        return T @ tmpl.Tw_e

    def _row(tmpls, y, spacing=0.22):
        xs = [(i - (len(tmpls) - 1) / 2) * spacing for i in range(len(tmpls))]
        return [_pose(x, y, t) for x, t in zip(xs, tmpls)]

    l_poses   = _row(l_tmpls,   y= 0.18)
    t_poses   = _row(t_tmpls,   y= 0.00)
    mug_poses = _row(mug_tmpls_viz, y=-0.18)

    # ── Composite renderers ────────────────────────────────────────────────

    def _add_box(pl, bounds, R, t_w, color):
        """Render an axis-aligned box (in object frame) at world pose R, t_w."""
        b = pv.Box(bounds=bounds)
        b.points = (R @ b.points.T).T + t_w
        pl.add_mesh(b, color=color, smooth_shading=True,
                    lighting=True, specular=0.4, diffuse=0.8, ambient=0.15,
                    show_edges=True, edge_color="#e0e0e0", line_width=1.0)

    # L-shape: two boxes in object frame (origin = l_com)
    _L_STEM = (0 - l_com[0],          L_LX_STEM - l_com[0],
               0 - l_com[1],          L_LY      - l_com[1],
               0 - l_com[2],          L_LZ      - l_com[2])
    _L_BASE = (L_LX_STEM - l_com[0],  L_LX      - l_com[0],
               0 - l_com[1],          L_LY      - l_com[1],
               0 - l_com[2],          L_LZ_BASE - l_com[2])

    def l_renderer(pl, pose_4x4, color):
        R, t = pose_4x4[:3, :3], pose_4x4[:3, 3]
        _add_box(pl, _L_STEM, R, t, color)
        _add_box(pl, _L_BASE, R, t, color)

    # T-shape: stem box + bar box in object frame (origin = t_com)
    _T_STEM = (T_SX0     - t_com[0],  T_SX1              - t_com[0],
               0         - t_com[1],  T_LY               - t_com[1],
               0         - t_com[2],  T_LZ_STEM          - t_com[2])
    _T_BAR  = (0         - t_com[0],  T_LX_BAR           - t_com[0],
               0         - t_com[1],  T_LY               - t_com[1],
               T_LZ_STEM - t_com[2],  T_LZ_STEM+T_LZ_BAR - t_com[2])

    def t_renderer(pl, pose_4x4, color):
        R, t = pose_4x4[:3, :3], pose_4x4[:3, 3]
        _add_box(pl, _T_STEM, R, t, color)
        _add_box(pl, _T_BAR,  R, t, color)

    # Mug: cylinder + handle box in object frame (origin = mug_com)
    _MUG_CYL_CTR = -mug_com   # cylinder natural-frame centre is origin
    _MUG_HDL = (MUG_R        - mug_com[0],  MUG_R + HDL_LX - mug_com[0],
                -HDL_LY / 2,                 HDL_LY / 2,
                -HDL_LZ / 2,                 HDL_LZ / 2)

    def mug_renderer(pl, pose_4x4, color):
        R, t = pose_4x4[:3, :3], pose_4x4[:3, 3]
        ctr_w  = R @ _MUG_CYL_CTR + t
        axis_w = R @ np.array([0., 0., 1.])
        cyl = pv.Cylinder(center=ctr_w, direction=axis_w,
                          radius=MUG_R, height=MUG_H, resolution=40, capping=True)
        pl.add_mesh(cyl, color=color, smooth_shading=True,
                    lighting=True, specular=0.4, diffuse=0.8, ambient=0.15,
                    show_edges=True, edge_color="#e0e0e0", line_width=0.8)
        _add_box(pl, _MUG_HDL, R, t, color)

    C_L   = (0.122, 0.467, 0.706)  # blue
    C_T   = (0.173, 0.627, 0.173)  # green
    C_MUG = (0.839, 0.153, 0.157)  # red

    subjects = (
        [(l_renderer,     p, C_L)   for p in l_poses]
        + [(t_renderer,   p, C_T)   for p in t_poses]
        + [(mug_renderer, p, C_MUG) for p in mug_poses]
    )

    out = "assets/mesh_placements.png"
    TSRVisualizer(
        window_size=(1600, 900),
        camera_az=215., camera_el=35., camera_dist=1.55,
        focus=(0., 0., 0.06),
        title="Stable placements — L-shape · T-shape · mug",
        crop_pad=28,
        parallel_projection=True,
    ).render_multi(
        reference_renderer=table_surface_renderer(TX, TY),
        subjects=subjects,
        out=out,
    )
    print(f"\nSaved {out}")

except ImportError:
    print("\n(install viz extra for visualization: uv sync --extra viz)")
