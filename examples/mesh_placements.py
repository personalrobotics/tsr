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

from tsr.placement import StablePlacer

placer = StablePlacer(table_x=0.60, table_y=0.40)
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


# ── L-shape ──────────────────────────────────────────────────────────────────
#
#   lz_total │ ┌──────┐
#            │ │ stem │
#   lz_base  │ │      ├──────────────────┐
#            │ │      │   base arm       │
#            └─┴──────┴──────────────────┘
#              0    lx_stem           lx_total

L_LX, L_LX_STEM = 0.10, 0.04
L_LY = 0.04
L_LZ, L_LZ_BASE = 0.12, 0.04

l_verts = _extrude_xz(
    [
        (0, 0),
        (L_LX, 0),
        (L_LX, L_LZ_BASE),
        (L_LX_STEM, L_LZ_BASE),
        (L_LX_STEM, L_LZ),
        (0, L_LZ),
    ],
    L_LY,
)

l_com = _weighted_com(
    (np.array([L_LX_STEM / 2, L_LY / 2, L_LZ / 2]), L_LX_STEM * L_LY * L_LZ),
    (
        np.array([(L_LX_STEM + L_LX) / 2, L_LY / 2, L_LZ_BASE / 2]),
        (L_LX - L_LX_STEM) * L_LY * L_LZ_BASE,
    ),
)

l_tmpls = placer.place_mesh(l_verts, l_com, subject="L-shape")
print(f"L-shape  → {len(l_tmpls)} stable pose(s)  (COM in natural frame: {l_com.round(3)})")
for t in l_tmpls:
    pose = t.sample(table_pose)
    print(f"  [{t.variant:6s}]  origin z = {pose[2, 3]:.3f} m  margin = {np.degrees(t.stability_margin):.1f}°")
print()


# ── T-shape ───────────────────────────────────────────────────────────────────
#
#   ┌─────────────────────────┐   lz_bar
#   └──────────┐  ┌───────────┘
#              │  │               lz_stem
#              └──┘
#   0        sx0  sx1           lx_bar

T_LX_BAR, T_LX_STEM = 0.12, 0.04
T_LY = 0.04
T_LZ_STEM, T_LZ_BAR = 0.08, 0.04

T_SX0 = (T_LX_BAR - T_LX_STEM) / 2  # 0.04
T_SX1 = T_SX0 + T_LX_STEM  # 0.08

t_verts = _extrude_xz(
    [
        (T_SX0, 0),
        (T_SX1, 0),
        (T_SX1, T_LZ_STEM),
        (T_LX_BAR, T_LZ_STEM),
        (T_LX_BAR, T_LZ_STEM + T_LZ_BAR),
        (0, T_LZ_STEM + T_LZ_BAR),
        (0, T_LZ_STEM),
        (T_SX0, T_LZ_STEM),
    ],
    T_LY,
)

t_com = _weighted_com(
    (np.array([T_LX_BAR / 2, T_LY / 2, T_LZ_STEM / 2]), T_LX_STEM * T_LY * T_LZ_STEM),
    (
        np.array([T_LX_BAR / 2, T_LY / 2, T_LZ_STEM + T_LZ_BAR / 2]),
        T_LX_BAR * T_LY * T_LZ_BAR,
    ),
)

t_tmpls = placer.place_mesh(t_verts, t_com, subject="T-shape")
print(f"T-shape  → {len(t_tmpls)} stable pose(s)  (COM in natural frame: {t_com.round(3)})")
for t in t_tmpls:
    pose = t.sample(table_pose)
    print(f"  [{t.variant:6s}]  origin z = {pose[2, 3]:.3f} m  margin = {np.degrees(t.stability_margin):.1f}°")
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
MUG_R, MUG_H = 0.04, 0.10
HDL_LX, HDL_LY, HDL_LZ = 0.06, 0.02, 0.06  # handle: extends in +x, spans ±y, ±z

ang = np.linspace(0, 2 * np.pi, 40, endpoint=False)
mug_verts = np.vstack(
    [
        # top and bottom circles of the cylinder
        np.column_stack([MUG_R * np.cos(ang), MUG_R * np.sin(ang), np.full(40, MUG_H / 2)]),
        np.column_stack([MUG_R * np.cos(ang), MUG_R * np.sin(ang), np.full(40, -MUG_H / 2)]),
        # 8 corners of the handle box
        np.array(
            [
                [MUG_R + dx * HDL_LX, dy * HDL_LY / 2, dz * HDL_LZ / 2]
                for dx in (0, 1)
                for dy in (-1, 1)
                for dz in (-1, 1)
            ]
        ),
    ]
)

mug_com = _weighted_com(
    (np.zeros(3), np.pi * MUG_R**2 * MUG_H),
    (np.array([MUG_R + HDL_LX / 2, 0.0, 0.0]), HDL_LX * HDL_LY * HDL_LZ),
)

# The mug's body is tessellated into 40 flat sides, each a real resting face of the
# polyhedron supplied, at 180/40 = 4.5 deg. Filtering above that leaves the poses a
# caller who meant a cylinder would want (see docs/ARCHITECTURE.md, issue #152).
_MIN_MARGIN_DEG = 5.0
mug_tmpls = placer.place_mesh(mug_verts, mug_com, subject="mug")
print(f"Mug      → {len(mug_tmpls)} stable pose(s)  (COM in natural frame: {mug_com.round(4)})")
for t in mug_tmpls:
    pose = t.sample(table_pose)
    print(f"  [{t.variant:6s}]  origin z = {pose[2, 3]:.3f} m  margin = {np.degrees(t.stability_margin):.1f}°")
mug_tmpls_viz = placer.place_mesh(mug_verts, mug_com, subject="mug", min_margin_deg=_MIN_MARGIN_DEG)
print(f"         → {len(mug_tmpls_viz)} pose(s) with margin ≥ {_MIN_MARGIN_DEG}° shown in visualisation")


# ── Visualization ────────────────────────────────────────────────────────────
# Rendering lives in the Viser viewer, not in this example:
#
#   from tsr.viser import studio; studio().sleep_forever()   # interactive bench
#   uv run python scripts/render_readme_figures.py           # the README figures
#
# Install it with: pip install "sstsr[viz]"   (see docs/VISER.md)
