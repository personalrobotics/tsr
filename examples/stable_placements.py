"""stable_placements.py — Stable placement TSR generation with TablePlacer.

Demonstrates how to generate placement TSRs for geometric primitives and
arbitrary meshes, then sample valid placement poses on a table surface.

Also renders a visualization of a box in each of its 3 stable poses.

Usage::

    uv run python examples/stable_placements.py
    uv sync --extra viz && uv run python examples/stable_placements.py
"""
import numpy as np

from tsr.placement import TablePlacer

# 60 × 40 cm table surface; z points up; table origin at surface center.
placer = TablePlacer(table_x=0.30, table_y=0.20, reference="table")

# World pose of the table surface.
table_pose = np.eye(4)
table_pose[2, 3] = 0.75  # table at z = 0.75 m

print("=== TablePlacer Demo ===\n")


# ── Cylinder (mug) ──────────────────────────────────────────────────────────
templates = placer.place_cylinder(cylinder_radius=0.04, cylinder_height=0.12,
                                  subject="mug")
t = templates[0]
tsr = t.instantiate(table_pose)
pose = tsr.sample()
print(f"Cylinder ({len(templates)} template)")
print(f"  variant  : {t.variant}")
print(f"  COM z    : {pose[2, 3]:.3f} m  (expected ≈ {0.75 + 0.06:.3f} m)\n")


# ── Box (cereal box, three distinct dimensions) ──────────────────────────────
templates = placer.place_box(lx=0.20, ly=0.08, lz=0.28, subject="box")
print(f"Box ({len(templates)} templates, one per stable face)")
for t in templates:
    pose = t.instantiate(table_pose).sample()
    print(f"  [{t.variant:6s}]  COM z = {pose[2, 3]:.3f} m")
print()


# ── Sphere ──────────────────────────────────────────────────────────────────
templates = placer.place_sphere(radius=0.05, subject="ball")
t = templates[0]
print(f"Sphere ({len(templates)} template)")
print(f"  variant        : {t.variant}")
print(f"  roll/pitch Bw  : {t.Bw[3].tolist()}  (all orientations free)\n")


# ── Torus (ring) ─────────────────────────────────────────────────────────────
templates = placer.place_torus(major_radius=0.05, minor_radius=0.012,
                               subject="ring")
t = templates[0]
pose = t.instantiate(table_pose).sample()
print(f"Torus ({len(templates)} template)")
print(f"  variant  : {t.variant}")
print(f"  COM z    : {pose[2, 3]:.3f} m  (expected ≈ {0.75 + 0.012:.3f} m)\n")


# ── Mesh (cube approximated as 8 vertices) ───────────────────────────────────
L = 0.05
cube_verts = np.array([
    [-L, -L, -L], [L, -L, -L], [L, L, -L], [-L, L, -L],
    [-L, -L,  L], [L, -L,  L], [L, L,  L], [-L, L,  L],
])
cube_com = np.zeros(3)
templates = placer.place_mesh(cube_verts, cube_com, subject="cube")
print(f"Mesh/cube ({len(templates)} stable faces, sorted by stability margin)")
for t in templates:
    pose = t.instantiate(table_pose).sample()
    margin_str = t.name.split("margin ")[1].rstrip(")")
    print(f"  [{t.variant:6s}]  COM z = {pose[2, 3]:.3f} m  margin = {margin_str}")


# ── Visualization ────────────────────────────────────────────────────────────
# Show a box (10×6×18 cm) in each of its 3 stable poses on a table.
# Requires: uv sync --extra viz
try:
    from tsr.viz import (TSRVisualizer, table_surface_renderer,
                         placed_box_renderer, plasma_colors)

    TX, TY = 0.30, 0.20    # table half-extents
    LX, LY, LZ = 0.10, 0.06, 0.18   # box dimensions

    box_templates = TablePlacer(table_x=TX, table_y=TY).place_box(LX, LY, LZ)

    # Canonical placement pose at each x offset (Bw = 0, yaw = 0).
    x_offsets = np.linspace(-TX * 0.55, TX * 0.55, len(box_templates))
    poses = []
    for tmpl, x in zip(box_templates, x_offsets):
        T = np.eye(4)
        T[0, 3] = x
        poses.append(T @ tmpl.Tw_e)

    out = "assets/stable_placements.png"
    TSRVisualizer(
        window_size=(1400, 820),
        camera_az=215., camera_el=28., camera_dist=0.62,
        focus=(0., 0., 0.06),
        title="Box — 3 stable placement poses (z-face · y-face · x-face)",
        crop_pad=28,
    ).render(
        reference_renderer=table_surface_renderer(TX, TY),
        subject_renderer=placed_box_renderer(LX, LY, LZ),
        poses=poses,
        out=out,
        colors=plasma_colors(len(poses)),
    )
    print(f"\nSaved {out}")

except ImportError:
    print("\n(install viz extra for visualization: uv sync --extra viz)")
