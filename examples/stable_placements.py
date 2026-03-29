"""stable_placements.py — Stable placement TSR generation with StablePlacer.

Demonstrates how to generate placement TSRs for geometric primitives and
arbitrary meshes, then sample valid placement poses on a table surface.

Also renders a visualization of a box in each of its 3 stable poses.

Usage::

    uv run python examples/stable_placements.py
    uv sync --extra viz && uv run python examples/stable_placements.py
"""
import numpy as np

from tsr.placement import StablePlacer

# 60 × 40 cm table surface; z points up; table origin at surface center.
placer = StablePlacer(table_x=0.30, table_y=0.20, reference="table")

# World pose of the table surface.
table_pose = np.eye(4)
table_pose[2, 3] = 0.75  # table at z = 0.75 m

print("=== StablePlacer Demo ===\n")


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
    pose = t.sample(table_pose)
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
pose = t.sample(table_pose)
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
    pose = t.sample(table_pose)
    print(f"  [{t.variant:6s}]  COM z = {pose[2, 3]:.3f} m  margin = {np.degrees(t.stability_margin):.1f}°")


# ── Visualization ────────────────────────────────────────────────────────────
# Show all 4 primitives in all their stable poses on a single table.
# Each object type is one colour; multiple stable poses share that colour.
# Requires: uv sync --extra viz
try:
    import matplotlib.cm as cm
    from tsr.viz import (TSRVisualizer, table_surface_renderer,
                         placed_box_renderer, placed_cylinder_renderer,
                         placed_sphere_renderer, placed_torus_renderer)

    TX, TY = 0.55, 0.38              # table half-extents (110 × 76 cm)

    placer = StablePlacer(table_x=TX, table_y=TY)

    CYL_R, CYL_H            = 0.04, 0.12
    BOX_LX, BOX_LY, BOX_LZ  = 0.10, 0.06, 0.18
    SPH_R                   = 0.05
    TOR_R, TOR_r             = 0.05, 0.012

    cyl_tmpls = placer.place_cylinder(CYL_R, CYL_H)   # 2 faces
    box_tmpls = placer.place_box(BOX_LX, BOX_LY, BOX_LZ)   # 6 faces
    sph_tmpls = placer.place_sphere(SPH_R)
    tor_tmpls = placer.place_torus(TOR_R, TOR_r)      # 2 faces

    def _pose(x, y, tmpl):
        T = np.eye(4); T[0, 3] = x; T[1, 3] = y
        return T @ tmpl.Tw_e

    # Row 1 (y=+0.22): cyl -z · cyl +z · sphere · tor -z · tor +z
    # Row 2 (y= 0.00): box -z · box +z · box -y · box +y
    # Row 3 (y=-0.22): box -x · box +x
    r1, r2, r3 = +0.22, 0.00, -0.22
    cyl_poses = [_pose(x, r1, t) for x, t in zip([-0.44, -0.28], cyl_tmpls)]
    sph_pose  = _pose(-0.10, r1, sph_tmpls[0])
    tor_poses = [_pose(x, r1, t) for x, t in zip([+0.10, +0.28], tor_tmpls)]
    box_poses = (
        [_pose(x, r2, box_tmpls[i]) for i, x in enumerate([-0.38, -0.20], 0)]
        + [_pose(x, r2, box_tmpls[i]) for i, x in enumerate([-0.02, +0.16], 2)]
        + [_pose(x, r3, box_tmpls[i]) for i, x in enumerate([-0.22, +0.10], 4)]
    )

    # Box/cylinder/torus renderers use fixed per-face palettes; only sphere
    # uses the passed colour.
    c_sph = cm.tab20(8)[:3]
    _nc   = (0.5, 0.5, 0.5)  # placeholder, ignored by multi-face renderers

    subjects = (
        [(placed_cylinder_renderer(CYL_R, CYL_H),        p, _nc) for p in cyl_poses]
        + [(placed_box_renderer(BOX_LX, BOX_LY, BOX_LZ), p, _nc) for p in box_poses]
        + [(placed_sphere_renderer(SPH_R),                sph_pose, c_sph)]
        + [(placed_torus_renderer(TOR_R, TOR_r),          p, _nc) for p in tor_poses]
    )

    out = "assets/stable_placements.png"
    TSRVisualizer(
        window_size=(1600, 1000),
        camera_az=215., camera_el=35., camera_dist=1.20,
        focus=(0., 0., 0.07),
        title="Stable placements — cylinder · box (6 faces) · sphere · torus",
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
