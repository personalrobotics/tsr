"""stable_placements.py — Stable placement TSR generation with StablePlacer.

Demonstrates how to generate placement TSRs for geometric primitives and
arbitrary meshes, then sample valid placement poses on a table surface.


Usage::

    uv run python examples/stable_placements.py
    uv run python examples/stable_placements.py
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
templates = placer.place_cylinder(cylinder_radius=0.04, cylinder_height=0.12, subject="mug")
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
templates = placer.place_torus(major_radius=0.05, minor_radius=0.012, subject="ring")
t = templates[0]
pose = t.sample(table_pose)
print(f"Torus ({len(templates)} template)")
print(f"  variant  : {t.variant}")
print(f"  COM z    : {pose[2, 3]:.3f} m  (expected ≈ {0.75 + 0.012:.3f} m)\n")


# ── Mesh (cube approximated as 8 vertices) ───────────────────────────────────
L = 0.05
cube_verts = np.array(
    [
        [-L, -L, -L],
        [L, -L, -L],
        [L, L, -L],
        [-L, L, -L],
        [-L, -L, L],
        [L, -L, L],
        [L, L, L],
        [-L, L, L],
    ]
)
cube_com = np.zeros(3)
templates = placer.place_mesh(cube_verts, cube_com, subject="cube")
print(f"Mesh/cube ({len(templates)} stable faces, sorted by stability margin)")
for t in templates:
    pose = t.sample(table_pose)
    print(f"  [{t.variant:6s}]  COM z = {pose[2, 3]:.3f} m  margin = {np.degrees(t.stability_margin):.1f}°")


# ── Visualization ────────────────────────────────────────────────────────────
# Rendering lives in the Viser viewer, not in this example:
#
#   from tsr.viser import studio; studio().sleep_forever()   # interactive bench
#   uv run python scripts/render_readme_figures.py           # the README figures
#
# Install it with: pip install "sstsr[viz]"   (see docs/VISER.md)
