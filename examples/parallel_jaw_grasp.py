"""Parallel jaw gripper TSR example.

Demonstrates all three cylinder grasp modes: side, top-down, and bottom-up.

Gripper frame convention (canonical for this library):
    z = approach direction (toward object surface)
    y = finger opening direction
    x = palm normal  (right-hand rule: x = y × z)

AnyGrasp / GraspNet uses x=approach — convert with:
    R_convert = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])

Saves: assets/tsr_viz.png

Usage:
    uv run python examples/parallel_jaw_grasp.py
"""
import numpy as np

from tsr.hands import ParallelJawGripper
from tsr.viz import TSRVisualizer, cylinder_renderer, plasma_colors

# ── Scene parameters ──────────────────────────────────────────────────────────
MUG_R = 0.040   # mug radius [m]
MUG_H = 0.120   # mug height [m]
N     = 6       # samples per template


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    from pathlib import Path

    out = Path(__file__).parent.parent / "assets" / "tsr_viz.png"

    # ── 1. Define your gripper ────────────────────────────────────────────────
    gripper = ParallelJawGripper(finger_length=0.055, max_aperture=0.140)

    # ── 2. Generate templates for all three approach modes ────────────────────
    side_templates   = gripper.grasp_cylinder_side(
        cylinder_radius=MUG_R, cylinder_height=MUG_H, reference="mug")
    top_templates    = gripper.grasp_cylinder_top(
        cylinder_radius=MUG_R, cylinder_height=MUG_H, reference="mug")
    bottom_templates = gripper.grasp_cylinder_bottom(
        cylinder_radius=MUG_R, cylinder_height=MUG_H, reference="mug")

    all_groups = [
        (side_templates,   plasma_colors(len(side_templates),   lo=0.05, hi=0.38)),
        (top_templates,    plasma_colors(len(top_templates),    lo=0.45, hi=0.65)),
        (bottom_templates, plasma_colors(len(bottom_templates), lo=0.72, hi=0.95)),
    ]

    # ── 3. Instantiate at object pose and sample ──────────────────────────────
    mug_pose = np.eye(4)
    poses, colors = [], []
    for templates, tsr_colors in all_groups:
        for i, template in enumerate(templates):
            tsr   = template.instantiate(mug_pose)
            batch = [tsr.sample() for _ in range(N)]
            poses.extend(batch)
            colors.extend([tsr_colors[i]] * N)

    n_side   = len(side_templates)
    n_top    = len(top_templates)
    n_bottom = len(bottom_templates)
    print(f"Templates: {n_side} side · {n_top} top · {n_bottom} bottom")
    print(f"Total poses: {len(poses)}")

    # ── 4. Visualize ──────────────────────────────────────────────────────────
    TSRVisualizer(
        title=(f"Task Space Regions — Side (purple) · Top (green) · Bottom (orange)\n"
               f"{n_side+n_top+n_bottom} templates · {len(poses)} sampled poses"),
        focus=(0., 0., MUG_H / 2.),
        camera_dist=0.65,
        camera_el=30.,
    ).render(
        reference_renderer=cylinder_renderer(radius=MUG_R, height=MUG_H),
        subject_renderer=gripper.renderer(),
        poses=poses,
        colors=colors,
        out=str(out),
    )
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
