"""Parallel jaw gripper TSR example.

Demonstrates grasp template generation and visualization using tsr.hands.

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
N     = 15      # gripper samples to visualize


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    from pathlib import Path

    out = Path(__file__).parent.parent / "assets" / "tsr_viz.png"

    # ── 1. Define your gripper ────────────────────────────────────────────────
    # Robotiq 2F-140: 140mm max aperture, 55mm finger length.
    gripper = ParallelJawGripper(finger_length=0.055, max_aperture=0.140)

    # ── 2. Generate TSR templates from object geometry ────────────────────────
    # preshape and clearance are auto-computed from geometry (10% of finger_length).
    templates = gripper.grasp_cylinder(
        object_radius=MUG_R,
        height_range=(0.025, MUG_H - 0.025),
        reference="mug",
    )
    for t in templates:
        print(f"Template: {t.name}")
        print(f"  Tw_e:\n{np.array2string(t.Tw_e, precision=4, suppress_small=True)}")
    print(f"  Bw:\n{np.array2string(templates[0].Bw, precision=4, suppress_small=True)}")

    # ── 3. Instantiate at object pose and sample from all templates ───────────
    mug_pose   = np.eye(4)
    tsr_colors = plasma_colors(len(templates), lo=0.05, hi=0.95)
    n_per      = max(N // len(templates), 1)

    poses, colors = [], []
    for i, template in enumerate(templates):
        tsr   = template.instantiate(mug_pose)
        batch = [tsr.sample() for _ in range(n_per)]
        poses.extend(batch)
        colors.extend([tsr_colors[i]] * n_per)

    # ── 4. Visualize ──────────────────────────────────────────────────────────
    TSRVisualizer(
        title=f"Task Space Region — Cylinder Side Grasp\n"
              f"{len(templates)} templates · {len(poses)} sampled gripper poses",
        focus=(0., 0., MUG_H / 2.),
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
