"""Parallel jaw gripper box grasp TSR example.

Demonstrates all four box grasp modes: ±x faces, ±y faces, top-down, bottom-up.

Gripper frame convention (canonical for this library):
    z = approach direction (toward object surface)
    y = finger opening direction
    x = palm normal  (right-hand rule: x = y × z)

Box coordinate frame (reference object convention for all grasp_box_* methods):

    x ∈ [-BOX_X/2, +BOX_X/2]   (centered)
    y ∈ [-BOX_Y/2, +BOX_Y/2]   (centered)
    z ∈ [0,         BOX_Z   ]   (bottom at z=0, top at z=BOX_Z)

    box_pose = np.eye(4) places the box bottom-center at the world origin.

Saves: assets/box_tsr_viz.png

Usage:
    uv run python examples/box_grasp.py
"""
import numpy as np

from tsr.hands import ParallelJawGripper
from tsr.viz import TSRVisualizer, box_renderer, plasma_colors

# ── Scene parameters ───────────────────────────────────────────────────────────
BOX_X = 0.080   # box width  [m]
BOX_Y = 0.060   # box depth  [m]
BOX_Z = 0.180   # box height [m]
N     = 4       # samples per template


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    from pathlib import Path

    out = Path(__file__).parent.parent / "assets" / "box_tsr_viz.png"

    # ── 1. Define your gripper ─────────────────────────────────────────────────
    gripper = ParallelJawGripper(finger_length=0.055, max_aperture=0.140)

    # ── 2. Generate templates for all four approach modes ──────────────────────
    templates = gripper.grasp_box(
        box_x=BOX_X, box_y=BOX_Y, box_z=BOX_Z, reference="box")
    print(f"Templates: {len(templates)}")

    # ── 3. Instantiate at object pose and sample ───────────────────────────────
    box_pose   = np.eye(4)
    tsr_colors = plasma_colors(len(templates), lo=0.05, hi=0.95)
    poses, colors = [], []
    for i, template in enumerate(templates):
        tsr   = template.instantiate(box_pose)
        batch = [tsr.sample() for _ in range(N)]
        poses.extend(batch)
        colors.extend([tsr_colors[i]] * N)

    print(f"Total poses: {len(poses)}")

    # ── 4. Visualize ───────────────────────────────────────────────────────────
    TSRVisualizer(
        title=(f"Task Space Regions — Box Grasps (±x · ±y · top · bottom)\n"
               f"{len(templates)} templates · {len(poses)} sampled poses"),
        focus=(0., 0., BOX_Z / 2.),
        camera_dist=0.65,
        camera_el=25.,
    ).render(
        reference_renderer=box_renderer(box_x=BOX_X, box_y=BOX_Y, box_z=BOX_Z),
        subject_renderer=gripper.renderer(),
        poses=poses,
        colors=colors,
        out=str(out),
    )
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
