"""Parallel jaw gripper TSR example — cylinder, sphere, and box grasps.

Shows all grasp modes for all three primitives in a single scene:
  Cylinder (left):  side · top · bottom
  Sphere  (centre): equatorial, k depths × 2 rolls
  Box (right):      ±x · ±y · top · bottom, two finger orientations per face
                    (filtered by max_aperture)

Gripper frame convention:
    z = approach direction (toward object surface)
    y = finger opening direction
    x = palm normal  (right-hand rule: x = y × z)

Saves: assets/tsr_viz.png

Usage:
    uv run python examples/parallel_jaw_grasp.py
"""
import numpy as np

from tsr.hands import ParallelJawGripper
from tsr.viz import (
    TSRVisualizer, box_renderer, cylinder_renderer, sphere_renderer, plasma_colors,
)

# ── Scene parameters ───────────────────────────────────────────────────────────
MUG_R = 0.040   # mug radius [m]
MUG_H = 0.120   # mug height [m]

SPH_R = 0.040   # sphere radius [m]

BOX_X = 0.080   # box width  [m]
BOX_Y = 0.060   # box depth  [m]
BOX_Z = 0.180   # box height [m]

N = 3           # samples per template

# Objects separated along world-x, which is image-horizontal at az=270°
CYL_OFF = (-0.22, 0.,     0.      )  # cylinder on the left
SPH_OFF = ( 0.,   0.,     SPH_R   )  # sphere in the centre (center at z=r)
BOX_OFF = ( 0.22, 0.,     0.      )  # box on the right


# ── Helpers ────────────────────────────────────────────────────────────────────
def _collect(templates, object_pose, n):
    """Sample n poses per template; return (poses, colors) lists."""
    cols = plasma_colors(len(templates), lo=0.05, hi=0.95)
    poses, colors = [], []
    for i, t in enumerate(templates):
        tsr = t.instantiate(object_pose)
        poses.extend(tsr.sample() for _ in range(n))
        colors.extend([cols[i]] * n)
    return poses, colors


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    from pathlib import Path

    out = Path(__file__).parent.parent / "assets" / "tsr_viz.png"

    gripper = ParallelJawGripper(finger_length=0.055, max_aperture=0.140)

    # Cylinder grasps
    cyl_pose = np.eye(4); cyl_pose[:3, 3] = CYL_OFF
    cyl_templates = gripper.grasp_cylinder(
        cylinder_radius=MUG_R, cylinder_height=MUG_H, reference="mug")
    cyl_poses, cyl_colors = _collect(cyl_templates, cyl_pose, N)

    # Sphere grasps
    sph_pose = np.eye(4); sph_pose[:3, 3] = SPH_OFF
    sph_templates = gripper.grasp_sphere(object_radius=SPH_R, reference="ball")
    sph_poses, sph_colors = _collect(sph_templates, sph_pose, N)

    # Box grasps
    box_pose = np.eye(4); box_pose[:3, 3] = BOX_OFF
    box_templates = gripper.grasp_box(
        box_x=BOX_X, box_y=BOX_Y, box_z=BOX_Z, reference="box")
    box_poses, box_colors = _collect(box_templates, box_pose, N)

    all_poses  = cyl_poses  + sph_poses  + box_poses
    all_colors = cyl_colors + sph_colors + box_colors

    n_cyl = len(cyl_templates)
    n_sph = len(sph_templates)
    n_box = len(box_templates)
    print(f"Cylinder: {n_cyl} templates · {len(cyl_poses)} poses")
    print(f"Sphere:   {n_sph} templates · {len(sph_poses)} poses")
    print(f"Box:      {n_box} templates · {len(box_poses)} poses")

    # Camera at az=270° looks along +y; world-x = image-horizontal → side by side
    focus = (0., 0., (MUG_H + BOX_Z) / 4.)

    def scene_renderer(pl):
        cylinder_renderer(MUG_R, MUG_H, offset=CYL_OFF)(pl)
        sphere_renderer(SPH_R, offset=SPH_OFF)(pl)
        box_renderer(BOX_X, BOX_Y, BOX_Z, offset=BOX_OFF)(pl)

    TSRVisualizer(
        title=(f"Task Space Regions — cylinder ({n_cyl}) · sphere ({n_sph}) · "
               f"box ({n_box}) templates\n"
               f"{len(all_poses)} sampled poses"),
        focus=focus,
        camera_dist=1.15,
        camera_az=270.,
        camera_el=22.,
    ).render(
        reference_renderer=scene_renderer,
        subject_renderer=gripper.renderer(),
        poses=all_poses,
        colors=all_colors,
        out=str(out),
    )
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
