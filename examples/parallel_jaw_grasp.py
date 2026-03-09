"""Parallel jaw gripper TSR example.

Shows how to write a hand class that generates TSRTemplates from object
geometry, and how to visualize sampled poses with TSRVisualizer.

This is the canonical pattern for adding a new hand to the TSR library:
  1. Implement grasp_* methods that return TSRTemplates
  2. Pass a renderer factory to TSRVisualizer for visualization

Gripper frame convention (canonical for this library):
    z = approach direction (toward object surface)
    y = finger opening direction
    x = palm normal  (right-hand rule: x = y × z)

AnyGrasp / GraspNet uses x=approach → convert with R_y(90°):
    R_convert = [[0, 0, -1], [0, 1, 0], [1, 0, 0]]

Saves: assets/tsr_viz.png

Usage:
    uv run python examples/parallel_jaw_grasp.py
"""
import numpy as np
from typing import List, Optional, Tuple

from tsr.template import TSRTemplate
from tsr.viz import TSRVisualizer, cylinder_renderer, parallel_jaw_renderer, plasma_colors

# ── Scene parameters ──────────────────────────────────────────────────────────
MUG_R = 0.040   # mug radius [m]
MUG_H = 0.120   # mug height [m]
N     = 15      # gripper samples to visualize


# ── Gripper ───────────────────────────────────────────────────────────────────
class ParallelJawGripper:
    """Parallel jaw gripper: generates TSRTemplates from object geometry.

    The poses sampled from these TSRs are pre-grasp configurations — the hand
    is open at the specified preshape, positioned so that closing the fingers
    achieves force closure on the object. The TSR itself does not enforce
    force closure; it constrains where the hand must be before closing.

    Frame convention:
        z = approach direction (toward object surface)
        y = finger opening direction
        x = palm normal (right-hand rule: x = y × z)

    AnyGrasp / GraspNet uses x=approach → convert with R_y(90°):
        R = [[0, 0, -1], [0, 1, 0], [1, 0, 0]]

    To add a new hand:
        1. Copy this class
        2. Set finger_length and max_aperture for your hand
        3. Register a renderer in tsr.viz (or write a custom one)

    Args:
        finger_length: Distance from palm to fingertip [m].
        max_aperture:  Maximum jaw opening [m].
    """

    def __init__(self, finger_length: float, max_aperture: float):
        self.finger_length = finger_length
        self.max_aperture  = max_aperture

    def grasp_cylinder(
        self,
        object_radius: float,
        height_range: Tuple[float, float],
        preshape: Optional[float] = None,
        k: int = 3,
        clearance: Optional[float] = None,
        angle_range: Tuple[float, float] = (0., 2 * np.pi),
        subject: str = "gripper",
        reference: str = "cylinder",
        name: str = "",
        description: str = "",
    ) -> List[TSRTemplate]:
        """Side grasp templates for a cylinder — 2*k templates total.

        Returns k depth levels × 2 roll orientations. Each template has a fixed
        radial offset baked into Tw_e, covering the full pre-grasp volume:

          depth 1/k : fingertips clearance inside surface (shallowest)
          depth k/k : palm clearance from surface (deepest)

        Two roll orientations per depth (180° apart around z_EE):
          roll=0 : palm normal = +cylinder_axis, fingers open in +tangential
          roll=π : palm normal = -cylinder_axis, fingers open in -tangential
        A symmetric hand produces identical poses; a non-symmetric hand needs both.

        The radial approach direction couples with yaw and cannot be encoded in
        Bw directly. Instead, k discrete depths are baked into Tw_e so the full
        pre-grasp volume is covered without post-processing.

        Args:
            preshape:   Pre-grasp jaw opening [m]. Default: 2*r + clearance
                        (minimum viable opening). Must exceed cylinder diameter.
            k:          Number of discrete approach depths (default 3).
            clearance:  Safety buffer [m] applied to: height ends, fingertip
                        start depth, and palm stop depth. Default: 10% of finger_length.

        Returns [] if preshape <= cylinder diameter (fingers can't span it).
        Raises ValueError for invalid geometry parameters.
        """
        if clearance is None:
            clearance = 0.1 * self.finger_length
        if preshape is None:
            preshape = 2. * object_radius + clearance
        if object_radius <= 0:
            raise ValueError("object_radius must be > 0")
        if preshape > self.max_aperture:
            raise ValueError(
                f"preshape {preshape:.3f}m > max_aperture {self.max_aperture:.3f}m"
            )
        if preshape <= 2. * object_radius:
            return []   # preshape too small to span cylinder
        h0, h1 = height_range[0] + clearance, height_range[1] - clearance
        if h1 <= h0:
            raise ValueError("height_range too narrow for the given clearance")

        if not name:
            name = f"{reference.title()} Cylinder Side Grasp"
        if not description:
            description = (
                f"Side grasp on {reference} (r={object_radius*100:.0f}cm, "
                f"h=[{h0*100:.0f},{h1*100:.0f}]cm)"
            )

        z_mid, z_half = (h0 + h1) / 2., (h1 - h0) / 2.
        T_ref_tsr = np.eye(4)
        T_ref_tsr[2, 3] = z_mid

        # Bw is identical for all templates (height + yaw freedom only).
        # Radial depth is NOT in Bw — it's baked into Tw_e per depth level.
        Bw = np.array([
            [0.,             0.            ],  # x: no radial freedom
            [0.,             0.            ],  # y: no tangential freedom
            [-z_half,        z_half        ],  # z: height range (symmetric)
            [0.,             0.            ],  # roll: fixed (encoded in Tw_e)
            [0.,             0.            ],  # pitch
            [angle_range[0], angle_range[1]],  # yaw: angular freedom
        ])

        # k depths: fingertips clearance-inside-surface → palm clearance-from-surface.
        # Both ends maintain a clearance buffer from the cylinder surface.
        approach_max = min(self.finger_length, object_radius) - clearance
        depths = np.linspace(clearance, approach_max, max(k, 1))

        # Human-readable depth labels: shallow/mid/deep for k≤3, else "depth i/k".
        _depth_labels = {1: ["mid"], 2: ["shallow", "deep"],
                         3: ["shallow", "mid", "deep"]}

        # Tw_e encodes both roll orientation and radial depth.
        # z_EE = [-1,0,0] in TSR frame (radially inward) for all variants.
        # roll=0:  x_EE=[0,0,1],  y_EE=[0, 1,0]
        # roll=π:  x_EE=[0,0,-1], y_EE=[0,-1,0]  (180° around z_EE)
        common = dict(
            T_ref_tsr=T_ref_tsr, Bw=Bw,
            task="grasp", subject=subject, reference=reference,
            preshape=np.array([preshape]),
        )
        templates = []
        for i, d in enumerate(depths):
            ro = object_radius + self.finger_length - d  # radial offset for this depth
            depth_label = (_depth_labels.get(k) or [f"depth {j+1}/{k}" for j in range(k)])[i]
            Tw_e_0 = np.array([
                [ 0.,  0., -1., ro],
                [ 0.,  1.,  0., 0.],
                [ 1.,  0.,  0., 0.],
                [ 0.,  0.,  0., 1.],
            ])
            Tw_e_pi = np.array([
                [ 0.,  0., -1., ro],
                [ 0., -1.,  0., 0.],
                [-1.,  0.,  0., 0.],
                [ 0.,  0.,  0., 1.],
            ])
            for Tw_e, roll_label in ((Tw_e_0, "roll 0°"), (Tw_e_pi, "roll 180°")):
                t_desc = (description or (
                    f"{depth_label.capitalize()} side grasp on {reference}: "
                    f"standoff {ro*1000:.0f}mm from axis, {roll_label}, "
                    f"preshape {preshape*1000:.0f}mm"
                ))
                templates.append(TSRTemplate(
                    Tw_e=Tw_e,
                    name=f"{name} — {depth_label}, {roll_label}",
                    description=t_desc,
                    **common,
                ))
        return templates



# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    from pathlib import Path

    out = Path(__file__).parent.parent / "assets" / "tsr_viz.png"

    # ── 1. Define your gripper ────────────────────────────────────────────────
    # Robotiq 2F-140: 140mm max aperture, 55mm finger length.
    # For a different hand: change these two numbers and register a renderer.
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

    # ── 3. Instantiate at object pose and sample from all templates ──────────
    mug_pose = np.eye(4)

    # One color per TSR template.
    tsr_colors = plasma_colors(len(templates), lo=0.05, hi=0.95)
    n_per = max(N // len(templates), 1)

    poses, colors = [], []
    for i, template in enumerate(templates):
        tsr = template.instantiate(mug_pose)
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
        subject_renderer=parallel_jaw_renderer(
            finger_length=gripper.finger_length,
            half_aperture=gripper.max_aperture / 2,
        ),
        poses=poses,
        colors=colors,
        out=str(out),
    )
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
