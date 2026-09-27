"""Parallel jaw grasp templates for all four primitives.

Shows every grasp mode the factories produce:
  Cylinder: side, top, bottom
  Sphere:   full SO(3), k depths
  Torus:    radial side + span (when the aperture allows)
  Box:      six faces, two finger orientations per face (aperture-filtered)

Gripper frame convention:
    z = approach direction (toward the object surface)
    y = finger opening direction
    x = palm normal  (right-hand rule: x = y x z)

To SEE these, use the viewer rather than this script:

    from tsr.viser import studio; studio().sleep_forever()   # interactive bench
    uv run python examples/viser_cylinder_grasps.py          # one worked scene

Usage:
    uv run python examples/parallel_jaw_grasp.py
"""

from tsr.hands import ParallelJawGripper

MUG_R, MUG_H = 0.040, 0.120
SPH_R = 0.040
TOR_R, TOR_r = 0.035, 0.015
BOX_X, BOX_Y, BOX_Z = 0.080, 0.060, 0.180


def main() -> None:
    gripper = ParallelJawGripper(finger_length=0.055, max_aperture=0.14)
    print(f"gripper: finger_length={gripper.finger_length} m, max_aperture={gripper.max_aperture} m\n")

    families = (
        ("cylinder", gripper.grasp_cylinder(MUG_R, MUG_H, reference="mug")),
        ("sphere", gripper.grasp_sphere(SPH_R, reference="ball")),
        ("torus", gripper.grasp_torus(TOR_R, TOR_r, reference="handle")),
        ("box", gripper.grasp_box(BOX_X, BOX_Y, BOX_Z, reference="box")),
    )
    for label, templates in families:
        modes = {}
        for t in templates:
            p = t.provenance
            modes.setdefault(f"{p.mode}/{p.approach}/{p.finger_orientation}", []).append(t)
        print(f"{label}: {len(templates)} templates")
        for mode, members in sorted(modes.items()):
            preshape = float(members[0].preshape[0])
            depths = ", ".join(f"{t.provenance.depth * 1000:.0f}" for t in members[: members[0].provenance.depth_count])
            print(f"   {mode:26s} x{len(members):<3d} preshape {preshape * 1000:.0f} mm, depths {depths} mm")
        print()

    # A torus is the clearest case of the aperture deciding WHICH modes exist: the side
    # grasp closes on the tube (2r), the span grasp on the whole ring (2(R+r)).
    narrow = ParallelJawGripper(finger_length=0.055, max_aperture=0.06)
    spans = {t.provenance.mode for t in narrow.grasp_torus(TOR_R, TOR_r)}
    print(f"with a {narrow.max_aperture} m aperture the torus keeps only: {sorted(spans)}")


if __name__ == "__main__":
    main()
