"""Interactive cylinder grasp inspection with Viser — the #77 evaluation example.

Opens a browser-based viewer showing, for a parallel-jaw side grasp of a cylinder:

  * the reference cylinder, in the template's own frame (z from 0 to height);
  * reproducibly sampled end-effector poses from each template's continuous Bw;
  * the canonical end-effector axes per pose (x red, y green, z blue), so approach
    (z) and jaw opening (y) are readable directly;
  * the idealized jaw geometry at each pose, drawn at the template's preshape.

Drag to orbit; the poses spread over the yaw freedom and the discrete depths.

With --explore, one grasp is driven instead by a slider per free Bw coordinate, so the
continuous freedom is scrubbed rather than sampled: pick a depth from the dropdown,
drag z and yaw, and optionally overlay the sampled cloud for comparison.

Requires the experimental extra:  pip install "sstsr[viser]"   (uv sync --extra viser)

Usage:
    uv run python examples/viser_cylinder_grasps.py
    uv run python examples/viser_cylinder_grasps.py --port 8080 --samples 8

Over SSH, forward the port and open the URL locally:
    ssh -L 8080:localhost:8080 user@host
"""

import argparse

from tsr.hands import ParallelJawGripper
from tsr.viser import explore_templates, free_coordinates, show_templates

RADIUS, HEIGHT = 0.03, 0.12


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--samples", type=int, default=2, help="poses sampled per template")
    parser.add_argument("--seed", type=int, default=0, help="explicit seed, so a view is reproducible")
    parser.add_argument("--explore", action="store_true", help="slider-driven single grasp instead of a pose cloud")
    args = parser.parse_args()

    gripper = ParallelJawGripper(finger_length=0.08, max_aperture=0.14)
    templates = gripper.grasp_cylinder_side(RADIUS, HEIGHT)
    print(f"{len(templates)} side templates x {args.samples} samples = {len(templates) * args.samples} poses")
    for t in templates:
        p = t.provenance
        family = f"{p.mode}/{p.approach}/{p.finger_orientation}"
        print(f"  {family} depth {p.depth:.3f} ({p.depth_index + 1}/{p.depth_count})")

    if args.explore:
        free = ", ".join(label for _, label, _, _ in free_coordinates(templates[0]))
        print(f"\nslider coordinates: {free}")
        server = explore_templates(templates, cylinder=(RADIUS, HEIGHT), gripper=gripper, port=args.port)
    else:
        server = show_templates(
            templates,
            cylinder=(RADIUS, HEIGHT),
            gripper=gripper,
            n_per_template=args.samples,
            seed=args.seed,
            port=args.port,
        )
    print(f"\nOpen http://localhost:{args.port} — Ctrl-C to stop.")
    try:
        server.sleep_forever()
    except KeyboardInterrupt:
        server.stop()


if __name__ == "__main__":
    main()
