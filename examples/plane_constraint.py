#!/usr/bin/env python
"""
Plane Constraint Example: the new geometric-primitive constraint.

A ``PlaneConstraint`` keeps the end-effector *origin* on a CGA plane (and,
optionally, pins its orientation). Unlike a box-shaped ``TSR``, it is defined
directly by a gafro geometric primitive (:class:`gafro.Plane`).

This example demonstrates:
- Building a plane constraint from a point + normal (and from a raw gafro Plane)
- Computing the distance from a pose to the plane
- Projecting an off-plane pose onto the constraint manifold
- Sampling poses that lie on the plane
- Pinning the orientation
- Serializing / deserializing a constraint

Every ``PlaneConstraint`` is a :class:`tsr.Constraint`, exactly like ``TSR``,
so it plugs into the same planner seam (``distance`` / ``to_transform`` /
``sample``).
"""

import numpy as np
from numpy import pi

from gafro import Motor, Plane, Point, Rotor
from tsr import Constraint, PlaneConstraint


def pose_at(x, y, z, biv=(0.0, 0.0, 0.0)):
    """Build a world-frame end-effector Motor at (x, y, z) with rotor-bivector log ``biv``."""
    rotor = Rotor.exp(float(biv[0]), float(biv[1]), float(biv[2]))
    return Motor.from_translation_rotor(float(x), float(y), float(z), rotor)


def position_of(motor):
    """World-frame position (3-vector) of a Motor, from its Translator."""
    t = motor.get_translator()
    return np.array([t.x(), t.y(), t.z()])


def bivector_log_of(motor):
    """Rotor-bivector log (3-vector [b12, b13, b23]) of a Motor's rotation."""
    return np.asarray(motor.get_rotor().log().to_array(), dtype=float)


def main():
    print("Plane Constraint Example")
    print("=" * 50)

    # ------------------------------------------------------------------
    # 1. Build a constraint: the table-top plane z = 0.3, normal +z.
    # ------------------------------------------------------------------
    print("\n1. Construction")
    print("-" * 30)
    table = PlaneConstraint.from_point_normal(point=[0.0, 0.0, 0.3], normal=[0.0, 0.0, 1.0])
    print(f"   from point+normal: {table}")
    print(f"   is a Constraint:   {isinstance(table, Constraint)}")

    # You can also wrap a gafro Plane directly (here: the y=0 plane via 3 points).
    raw_plane = Plane(Point(0, 0, 0), Point(1, 0, 0), Point(0, 0, 1))
    wall = PlaneConstraint(raw_plane)
    print(f"   from gafro.Plane: {wall}")

    # ------------------------------------------------------------------
    # 2. Distance: 0 on the plane, perpendicular offset otherwise.
    # ------------------------------------------------------------------
    print("\n2. Distance to the plane")
    print("-" * 30)
    on_plane = pose_at(0.4, -0.2, 0.3)  # z == 0.3
    off_plane = pose_at(0.4, -0.2, 0.55)  # 0.25 above the table
    d_on, _ = table.distance(on_plane)
    d_off, _ = table.distance(off_plane)
    print(f"   pose on the plane:  distance = {d_on:.4f}")
    print(f"   pose 0.25 above:    distance = {d_off:.4f}")
    print(f"   contains(on):  {table.contains(on_plane, tolerance=1e-6)}")
    print(f"   contains(off): {table.contains(off_plane, tolerance=1e-6)}")

    # ------------------------------------------------------------------
    # 3. Projection: distance() returns a witness; to_transform() maps it back.
    # ------------------------------------------------------------------
    print("\n3. Projecting a pose onto the plane")
    print("-" * 30)
    _, witness = table.distance(off_plane)
    projected = table.to_transform(witness)
    pxyz = position_of(projected)
    print(f"   off-plane position:  {position_of(off_plane).round(3)}")
    print(f"   projected position:  {pxyz.round(3)}  (z snapped to 0.3)")
    print(f"   projected distance:  {table.distance(projected)[0]:.4f}")

    # ------------------------------------------------------------------
    # 4. Sampling: every sample lies on the plane.
    # ------------------------------------------------------------------
    print("\n4. Sampling poses on the plane")
    print("-" * 30)
    np.random.seed(0)
    for i in range(3):
        s = table.sample()
        pos = position_of(s)
        dist, _ = table.distance(s)
        print(f"   sample {i + 1}: pos=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]  distance={dist:.4f}")

    # ------------------------------------------------------------------
    # 5. Pinning orientation: rotation error now contributes to distance.
    # ------------------------------------------------------------------
    print("\n5. Orientation pinning")
    print("-" * 30)
    yaw90 = [0.0, 0.0, pi / 2]  # rotor-bivector log: 90 deg about z
    pinned = PlaneConstraint.from_point_normal([0, 0, 0.3], [0, 0, 1.0], orientation=yaw90)
    aligned = pose_at(0.1, 0.1, 0.3, biv=yaw90)
    misaligned = pose_at(0.1, 0.1, 0.3, biv=(0.0, 0.0, 0.0))
    print(f"   aligned pose:     distance = {pinned.distance(aligned)[0]:.4f}")
    print(f"   misaligned pose:  distance = {pinned.distance(misaligned)[0]:.4f}")
    # Projection restores the pinned orientation.
    _, w = pinned.distance(misaligned)
    biv = bivector_log_of(pinned.to_transform(w))
    print(f"   projected orientation (bivector log): {biv.round(3)}  (target {np.round(yaw90, 3)})")

    # ------------------------------------------------------------------
    # 6. Serialization round-trip.
    # ------------------------------------------------------------------
    print("\n6. Serialization")
    print("-" * 30)
    d = table.to_dict()
    print(f"   to_dict: {d}")
    restored = PlaneConstraint.from_dict(d)
    print(f"   restored: {restored}")
    print(f"   same distance on a probe: {np.isclose(table.distance(off_plane)[0], restored.distance(off_plane)[0])}")

    print("\nDone!")


if __name__ == "__main__":
    main()
