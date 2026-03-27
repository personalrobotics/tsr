"""tsr.placement — Stable placement TSR generator.

Usage::

    from tsr.placement import StablePlacer

    placer    = StablePlacer(table_x=0.3, table_y=0.2)
    templates = placer.place_cylinder(cylinder_radius=0.04,
                                      cylinder_height=0.12,
                                      subject="mug")
    tsr  = templates[0].instantiate(surface_pose)
    pose = tsr.sample()
"""
from .stable_placer import StablePlacer

# Deprecated alias — will be removed in a future release.
TablePlacer = StablePlacer

__all__ = ["StablePlacer", "TablePlacer"]
