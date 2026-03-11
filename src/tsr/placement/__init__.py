"""tsr.placement — Stable placement TSR generator.

Usage::

    from tsr.placement import TablePlacer

    placer    = TablePlacer(table_x=0.3, table_y=0.2)
    templates = placer.place_cylinder(cylinder_radius=0.04,
                                      cylinder_height=0.12,
                                      subject="mug")
    tsr  = templates[0].instantiate(table_pose)
    pose = tsr.sample()
"""
from .table_placer import TablePlacer

__all__ = ["TablePlacer"]
