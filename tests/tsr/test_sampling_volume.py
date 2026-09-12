import numpy as np

from tsr import TSR


def test_tsr_volume_equals_interval_sum():
    Bw = np.array([[-1.0, 1.0], [0.0, 0.0], [0.0, 0.0],
                   [-0.5, 0.5], [0.0, 0.0], [0.0, 0.0]])
    tsr = TSR(Bw=Bw)
    # (2.0) + (1.0) = 3.0
    assert tsr.volume == 3.0


def test_tsr_volume_clamps_rotation_row_to_two_pi():
    # Row 0 is a rotation (rotor-bivector) DOF. A span wider than a full turn
    # must contribute at most 2π to the volume weight.
    Bw = np.zeros((6, 2))
    Bw[0] = [-10.0, 10.0]  # rotation row, width 20 >> 2π
    tsr = TSR(Bw=Bw)
    assert tsr.volume == np.pi * 2.0


def test_tsr_volume_does_not_clamp_translation_row():
    # Row 3 is a translation DOF: a wide span is a real 10 m box and must
    # contribute its full width, NOT be clamped to 2π.
    Bw = np.zeros((6, 2))
    Bw[3] = [-5.0, 5.0]  # translation row, width 10
    tsr = TSR(Bw=Bw)
    assert tsr.volume == 10.0
