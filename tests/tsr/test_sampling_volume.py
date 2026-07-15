import numpy as np
from tsr import TSR


def test_tsr_volume_equals_interval_sum():
    Bw = np.array([[-1.0, 1.0], [0.0, 0.0], [0.0, 0.0],
                   [-0.5, 0.5], [0.0, 0.0], [0.0, 0.0]])
    tsr = TSR(Bw=Bw)
    # (2.0) + (1.0) = 3.0
    assert tsr.volume == 3.0
