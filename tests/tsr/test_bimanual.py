import numpy as np
import pytest
from gafropy import Motor
from tsr import TSR
from tsr.bimanual import BimanualPose, BimanualTSR


def _free_tsr():
    # A TSR free in all 6 DOF around identity (a wide box).
    Bw = np.array([[-np.pi, np.pi]] * 3 + [[-1.0, 1.0]] * 3)
    return TSR(T0_w=np.eye(4), Tw_e=np.eye(4), Bw=Bw)


def _fixed_tsr(motor):
    # A TSR pinned to a single pose (zero-width box), centered on `motor`.
    Bw = np.zeros((6, 2))
    return TSR(T0_w=motor, Tw_e=np.eye(4), Bw=Bw)


def test_requires_at_least_one_component():
    with pytest.raises(ValueError):
        BimanualTSR(absolute=None, relative=None)


def test_sample_both_present_returns_two_motors():
    bt = BimanualTSR(absolute=_free_tsr(), relative=_free_tsr())
    pose = bt.sample()
    assert isinstance(pose.absolute, Motor)
    assert isinstance(pose.relative, Motor)


def test_sample_relative_only_leaves_absolute_none():
    bt = BimanualTSR(relative=_free_tsr())
    pose = bt.sample()
    assert pose.absolute is None
    assert isinstance(pose.relative, Motor)


def test_distance_zero_when_pose_inside_both():
    abs_tsr = _free_tsr()
    rel_tsr = _free_tsr()
    bt = BimanualTSR(absolute=abs_tsr, relative=rel_tsr)
    pose = bt.sample()
    dist, _ = bt.distance(pose)
    assert dist == pytest.approx(0.0, abs=1e-6)


def test_distance_sums_present_components_only():
    # Fixed absolute at identity; a pose offset in absolute costs; relative absent.
    bt = BimanualTSR(absolute=_fixed_tsr(Motor()))
    offset = Motor.exp(0.0, 0.0, 0.0, 0.5, 0.0, 0.0)  # 0.5 m in x
    pose = BimanualPose(absolute=offset, relative=None)
    dist, witness = bt.distance(pose)
    assert dist == pytest.approx(0.5, abs=1e-6)
    assert witness.relative is None


def test_volume_sums_present_components():
    abs_tsr = _free_tsr()
    rel_tsr = _fixed_tsr(Motor())  # zero volume
    both = BimanualTSR(absolute=abs_tsr, relative=rel_tsr)
    abs_only = BimanualTSR(absolute=abs_tsr)
    assert both.volume == pytest.approx(abs_only.volume)
