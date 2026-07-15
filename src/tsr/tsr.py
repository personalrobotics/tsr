# SPDX-License-Identifier: BSD-2-Clause
# Authors: Siddhartha Srinivasa and contributors to TSR

import numpy
import numpy.random
from numpy import pi

from gafropy import Motor

from .constraints.base import Constraint
from .utils import EPSILON, geodesic_distance

# bw / Bw are in gafro ``Motor.log()`` order: [b12, b13, b23, e1i, e2i, e3i] =
# [rotor bivector (3), translation (3)]. Index helpers keep that explicit.
_ROT = slice(0, 3)
_TRANS = slice(3, 6)

NANBW = numpy.ones(6) * float("nan")


def _load_transform(value):
    """Deserialize a transform stored as a 6-vector bivector log or a 4x4 matrix.

    Returns a ``Motor``. Length-6 values are read as a bivector log; 4x4 nested
    lists are read as the legacy matrix format.
    """
    return Motor(numpy.asarray(value, dtype=float))


class TSR(Constraint):
    """
    Core Task Space Region (TSR) class — geometry-only, robot-agnostic.

    A TSR is defined by a transform T0_w to the TSR frame, a transform Tw_e
    from the TSR frame to the end-effector, and a bounding box Bw over 6 DoFs.

    **Coordinate system (CGA split).** Every rigid transform is a gafropy
    ``Motor`` ``M = Translator(t) * Rotor(r)``. The 6-vector ``bw`` and the box
    ``Bw`` are split accordingly:

        bw = [tx, ty, tz,  b12, b13, b23]
             └ translation ┘ └ rotor-bivector log (axis*angle) ┘

    The rotation half is the rotor bivector log (``Rotor.log()``), i.e. the
    rotation axis scaled by the rotation angle, with the angle wrapped to
    ``(-pi, pi]``. A box on these three components is a decoupled, axis-clean
    rotation region (e.g. "rotate freely about z" is ``b23 in [-pi, pi]``). This
    replaces the legacy Euler ``[roll, pitch, yaw]`` parametrization.

    Both constructor arguments and methods that take a transform accept either a
    ``Motor`` or a 4x4 numpy matrix (or a 6-vector bivector log); methods that
    return a transform return a ``Motor``.
    """

    def __init__(self, T0_w=None, Tw_e=None, Bw=None):
        if T0_w is None:
            T0_w = numpy.eye(4)
        if Tw_e is None:
            Tw_e = numpy.eye(4)
        if Bw is None:
            Bw = numpy.zeros((6, 2))

        self.T0_w = Motor(T0_w)
        self.Tw_e = Motor(Tw_e)
        self.Bw = numpy.array(Bw, dtype=float)

        if self.Bw.shape != (6, 2):
            raise ValueError("Bw must be shape (6,2)", Bw)
        if numpy.any(self.Bw[:, 0] > self.Bw[:, 1] + EPSILON):
            raise ValueError("Bw bounds must be [min, max] for every row", Bw)

        # Continuous bound. Translation rows pass through unchanged. Rotation
        # (bivector) rows are clamped to the representable range [-pi, pi]: the
        # rotor bivector log angle is wrapped to (-pi, pi], so a component can
        # never exceed pi in magnitude. A box of [-pi, pi] is a full turn. In the
        # Motor.log order the rotor bivector occupies rows 0:3.
        Bw_cont = numpy.copy(self.Bw)
        Bw_cont[_ROT, 0] = numpy.clip(Bw_cont[_ROT, 0], -pi, pi)
        Bw_cont[_ROT, 1] = numpy.clip(Bw_cont[_ROT, 1], -pi, pi)
        self._Bw_cont = Bw_cont

    def __repr__(self) -> str:
        _DOF = ("b12", "b13", "b23", "x", "y", "z")
        free = [_DOF[i] for i in range(6) if not numpy.isclose(self.Bw[i, 0], self.Bw[i, 1])]
        t0 = numpy.asarray(self.T0_w.get_translator().to_array())
        te = numpy.asarray(self.Tw_e.get_translator().to_array())
        free_str = ",".join(free) if free else "fixed"
        return f"TSR(free=[{free_str}], T0_w.t={t0.round(3)}, Tw_e.t={te.round(3)})"

    @property
    def volume(self) -> float:
        """Union-sampling weight: summed Bw width, rotation rows clamped to 2π.

        Rows 0:3 are the rotor-bivector (rotation) DOF in Motor.log order; a full
        turn is 2π, so they are clamped there to keep a free rotation from
        dominating the weight. Rows 3:6 (translation) are summed as-is.
        """
        import numpy as _np
        widths = _np.asarray(self.Bw[:, 1] - self.Bw[:, 0], dtype=float)
        widths[0:3] = _np.minimum(widths[0:3], 2.0 * _np.pi)
        widths = _np.maximum(widths, 0.0)
        return float(_np.sum(widths))

    # ------------------------------------------------------------------
    # bw <-> transform
    # ------------------------------------------------------------------

    def bw_to_trans(self, bw):
        """Convert a 6-vector ``bw`` (the Motor bivector log) into a Motor in the
        TSR ``w`` frame. ``bw`` is exactly ``Motor.log()`` order, so this is
        ``Motor.exp(*bw)``."""
        bw = numpy.asarray(bw, dtype=float).reshape(6)
        return Motor.exp(*(float(v) for v in bw))

    def to_transform(self, bw):
        """
        Converts a 6-vector ``bw`` (split translation + rotor-bivector) into an
        end-effector transform.

        @param  bw     [tx, ty, tz, b12, b13, b23]
        @return trans  Motor end-effector transform
        """
        if len(bw) != 6:
            raise ValueError("bw must be of length 6")
        validity = self.is_valid(bw)
        if not all(validity):
            violated = [i for i, v in enumerate(validity) if not v]
            raise ValueError(
                f"bw violates bounds at dimensions {violated}: bw={bw}, bounds={self._Bw_cont[violated]}"
            )
        Tw = self.bw_to_trans(bw)
        return self.T0_w.multiply(Tw).multiply(self.Tw_e)

    def to_bw(self, trans):
        """Convert an end-effector transform to its ``bw`` 6-vector.

        Implements Berenson et al. 2011 Eqs. 5-6 in CGA:
            Tw_s' = (T0_w)^-1 * T0_s * (Tw_e)^-1
        and returns its ``Motor.log()`` (the ``bw`` coordinate system).
        """
        trans = Motor(trans)
        Tw_s_prime = self.T0_w.inverse().multiply(trans).multiply(self.Tw_e.inverse())
        return numpy.asarray(Tw_s_prime.log(), dtype=float)

    def is_valid(self, bw, ignoreNAN=False):
        """
        Checks if a ``bw`` 6-vector is within the TSR bounds.

        @param bw 6-vector [b12, b13, b23, e1i, e2i, e3i] (Motor.log order)
        @param ignoreNAN (optional) ignore NaN components
        @return a 6-vector of True where the bound is satisfied
        """
        bw = numpy.asarray(bw, dtype=float)
        check = numpy.array(
            [
                ((bw[i] + EPSILON) >= self._Bw_cont[i, 0]) and ((bw[i] - EPSILON) <= self._Bw_cont[i, 1])
                for i in range(6)
            ]
        )
        if ignoreNAN:
            check |= numpy.isnan(bw)
        return check

    def contains(self, trans):
        """
        Checks if the TSR contains the transform.

        @param  trans  Motor or 4x4 transform
        @return True if transform is within TSR bounds, False otherwise
        """
        return all(self.is_valid(self.to_bw(trans)))

    def _displacement_to_tsr(self, trans):
        """
        Compute the displacement vector from a transform to the TSR.

        In the split parametrization the displacement is simply, per component,
        the signed distance outside the box (0 if inside). The rotation half is
        the rotor-bivector log, which has no Euler redundancy, so the
        9-candidate enumeration of the legacy RPY implementation is gone.

        @param trans Motor or 4x4 transform (T0_s — pose in world frame)
        @return dx 6-vector displacement to TSR
        @return dw 6-vector clamped split coords in the w frame (the bwopt seed)
        """
        dw = self.to_bw(trans)

        dx = numpy.zeros(6)
        for i in range(6):
            if dw[i] < self._Bw_cont[i, 0]:
                dx[i] = dw[i] - self._Bw_cont[i, 0]
            elif dw[i] > self._Bw_cont[i, 1]:
                dx[i] = dw[i] - self._Bw_cont[i, 1]
        return dx, dw

    def distance(self, trans, rotation_weight=1.0):
        """
        Computes the distance from the TSR to a transform.

        Translation is in meters, rotation in radians (rotor-bivector log norm).

        @param trans Motor or 4x4 transform
        @param rotation_weight weight for rotation vs translation (default 1.0)
        @return dist Distance to TSR (0 if transform is inside TSR)
        @return bwopt Closest ``bw`` value to trans (6-vector)
        """
        dx, dw = self._displacement_to_tsr(trans)

        dx_weighted = dx.copy()
        dx_weighted[_ROT] *= rotation_weight
        dist = float(numpy.linalg.norm(dx_weighted))

        bwopt = numpy.clip(dw, self._Bw_cont[:, 0], self._Bw_cont[:, 1])
        return dist, bwopt

    def distance_optimize(self, trans):
        """
        Computes the geodesic distance from the TSR to a transform using
        numerical optimization. Slower but uses the full SE(3) geodesic metric.

        @param trans Motor or 4x4 transform
        @return dist Geodesic distance to TSR
        @return bwopt Closest ``bw`` value to trans
        """
        trans = Motor(trans)
        if self.contains(trans):
            return 0.0, self.to_bw(trans)

        import scipy.optimize

        def objective(bw):
            bwtrans = self.to_transform(bw)
            return geodesic_distance(bwtrans, trans)

        bwinit = (self._Bw_cont[:, 0] + self._Bw_cont[:, 1]) / 2
        bwbounds = [(self._Bw_cont[i, 0], self._Bw_cont[i, 1]) for i in range(6)]

        bwopt, dist, info = scipy.optimize.fmin_l_bfgs_b(
            objective, bwinit, fprime=None, args=(), bounds=bwbounds, approx_grad=True
        )
        return dist, bwopt

    # ------------------------------------------------------------------
    # sampling
    # ------------------------------------------------------------------

    def sample_bw(self, bw=NANBW):
        """
        Samples from Bw to generate a ``bw`` 6-vector. Components given as NaN
        are sampled uniformly within their bound; finite components are kept.

        @param bw (optional) a 6-vector with float('nan') for free dimensions
        @return a sampled ``bw`` 6-vector
        """
        check = self.is_valid(bw, ignoreNAN=True)
        if not all(check):
            raise ValueError("bw must be within bounds", check)

        return numpy.array(
            [
                self._Bw_cont[i, 0] + (self._Bw_cont[i, 1] - self._Bw_cont[i, 0]) * numpy.random.random_sample()
                if numpy.isnan(x)
                else x
                for i, x in enumerate(bw)
            ]
        )

    def sample(self, bw=NANBW):
        """
        Samples from Bw to generate an end-effector transform.

        @param bw (optional) a 6-vector with float('nan') for free dimensions
        @return Motor transform
        """
        return self.to_transform(self.sample_bw(bw))

    # ------------------------------------------------------------------
    # serialization
    # ------------------------------------------------------------------

    def to_dict(self):
        """Convert this TSR to a python dict.

        Transforms are serialized as their 6-vector bivector log
        (``Motor.get_log()``); Bw remains a 6x2 list (split coords).
        """
        return {
            "format": "cga-split-v1",
            "T0_w": numpy.asarray(self.T0_w.log(), dtype=float).tolist(),
            "Tw_e": numpy.asarray(self.Tw_e.log(), dtype=float).tolist(),
            "Bw": self.Bw.tolist(),
        }

    @staticmethod
    def from_dict(x):
        """Construct a TSR from a python dict.

        ``T0_w`` / ``Tw_e`` accept both the bivector-log (length-6) and legacy
        4x4-matrix formats. ``Bw`` is interpreted as split coords
        ``[tx,ty,tz,b12,b13,b23]``; dicts lacking the ``cga-split-v1`` format
        marker predate the CGA migration and are rejected to avoid silently
        reinterpreting Euler-RPY bounds as bivector bounds.
        """
        fmt = x.get("format")
        if fmt is not None and fmt != "cga-split-v1":
            raise ValueError(f"Unsupported TSR serialization format: {fmt!r}")
        if fmt is None:
            raise ValueError(
                "TSR dict has no 'format' marker; this looks like a legacy "
                "Euler-RPY TSR whose Bw bounds are NOT compatible with the CGA "
                "split parametrization. Re-author it as 'cga-split-v1'."
            )
        return TSR(
            T0_w=_load_transform(x["T0_w"]),
            Tw_e=_load_transform(x["Tw_e"]),
            Bw=numpy.array(x["Bw"]),
        )

    def to_json(self):
        """Convert this TSR to a JSON string."""
        import json

        return json.dumps(self.to_dict())

    @staticmethod
    def from_json(x, *args, **kw_args):
        """
        Construct a TSR from a JSON string.

        This method internally forwards all arguments to `json.loads`.
        """
        import json

        x_dict = json.loads(x, *args, **kw_args)
        return TSR.from_dict(x_dict)

    def to_yaml(self):
        """Convert this TSR to a YAML string."""
        import yaml

        return yaml.dump(self.to_dict())

    @staticmethod
    def from_yaml(x):
        """Construct a TSR from a YAML string."""
        import yaml

        x_dict = yaml.safe_load(x)
        return TSR.from_dict(x_dict)
