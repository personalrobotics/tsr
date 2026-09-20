# SPDX-License-Identifier: BSD-2-Clause
# Authors: Siddhartha Srinivasa and contributors to TSR

import logging
from functools import reduce

import numpy

from .tsr import NANBW, TSR
from .utils import EPSILON, geodesic_distance

logger = logging.getLogger(__name__)


class TSRChain:
    """
    A sequence of composed TSRs.

    TSRChain allows chaining multiple TSRs together where each TSR's frame
    is relative to the previous one. This is useful for articulated constraints
    like door handles attached to doors.
    """

    def __init__(self, TSR=None, TSRs=None, tsr=None):
        """
        Create a TSR chain from one or more TSRs.

        @param TSR a single TSR to use in this TSR chain
        @param TSRs a list of TSRs to use in this TSR chain
        @param tsr alias for TSR parameter
        """
        self.TSRs = []

        # Handle both TSR and tsr parameters
        single_tsr = TSR if TSR is not None else tsr
        if single_tsr is not None:
            self.append(single_tsr)
        if TSRs is not None:
            for tsr_item in TSRs:
                self.append(tsr_item)

    def append(self, tsr):
        self.TSRs.append(tsr)

    def to_dict(self):
        """Convert TSR chain to a python dict."""
        return {
            "tsrs": [tsr.to_dict() for tsr in self.TSRs],
        }

    @staticmethod
    def from_dict(x):
        """Construct a TSR chain from a python dict."""
        return TSRChain(
            TSRs=[TSR.from_dict(tsr) for tsr in x["tsrs"]],
        )

    def to_json(self):
        """Convert this TSR chain to a JSON string."""
        import json

        return json.dumps(self.to_dict())

    @staticmethod
    def from_json(x, *args, **kw_args):
        """
        Construct a TSR chain from a JSON string.

        This method internally forwards all arguments to `json.loads`.
        """
        import json

        x_dict = json.loads(x, *args, **kw_args)
        return TSRChain.from_dict(x_dict)

    def to_yaml(self):
        """Convert this TSR chain to a YAML string."""
        import yaml

        return yaml.dump(self.to_dict())

    @staticmethod
    def from_yaml(x):
        """Construct a TSR chain from a YAML string."""
        import yaml

        x_dict = yaml.safe_load(x)
        return TSRChain.from_dict(x_dict)

    def is_valid(self, xyzrpy_list, ignoreNAN=False):
        """
        Checks if a xyzrpy list is a valid sample from the TSR.
        @param xyzrpy_list a list of xyzrpy values
        @param ignoreNAN (optional, defaults to False) ignore NaN xyzrpy
        @return a list of 6x1 vector of True if bound is valid and False if not
        """

        if len(self.TSRs) == 0:
            raise ValueError("Cannot validate against empty TSR chain!")

        if len(xyzrpy_list) != len(self.TSRs):
            raise ValueError("Sample must be of equal length to TSR chain!")

        check = []
        for idx in range(len(self.TSRs)):
            check.append(self.TSRs[idx].is_valid(xyzrpy_list[idx], ignoreNAN))

        return check

    def to_transform(self, xyzrpy_list):
        """
        Converts a xyzrpy list into an end-effector transform.

        This implements TSR chain composition as described in Section 5.1 of
        Berenson et al. 2011:
            C_i.T0_w = (C_{i-1}.T0_w) * (C_{i-1}.Tw_sample) * (C_{i-1}.Tw_e)

        The final transform is: T0_sample = Cn.T0_w * Cn.Tw_sample * Cn.Tw_e

        @param xyzrpy_list  a list of xyzrpy values, one per TSR in the chain
        @return trans       4x4 transform
        """
        if len(self.TSRs) == 0:
            raise ValueError("Cannot compute transform for empty TSR chain")

        if len(xyzrpy_list) != len(self.TSRs):
            raise ValueError(f"xyzrpy_list length ({len(xyzrpy_list)}) must match number of TSRs ({len(self.TSRs)})")

        # Clamp values to bounds (required by the L-BFGS-B optimiser that drives
        # distance(); values slightly outside bounds are normal during line search).
        xyzrpy_list_clamped = []
        for idx in range(len(self.TSRs)):
            xyzrpy = numpy.array(xyzrpy_list[idx])
            Bw = self.TSRs[idx]._Bw_cont
            xyzrpy_clamped = numpy.clip(xyzrpy, Bw[:, 0], Bw[:, 1])
            if not numpy.allclose(xyzrpy, xyzrpy_clamped):
                logger.debug(
                    "TSRChain.to_transform: xyzrpy[%d] clamped to Bw (delta=%s)",
                    idx,
                    xyzrpy - xyzrpy_clamped,
                )
            xyzrpy_list_clamped.append(xyzrpy_clamped)

        # Compute the chained transform WITHOUT modifying original TSR objects
        # Start with the first TSR's T0_w
        T0_w_current = self.TSRs[0].T0_w

        for idx in range(len(self.TSRs)):
            tsr = self.TSRs[idx]
            xyzrpy = xyzrpy_list_clamped[idx]

            # Convert xyzrpy to transform in w frame
            Tw_sample = TSR.xyzrpy_to_trans(xyzrpy)

            # Compute end-effector transform: T0_w_current * Tw_sample * Tw_e
            T0_w_current = reduce(numpy.dot, [T0_w_current, Tw_sample, tsr.Tw_e])

        return T0_w_current

    def sample_xyzrpy(self, xyzrpy_list=None, rng=None):
        """
        Samples from Bw to generate a list of xyzrpy samples
        Can specify some values optionally as NaN.

        @param xyzrpy_list   (optional) a list of Bw with float('nan') for
                        dimensions to sample uniformly.
        @param rng      (optional) a numpy.random.Generator for reproducible
                        sampling. Defaults to the global numpy RNG when None.
        @return sample  a list of sampled xyzrpy
        """

        if xyzrpy_list is None:
            xyzrpy_list = [NANBW] * len(self.TSRs)

        sample = []
        for idx in range(len(self.TSRs)):
            sample.append(self.TSRs[idx].sample_xyzrpy(xyzrpy_list[idx], rng=rng))

        return sample

    def sample(self, xyzrpy_list=None, rng=None):
        """
        Samples from the Bw chain to generate an end-effector transform.
        Can specify some Bw values optionally.

        @param xyzrpy_list   (optional) a list of xyzrpy with float('nan') for
                             dimensions to sample uniformly.
        @param rng      (optional) a numpy.random.Generator for reproducible
                        sampling. Defaults to the global numpy RNG when None.
        @return T0_w         4x4 transform
        """
        return self.to_transform(self.sample_xyzrpy(xyzrpy_list, rng=rng))

    def distance(self, trans):
        """
        Computes the Geodesic Distance from the TSR chain to a transform
        @param trans 4x4 transform
        @return dist Geodesic distance to TSR
        @return bwopt Closest Bw value to trans output as a list of xyzrpy
        """
        import scipy.optimize

        n = len(self.TSRs)
        # Continuous bounds over all 6*n chain coordinates. _Bw_cont guarantees
        # lower <= upper even for outer (wrapping) rotation intervals, which
        # L-BFGS-B requires (raw Bw can have lower > upper and crash it).
        lower = numpy.concatenate([self.TSRs[i]._Bw_cont[:, 0] for i in range(n)])
        upper = numpy.concatenate([self.TSRs[i]._Bw_cont[:, 1] for i in range(n)])
        x_full = (lower + upper) / 2.0

        # Only optimise the FREE coordinates. Point-bound coordinates (lower ==
        # upper, e.g. a fixed [0, 0]) are held at their value and never handed to
        # L-BFGS-B: a zero-width finite-difference step there divides by zero,
        # produces a NaN gradient, and leaves the optimiser stalled at the
        # midpoint -- which made a chain reject a pose from its own sample() (#57).
        free = upper > lower

        def expand(x_free):
            x = x_full.copy()
            x[free] = x_free
            return x

        def objective(x_free):
            tsr_trans = self.to_transform(expand(x_free).reshape(n, 6))
            return geodesic_distance(tsr_trans, trans)

        if not numpy.any(free):
            # Fully fixed chain: a single candidate pose, nothing to optimise.
            return float(objective(numpy.empty(0))), x_full.reshape(n, 6)

        bounds = list(zip(lower[free], upper[free]))
        xopt_free, dist, _info = scipy.optimize.fmin_l_bfgs_b(
            objective, x_full[free], fprime=None, args=(), bounds=bounds, approx_grad=True
        )
        return dist, expand(xopt_free).reshape(n, 6)

    def contains(self, trans):
        """
        Checks if the TSR chain contains the transform.

        A chain is the set of poses reachable by **serially composing** a
        transform sampled from each component TSR -- not the Boolean
        intersection of the components' world-frame pose sets. Membership is
        tested via the composed ``distance()`` (they are consistent). For a
        single TSR this reduces to ``TSR.contains()``.

        @param  trans 4x4 transform
        @return       True if inside and False if not
        """
        if len(self.TSRs) == 0:
            return False
        # A single-TSR chain is exactly one TSR: use the closed-form check, which
        # is exact and avoids the optimiser stalling at the non-smooth geodesic
        # minimum (arccos kink at angle 0).
        if len(self.TSRs) == 1:
            return self.TSRs[0].contains(trans)
        dist, _ = self.distance(trans)
        return abs(dist) < EPSILON

    def to_xyzrpy(self, trans):
        """
        Converts an end-effector transform to a list of xyzrpy values
        @param  trans  4x4 transform
        @return xyzrpy_list list of xyzrpy values
        """
        _, xyzrpy_array = self.distance(trans)
        # Convert numpy array to list of arrays
        return [xyzrpy_array[i] for i in range(len(self.TSRs))]
