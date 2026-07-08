# SPDX-License-Identifier: BSD-2-Clause
# Authors: Siddhartha Srinivasa and contributors to TSR

import logging

import numpy

from gafropy import Motor

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

    def is_valid(self, bw_list, ignoreNAN=False):
        """
        Checks if a ``bw`` list is a valid sample from the TSR chain.
        @param bw_list a list of ``bw`` 6-vectors (split coords)
        @param ignoreNAN (optional, defaults to False) ignore NaN components
        @return a list of 6x1 vector of True if bound is valid and False if not
        """

        if len(self.TSRs) == 0:
            raise ValueError("Cannot validate against empty TSR chain!")

        if len(bw_list) != len(self.TSRs):
            raise ValueError("Sample must be of equal length to TSR chain!")

        check = []
        for idx in range(len(self.TSRs)):
            check.append(self.TSRs[idx].is_valid(bw_list[idx], ignoreNAN))

        return check

    def to_transform(self, bw_list):
        """
        Converts a ``bw`` list into an end-effector transform.

        This implements TSR chain composition as described in Section 5.1 of
        Berenson et al. 2011:
            C_i.T0_w = (C_{i-1}.T0_w) * (C_{i-1}.Tw_sample) * (C_{i-1}.Tw_e)

        The final transform is: T0_sample = Cn.T0_w * Cn.Tw_sample * Cn.Tw_e

        @param bw_list  a list of ``bw`` 6-vectors, one per TSR in the chain
        @return trans   Motor transform
        """
        if len(self.TSRs) == 0:
            raise ValueError("Cannot compute transform for empty TSR chain")

        if len(bw_list) != len(self.TSRs):
            raise ValueError(f"bw_list length ({len(bw_list)}) must match number of TSRs ({len(self.TSRs)})")

        # Clamp values to bounds (required by the L-BFGS-B optimiser that drives
        # distance(); values slightly outside bounds are normal during line search).
        bw_list_clamped = []
        for idx in range(len(self.TSRs)):
            bw = numpy.array(bw_list[idx])
            Bw = self.TSRs[idx]._Bw_cont
            bw_clamped = numpy.clip(bw, Bw[:, 0], Bw[:, 1])
            if not numpy.allclose(bw, bw_clamped):
                logger.debug(
                    "TSRChain.to_transform: bw[%d] clamped to Bw (delta=%s)",
                    idx,
                    bw - bw_clamped,
                )
            bw_list_clamped.append(bw_clamped)

        # Compute the chained transform WITHOUT modifying original TSR objects
        # Start with the first TSR's T0_w
        T0_w_current = self.TSRs[0].T0_w

        for idx in range(len(self.TSRs)):
            tsr = self.TSRs[idx]
            bw = bw_list_clamped[idx]

            # bw is in Motor.log() order; exp maps it to a transform in the w frame
            Tw_sample = Motor.exp(*(float(v) for v in bw))

            # Compute end-effector transform: T0_w_current * Tw_sample * Tw_e
            T0_w_current = T0_w_current.multiply(Tw_sample).multiply(tsr.Tw_e)

        return T0_w_current

    def sample_bw(self, bw_list=None):
        """
        Samples from Bw to generate a list of ``bw`` samples.
        Can specify some components optionally as NaN.

        @param bw_list   (optional) a list of ``bw`` 6-vectors with float('nan')
                        for dimensions to sample uniformly.
        @return sample  a list of sampled ``bw`` 6-vectors
        """

        if bw_list is None:
            bw_list = [NANBW] * len(self.TSRs)

        sample = []
        for idx in range(len(self.TSRs)):
            sample.append(self.TSRs[idx].sample_bw(bw_list[idx]))

        return sample

    def sample(self, bw_list=None):
        """
        Samples from the Bw chain to generate an end-effector transform.
        Can specify some Bw values optionally.

        @param bw_list   (optional) a list of ``bw`` 6-vectors with float('nan')
                             for dimensions to sample uniformly.
        @return T0_w         Motor transform
        """
        return self.to_transform(self.sample_bw(bw_list))

    def distance(self, trans):
        """
        Computes the Geodesic Distance from the TSR chain to a transform
        @param trans Motor or 4x4 transform
        @return dist Geodesic distance to TSR
        @return bwopt Closest Bw value to trans output as a list of ``bw`` vecs
        """
        import scipy.optimize

        trans = Motor(trans)

        def objective(bw_list):
            bw_stack = bw_list.reshape(len(self.TSRs), 6)
            tsr_trans = self.to_transform(bw_stack)
            return geodesic_distance(tsr_trans, trans)

        # Seed each TSR's block from its own closest-bw witness (TSR.distance
        # returns the box-clamped projection of ``trans``), not the box midpoint.
        # This puts L-BFGS-B's start at (or near) the true minimum, so its
        # finite-difference search converges reliably even near box corners.
        bwinit = []
        bwbounds = []
        for tsr in self.TSRs:
            _, bwseed = tsr.distance(trans)
            bwinit.extend(bwseed)
            bwbounds.extend([(tsr.Bw[i, 0], tsr.Bw[i, 1]) for i in range(6)])

        bwopt, dist, info = scipy.optimize.fmin_l_bfgs_b(
            objective, bwinit, fprime=None, args=(), bounds=bwbounds, approx_grad=True
        )
        return dist, bwopt.reshape(len(self.TSRs), 6)

    def contains(self, trans):
        """
        Checks if the TSR chain contains the transform.

        Uses the composed distance (consistent with ``distance()``).
        For a single TSR this is equivalent to ``TSR.contains()``.
        For multi-TSR chains, the transform must satisfy all constraints
        simultaneously (AND semantics).

        @param  trans Motor or 4x4 transform
        @return       True if inside and False if not
        """
        if len(self.TSRs) == 0:
            return False
        dist, _ = self.distance(trans)
        return abs(dist) < EPSILON

    def to_bw(self, trans):
        """
        Converts an end-effector transform to a list of ``bw`` 6-vectors.
        @param  trans  Motor or 4x4 transform
        @return bw_list list of ``bw`` 6-vectors (split coords)
        """
        _, bw_array = self.distance(trans)
        # Convert numpy array to list of arrays
        return [bw_array[i] for i in range(len(self.TSRs))]
