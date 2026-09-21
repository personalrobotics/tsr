# SPDX-License-Identifier: BSD-2-Clause
# Authors: Siddhartha Srinivasa and contributors to TSR

import logging
from dataclasses import dataclass
from functools import reduce

import numpy

from .tsr import NANBW, TSR
from .utils import EPSILON, geodesic_distance

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChainSample:
    """A sampled chain pose together with the coordinates that constructed it.

    ``coordinates`` (shape ``(n, 6)``) is a *constructive witness*: composing it
    forward reproduces ``pose`` exactly, so membership needs no optimizer (see
    :meth:`TSRChain.validate_witness`).
    """

    pose: numpy.ndarray
    coordinates: numpy.ndarray


@dataclass(frozen=True)
class ChainSolveResult:
    """Result of an inverse chain solve (:meth:`TSRChain.solve`).

    ``status`` is ``"satisfied"`` when a witness within tolerance was found, or
    ``"not_found"`` when the bounded numerical search did not find one.
    ``"not_found"`` is **not** a certificate of non-membership — a finite set of
    local solves cannot prove global infeasibility for a rotation-rich chain.
    ``residual`` is the geodesic residual at ``coordinates`` (an upper bound on
    the true minimum). ``nfev``/``restarts`` expose the search cost.
    """

    status: str
    coordinates: numpy.ndarray
    residual: float
    nfev: int
    restarts: int


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

    # Default bounded multi-start budget for the inverse chain solve (#85).
    _DEFAULT_MAX_RESTARTS = 8
    _DEFAULT_MAX_NFEV = 200

    def sample_with_witness(self, xyzrpy_list=None, rng=None):
        """Sample a pose and return it with the coordinates that constructed it.

        Same sampling and one forward composition as :meth:`sample`; performs no
        optimisation. The returned coordinates are a constructive witness, so
        membership is checkable exactly via :meth:`validate_witness` without an
        inverse solve (#85).
        """
        coords = numpy.array(self.sample_xyzrpy(xyzrpy_list, rng=rng))
        return ChainSample(pose=self.to_transform(coords), coordinates=coords)

    def validate_witness(self, trans, coordinates, tolerance=EPSILON):
        """Exact positive certificate: are ``coordinates`` a valid witness for ``trans``?

        Validates the coordinates against each component's bounds and recomposes
        the pose once, returning True iff the recomposition matches ``trans``
        within ``tolerance``. Uses no optimiser and no SciPy (#85).
        """
        coordinates = numpy.asarray(coordinates, dtype=float)
        n = len(self.TSRs)
        if coordinates.shape != (n, 6):
            return False
        for i in range(n):
            if not all(self.TSRs[i].is_valid(coordinates[i])):
                return False
        return bool(geodesic_distance(self.to_transform(coordinates), trans) < tolerance)

    def solve(self, trans, initial_guess=None, max_restarts=None, max_nfev=None, tolerance=EPSILON):
        """Inverse chain solve: find coordinates whose composition matches ``trans``.

        Returns a :class:`ChainSolveResult`. A valid ``initial_guess`` (a retained
        witness, or a neighbouring state's coordinates) is checked first and
        short-circuits the search. Otherwise a bounded, deterministic multi-start
        solve runs, exiting as soon as a tolerance-satisfying witness is found.
        ``"not_found"`` means no witness within the budget -- **never** a proof of
        non-membership, which a finite set of local solves cannot establish for a
        rotation-rich chain (#85).
        """
        import scipy.optimize

        max_restarts = self._DEFAULT_MAX_RESTARTS if max_restarts is None else max_restarts
        max_nfev = self._DEFAULT_MAX_NFEV if max_nfev is None else max_nfev
        n = len(self.TSRs)
        if n == 0:
            return ChainSolveResult("not_found", numpy.empty((0, 6)), float("inf"), 0, 0)

        # Fast path: a supplied witness that already validates.
        if initial_guess is not None and self.validate_witness(trans, initial_guess, tolerance):
            coords = numpy.asarray(initial_guess, dtype=float)
            res = geodesic_distance(self.to_transform(coords), trans)
            return ChainSolveResult("satisfied", coords, float(res), 0, 0)

        # Exact path: a single component reduces to the closed-form TSR check.
        if n == 1:
            tsr = self.TSRs[0]
            if tsr.contains(trans):
                coords = numpy.array([tsr.to_xyzrpy(trans)])
                res = geodesic_distance(self.to_transform(coords), trans)
                return ChainSolveResult("satisfied", coords, float(res), 0, 0)
            dist, bw = tsr.distance(trans)
            return ChainSolveResult("not_found", numpy.array([bw]), float(dist), 0, 0)

        # General case: bounded multi-start over a SMOOTH squared pose residual.
        # _Bw_cont guarantees lower <= upper even for wrapping rotation intervals;
        # point-bound coordinates are held fixed (never handed to the finite-
        # difference optimiser, whose zero-width step there yields a NaN gradient).
        lower = numpy.concatenate([self.TSRs[i]._Bw_cont[:, 0] for i in range(n)])
        upper = numpy.concatenate([self.TSRs[i]._Bw_cont[:, 1] for i in range(n)])
        x_full = (lower + upper) / 2.0
        free = upper > lower
        R_target, t_target = trans[:3, :3], trans[:3, 3]

        def expand(x_free):
            x = x_full.copy()
            x[free] = x_free
            return x

        def geodesic_at(x_free):
            return geodesic_distance(self.to_transform(expand(x_free).reshape(n, 6)), trans)

        if not numpy.any(free):
            res = geodesic_at(numpy.empty(0))
            status = "satisfied" if res < tolerance else "not_found"
            return ChainSolveResult(status, x_full.reshape(n, 6), float(res), 0, 0)

        # The geodesic's arccos rotation term is non-smooth at 0 and breeds local
        # minima; the chordal term (3 - tr(Rᵀ R')) is smooth and zero iff the
        # rotations match, so its global minimum coincides with membership.
        def residual_sq(x_free):
            T = self.to_transform(expand(x_free).reshape(n, 6))
            dt = T[:3, 3] - t_target
            return float(dt @ dt + (3.0 - numpy.trace(R_target.T @ T[:3, :3])))

        lo_f, hi_f = lower[free], upper[free]
        span_f = hi_f - lo_f
        bounds = list(zip(lo_f, hi_f))

        # Deterministic bounded restarts: an invalid initial guess still warm-starts
        # first, then midpoint, corners, and a fixed low-discrepancy set.
        gen = numpy.random.default_rng(0)
        starts = [x_full[free], lo_f, hi_f]
        starts += [lo_f + span_f * gen.random(int(free.sum())) for _ in range(max_restarts)]
        if initial_guess is not None:
            ig = numpy.asarray(initial_guess, dtype=float)
            if ig.shape == (n, 6):
                starts.insert(0, ig.reshape(-1)[free])

        best_x = x_full[free]
        best_res = residual_sq(best_x)
        nfev = 0
        restarts = 0
        for x0 in starts:
            restarts += 1
            xopt, res, info = scipy.optimize.fmin_l_bfgs_b(
                residual_sq, x0, fprime=None, args=(), bounds=bounds, approx_grad=True, maxfun=max_nfev
            )
            nfev += int(info.get("funcalls", 0))
            if res < best_res:
                best_res, best_x = res, xopt
            if geodesic_at(best_x) < tolerance:
                break

        geo = geodesic_at(best_x)
        status = "satisfied" if geo < tolerance else "not_found"
        return ChainSolveResult(status, expand(best_x).reshape(n, 6), float(geo), nfev, restarts)

    def distance(self, trans):
        """
        Best-found geodesic residual from the chain to a transform, and the
        recovered coordinates.

        Delegates to :meth:`solve`. If a witness within tolerance is found the
        residual is ~0; otherwise it is the **best found** residual -- an upper
        bound on the true minimum, not a certified distance (#85). Callers needing
        an exact positive certificate should use :meth:`validate_witness`.

        @param trans 4x4 transform
        @return dist  best-found geodesic residual
        @return bwopt recovered coordinates as an (n, 6) array of xyzrpy
        """
        result = self.solve(trans)
        return result.residual, result.coordinates

    def closest_transform(self, trans):
        """
        Best-found residual and the closest composed world-frame transform.

        Mirrors :meth:`TSR.closest_transform` for chains (#63); inherits
        :meth:`distance`'s best-found (upper-bound) approximation status (#85).
        """
        dist, bwopt = self.distance(trans)
        return dist, self.to_transform(bwopt)

    def contains(self, trans, initial_guess=None):
        """
        Whether the chain contains the transform.

        A chain is the set of poses reachable by **serially composing** a
        transform sampled from each component TSR -- not the Boolean intersection
        of the components' world-frame pose sets. A single-TSR chain uses the
        exact closed-form check. For multi-TSR chains this delegates to
        :meth:`solve`: ``True`` means a witness was found within the numerical
        budget; ``False`` means none was found and is **not** a certificate of
        non-membership (#85). Pass ``initial_guess`` (e.g. a retained witness) for
        the exact fast path.

        @param  trans 4x4 transform
        @param  initial_guess (optional) (n, 6) coordinates to check first
        @return       True if a witness is found, False otherwise
        """
        if len(self.TSRs) == 0:
            return False
        if len(self.TSRs) == 1:
            return self.TSRs[0].contains(trans)
        return self.solve(trans, initial_guess=initial_guess).status == "satisfied"

    def to_xyzrpy(self, trans):
        """
        Converts an end-effector transform to a list of xyzrpy values
        @param  trans  4x4 transform
        @return xyzrpy_list list of xyzrpy values
        """
        _, xyzrpy_array = self.distance(trans)
        # Convert numpy array to list of arrays
        return [xyzrpy_array[i] for i in range(len(self.TSRs))]
