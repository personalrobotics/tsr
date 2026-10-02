# SPDX-License-Identifier: BSD-2-Clause
# Authors: Siddhartha Srinivasa and contributors to TSR

"""The chain's cold inverse: projected Levenberg-Marquardt with an analytic Jacobian (#174).

Internal. :meth:`tsr.TSRChain.solve` is the public entry point; everything here is reached
only from its cold branch, after every exact path has been tried.

**Why least squares, and why it is not an approximation.** The objective is

    ||dt||^2 + (3 - tr(R_target^T R))

and since ``3 - tr(A^T B) = ||A - B||_F^2 / 2`` for ``A, B`` in SO(3), that is *exactly*
``||r||^2`` for the 12-vector ``r = [dt, (R - R_target) / sqrt(2)]``. So the problem is a
least-squares problem in disguise and Levenberg-Marquardt is its natural method. The scalar
minimised here is the same one the previous L-BFGS-B path minimised, to the last bit.

**Why analytic.** ``rpy_to_rot`` is ``Rz(y) Ry(p) Rx(r)``, so each rotation derivative is a
product of matrices already on hand, and one prefix/suffix sweep builds all ``6n`` columns in
O(n). The previous path finite-differenced, paying ``6n`` extra chain compositions per
gradient; measured over 560 cold solves on 70 random chains, this one needs 43.1 objective
evaluations against 360.2 and is 3.9x faster in wall-clock, at better recall.

**Why projected.** The bounds are a box, so a step is clamped back into it, and a coordinate
pinned at a bound whose gradient pushes further out is held fixed for that step. Without that
second part a chain carrying a full SO(3) ball -- where RPY runs through gimbal lock and the
Jacobian loses rank -- solves 17 of 40 poses instead of 40.

This mirrors ``cpp/src/chain_solver.hpp`` deliberately, structure for structure, so the two
can be read side by side. It does **not** make the cold inverse comparable between them: a
redundant chain's solution set is a continuum, so two solvers that both succeed agree on the
coordinates only about 1% of the time. See docs/CPP.md.
"""

from __future__ import annotations

import numpy

from .tsr import TSR

#: ``3 - tr(A^T B) = ||A - B||_F^2 / 2``, so the nine rotation rows carry this factor.
_ROOT_TWO = numpy.sqrt(2.0)

#: Skew generators of the three axes: ``d/dtheta Rx(theta) = Ex Rx(theta)``, and likewise.
_E = (
    numpy.array([[0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]),
    numpy.array([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]),
    numpy.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
)

_LAMBDA0, _LAMBDA_MIN, _LAMBDA_MAX = 1e-3, 1e-12, 1e14
_MAX_ITERATIONS = 100  # a hard cap, so the worst case is bounded and repeatable
_MAX_TRIALS = 20


def _rot(axis: int, angle: float) -> numpy.ndarray:
    c, s = numpy.cos(angle), numpy.sin(angle)
    if axis == 0:
        return numpy.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    if axis == 1:
        return numpy.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    return numpy.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def residual_and_jacobian(chain, coordinates, trans, jacobian: bool):
    """``(r, J)`` at ``coordinates``, with ``J`` ``None`` when ``jacobian`` is False.

    ``coordinates`` must already be in the chart; the caller canonicalises through
    ``TSRChain._to_continuous``, exactly as the forward path does, so the objective descended
    here is the one ``to_transform`` will evaluate at the end.
    """
    links = chain.TSRs
    n = len(links)
    Tw = [TSR.xyzrpy_to_trans(coordinates[i]) for i in range(n)]
    step = [Tw[i] @ links[i].Tw_e for i in range(n)]

    # Prefixes: only the FIRST link's T0_w participates, which is the composition rule.
    prefix = [None] * n
    acc = links[0].T0_w
    for i in range(n):
        prefix[i] = acc
        acc = acc @ step[i]
    T = acc

    r = numpy.empty(12)
    r[:3] = T[:3, 3] - trans[:3, 3]
    r[3:] = (T[:3, :3] - trans[:3, :3]).reshape(9) / _ROOT_TWO
    if not jacobian:
        return r, None

    suffix = [None] * n
    tail = numpy.eye(4)
    for i in range(n - 1, -1, -1):
        suffix[i] = links[i].Tw_e @ tail
        tail = step[i] @ tail

    J = numpy.zeros((12, 6 * n))
    for i in range(n):
        roll, pitch, yaw = coordinates[i][3], coordinates[i][4], coordinates[i][5]
        Rx, Ry, Rz = _rot(0, roll), _rot(1, pitch), _rot(2, yaw)
        rotations = (Rz @ Ry @ _E[0] @ Rx, Rz @ _E[1] @ Ry @ Rx, _E[2] @ Rz @ Ry @ Rx)
        for j in range(6):
            d = numpy.zeros((4, 4))
            if j < 3:
                d[j, 3] = 1.0
            else:
                d[:3, :3] = rotations[j - 3]
            dT = prefix[i] @ d @ suffix[i]
            column = i * 6 + j
            J[:3, column] = dT[:3, 3]
            J[3:, column] = dT[:3, :3].reshape(9) / _ROOT_TWO
    return r, J


def minimise(chain, trans, x_full, lower, upper, free, schedule, max_nfev, tolerance, geodesic_at):
    """Run the bounded multi-start search. Returns ``(coordinates_flat, nfev, starts)``.

    ``schedule`` is the caller's deterministic start list over the free coordinates, already
    truncated to ``max_starts``; an empty one means no optimiser runs at all, which is how
    ``max_starts == 0`` keeps reporting the un-optimised midpoint.
    """
    n = len(chain.TSRs)
    free_index = numpy.flatnonzero(free)
    m = free_index.size
    lo_f, hi_f = lower[free], upper[free]

    state = {"nfev": 0, "best_x": x_full[free].copy(), "best_sq": numpy.inf}

    def expand(x_free):
        x = x_full.copy()
        x[free] = x_free
        return chain._to_continuous(x.reshape(n, 6))

    def evaluate(x_free, want_jacobian):
        """One residual evaluation: one forward composition, the unit ``nfev`` counts.

        The budget is checked *before* the increment, so ``nfev <= max_nfev`` holds strictly.
        The Jacobian rides along on the same sweep and is not counted separately.
        """
        if state["nfev"] >= max_nfev:
            return None, None
        state["nfev"] += 1
        r, J = residual_and_jacobian(chain, expand(x_free), trans, want_jacobian)
        sq = float(r @ r)
        if sq < state["best_sq"]:
            state["best_sq"], state["best_x"] = sq, numpy.array(x_free, dtype=float)
        return r, J

    starts = 0
    for x0 in schedule:
        if state["nfev"] >= max_nfev:
            break  # before the increment, so a start that cannot afford an evaluation is not counted
        starts += 1

        x = numpy.clip(numpy.asarray(x0, dtype=float), lo_f, hi_f)
        r, J = evaluate(x, True)
        if r is None:
            break
        f0 = float(r @ r)
        lam = _LAMBDA0

        for _ in range(_MAX_ITERATIONS):
            if f0 <= 0.0:
                break
            Jf = J[:, free_index]
            g = Jf.T @ r
            H = Jf.T @ Jf

            # Gradient projection: a coordinate on a bound whose gradient pushes it further out
            # cannot move, so hold it fixed for this step rather than let a rank-deficient or
            # outward direction stall the whole solve.
            active = ((x <= lo_f) & (g > 0.0)) | ((x >= hi_f) & (g < 0.0))
            if active.all():
                break  # every direction points out of the box: this start is done

            diagonal = numpy.maximum(numpy.diag(H), 1e-10)
            stepped, step_inf = False, 0.0
            for _ in range(_MAX_TRIALS):
                if lam > _LAMBDA_MAX:
                    break
                system = H + lam * numpy.diag(diagonal)
                rhs = -g
                if active.any():
                    system = system.copy()
                    system[active, :] = 0.0
                    system[:, active] = 0.0
                    system[active, active] = 1.0
                    rhs = rhs.copy()
                    rhs[active] = 0.0
                try:
                    delta = numpy.linalg.solve(system, rhs)
                except numpy.linalg.LinAlgError:
                    lam *= 10.0  # singular: a larger lambda restores diagonal dominance
                    continue
                candidate = numpy.clip(x + delta, lo_f, hi_f)
                step_inf = float(numpy.max(numpy.abs(candidate - x))) if m else 0.0
                r_new, J_new = evaluate(candidate, True)
                if r_new is None:
                    break
                f1 = float(r_new @ r_new)
                if f1 < f0:
                    x, r, J, f0 = candidate, r_new, J_new, f1
                    lam = max(lam * 0.3, _LAMBDA_MIN)
                    stepped = True
                    break
                lam *= 10.0

            if not stepped or step_inf <= 1e-14 or state["nfev"] >= max_nfev:
                break

        if geodesic_at(state["best_x"]) < tolerance:
            break

    return state["best_x"], state["nfev"], starts
