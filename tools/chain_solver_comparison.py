# SPDX-License-Identifier: BSD-2-Clause
# Authors: Siddhartha Srinivasa and contributors to TSR

"""Is the chain's cold inverse no worse than the L-BFGS-B path it replaced (#174)?

The switch to projected Levenberg-Marquardt with an analytic Jacobian was justified on
recall and cost, so the comparison is checked in rather than left in a commit message.
``--against-scipy`` reconstructs the old path -- the same objective, the same deterministic
start schedule, the same budget -- and reports both.

Poses are drawn from each chain, so a witness provably exists and "recall" means something.
Equal budget throughout: the defaults ``TSRChain.solve`` itself uses.

    uv run python tools/chain_solver_comparison.py                  # the current solver
    uv run python tools/chain_solver_comparison.py --against-scipy  # and the old one

What this does NOT show, and cannot: that the two agree. A redundant chain's solution set is
a continuum, so two solvers that both succeed recover the same coordinates about 1% of the
time. Both answers are correct. That is why the cold inverse is specified by properties
rather than recorded in the conformance corpus, in either implementation -- see docs/CPP.md.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from tsr import TSR, TSRChain
from tsr.core.utils import EPSILON, geodesic_distance

SEED = 4242


def random_chains(rng: np.random.Generator, count: int) -> list[TSRChain]:
    """Chains spanning what makes the inverse hard: fixed rows, full turns, outer intervals."""

    def frame() -> np.ndarray:
        return TSR.xyzrpy_to_trans(np.concatenate([rng.uniform(-1, 1, 3), rng.uniform(-np.pi, np.pi, 3)]))

    def bounds() -> np.ndarray:
        b = np.zeros((6, 2))
        for i in range(3):
            if rng.random() < 0.5:
                width, centre = rng.uniform(0, 0.3), rng.uniform(-0.2, 0.2)
                b[i] = [centre - width, centre + width]
        for i in range(3, 6):
            draw = rng.random()
            if draw < 0.35:  # a point bound
                value = rng.uniform(-np.pi, np.pi)
                b[i] = [value, value]
            elif draw < 0.55:  # a full turn
                b[i] = [-np.pi, np.pi]
            elif draw < 0.70:  # an outer interval, wrapping through +-pi
                lower = rng.uniform(1.5, 3.0)
                b[i] = [lower, -lower]
            else:
                width, centre = rng.uniform(0, 1.2), rng.uniform(-1, 1)
                b[i] = [centre - width, centre + width]
        return b

    out = []
    while len(out) < count:
        # Only the first link may carry a frame; a later one is refused (#166).
        links = [TSR(frame(), frame(), bounds())]
        for _ in range(int(rng.integers(1, 3))):
            links.append(TSR(np.eye(4), frame(), bounds()))
        chain = TSRChain(TSRs=links)
        if any((t._Bw_cont[:, 1] > t._Bw_cont[:, 0]).any() for t in chain.TSRs):
            out.append(chain)  # an all-fixed chain never reaches the cold path
    return out


def solve_with_scipy(chain: TSRChain, trans: np.ndarray, max_starts: int, max_nfev: int, tolerance: float):
    """The pre-#174 cold path: L-BFGS-B over the same objective, schedule and budget."""
    import scipy.optimize

    n = len(chain.TSRs)
    lower = np.concatenate([t._Bw_cont[:, 0] for t in chain.TSRs])
    upper = np.concatenate([t._Bw_cont[:, 1] for t in chain.TSRs])
    x_full = (lower + upper) / 2.0
    free = upper > lower
    lo_f, hi_f = lower[free], upper[free]
    R_target, t_target = trans[:3, :3], trans[:3, 3]

    def expand(x_free):
        x = x_full.copy()
        x[free] = x_free
        return x

    def geodesic_at(x_free):
        return geodesic_distance(chain.to_transform(expand(x_free).reshape(n, 6)), trans)

    state = {"nfev": 0, "best_x": x_full[free], "best": np.inf}

    class Exhausted(Exception):
        pass

    def counted(x_free):
        if state["nfev"] >= max_nfev:
            raise Exhausted
        state["nfev"] += 1
        T = chain.to_transform(expand(x_free).reshape(n, 6))
        dt = T[:3, 3] - t_target
        value = float(dt @ dt + (3.0 - np.trace(R_target.T @ T[:3, :3])))
        if value < state["best"]:
            state["best"], state["best_x"] = value, np.array(x_free, dtype=float)
        return value

    schedule = [x_full[free], lo_f, hi_f]
    gen = np.random.default_rng(0)
    while len(schedule) < max_starts:
        schedule.append(lo_f + (hi_f - lo_f) * gen.random(int(free.sum())))
    schedule = schedule[:max_starts]

    starts = 0
    for x0 in schedule:
        if state["nfev"] >= max_nfev:
            break
        starts += 1
        try:
            scipy.optimize.fmin_l_bfgs_b(
                counted, x0, fprime=None, bounds=list(zip(lo_f, hi_f)), approx_grad=True,
                maxfun=max(1, max_nfev - state["nfev"]),
            )  # fmt: skip
        except Exhausted:
            pass
        if geodesic_at(state["best_x"]) < tolerance:
            break
    residual = geodesic_at(state["best_x"])
    return residual < tolerance, residual, state["nfev"], starts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--chains", type=int, default=70)
    parser.add_argument("--poses", type=int, default=8)
    parser.add_argument("--against-scipy", action="store_true", help="also run the pre-#174 path")
    args = parser.parse_args(argv)

    rng = np.random.default_rng(SEED)
    chains = random_chains(rng, args.chains)
    budget = dict(max_starts=TSRChain._DEFAULT_MAX_STARTS, max_nfev=TSRChain._DEFAULT_MAX_NFEV, tolerance=EPSILON)

    found = nfev = 0
    elapsed = 0.0
    old_found = old_nfev = 0
    old_elapsed = 0.0
    cases = 0
    for chain in chains:
        for _ in range(args.poses):
            target = chain.sample_with_witness(rng=rng).pose
            cases += 1

            t0 = time.perf_counter()
            result = chain.solve(target)
            elapsed += time.perf_counter() - t0
            found += result.status == "satisfied"
            nfev += result.nfev

            if args.against_scipy:
                t0 = time.perf_counter()
                ok, _, old_evals, _ = solve_with_scipy(chain, target, **budget)
                old_elapsed += time.perf_counter() - t0
                old_found += ok
                old_nfev += old_evals

    print(f"{cases} cold solves over {len(chains)} random chains (n=2..3), equal budget {budget}\n")

    def line(label, hits, evals, seconds):
        recall, per_call, millis = 100 * hits / cases, evals / cases, 1e3 * seconds / cases
        return f"  {label:<14} recall {recall:5.1f}%   {per_call:7.1f} evals   {millis:7.2f} ms"

    print(line("projected LM", found, nfev, elapsed))
    if args.against_scipy:
        print(line("L-BFGS-B", old_found, old_nfev, old_elapsed))
        print(f"\n  {old_nfev / max(nfev, 1):.1f}x fewer evaluations, {old_elapsed / max(elapsed, 1e-9):.1f}x faster")
        if found < old_found:
            print(f"\nREGRESSION: recall fell from {old_found}/{cases} to {found}/{cases}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
