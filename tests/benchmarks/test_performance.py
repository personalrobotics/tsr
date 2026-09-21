#!/usr/bin/env python
# SPDX-License-Identifier: BSD-2-Clause
# Authors: Siddhartha Srinivasa and contributors to TSR

"""
Performance benchmarks for TSR implementations.

These benchmarks ensure that the refactored implementation
doesn't introduce performance regressions.
"""

import time
import unittest

import numpy as np
from numpy import pi

from tests.tsr.hands._grasp_oracle import Cylinder, Sphere, Torus, certify, pose_from
from tsr.tsr import TSR
from tsr.tsr_chain import TSRChain


class PerformanceBenchmark(unittest.TestCase):
    """Performance benchmarks for TSR implementations."""

    def setUp(self):
        """Set up test fixtures."""
        # Common test parameters
        self.T0_w = np.array([[1, 0, 0, 0.1], [0, 1, 0, 0.2], [0, 0, 1, 0.3], [0, 0, 0, 1]])

        self.Tw_e = np.array([[0, 0, 1, 0.05], [1, 0, 0, 0], [0, 1, 0, 0.1], [0, 0, 0, 1]])

        self.Bw = np.array(
            [
                [-0.01, 0.01],  # x bounds
                [-0.01, 0.01],  # y bounds
                [-0.01, 0.01],  # z bounds
                [-pi / 4, pi / 4],  # roll bounds
                [-pi / 4, pi / 4],  # pitch bounds
                [-pi / 2, pi / 2],  # yaw bounds
            ]
        )

        # Create TSR instance
        self.tsr = TSR(T0_w=self.T0_w, Tw_e=self.Tw_e, Bw=self.Bw)

    def test_benchmark_tsr_creation(self):
        """Benchmark TSR creation performance."""
        num_iterations = 1000

        # Benchmark core creation
        start_time = time.time()
        for _ in range(num_iterations):
            TSR(T0_w=self.T0_w, Tw_e=self.Tw_e, Bw=self.Bw)
        creation_time = time.time() - start_time

        print("TSR Creation Benchmark:")
        print(f"  Core:   {creation_time:.4f}s ({num_iterations} iterations)")

        # Should be reasonably fast (less than 1 second for 1000 iterations)
        self.assertLess(creation_time, 1.0, "TSR creation is too slow")

    def test_benchmark_sampling(self):
        """Benchmark sampling performance."""
        num_samples = 10000

        # Benchmark core sampling
        start_time = time.time()
        for _ in range(num_samples):
            self.tsr.sample_xyzrpy()
        sampling_time = time.time() - start_time

        print("Sampling Benchmark:")
        print(f"  Core:   {sampling_time:.4f}s ({num_samples} samples)")

        # Should be reasonably fast (less than 5 seconds for 10000 samples)
        self.assertLess(sampling_time, 5.0, "TSR sampling is too slow")

    def test_benchmark_transform_calculation(self):
        """Benchmark transform calculation performance."""
        num_calculations = 10000
        # Use valid xyzrpy values that are within the TSR bounds
        test_inputs = [
            np.zeros(6),  # Center of bounds
            np.array([0.005, 0.005, 0.005, np.pi / 8, np.pi / 8, np.pi / 4]),  # Within bounds
            np.array([-0.005, -0.005, -0.005, -np.pi / 8, -np.pi / 8, -np.pi / 4]),  # Within bounds
        ]

        # Benchmark core transform calculation
        start_time = time.time()
        for _ in range(num_calculations):
            for xyzrpy in test_inputs:
                self.tsr.to_transform(xyzrpy)
        transform_time = time.time() - start_time

        print("Transform Calculation Benchmark:")
        print(f"  Core:   {transform_time:.4f}s ({num_calculations * len(test_inputs)} calculations)")

        # Should be reasonably fast (less than 5 seconds for 30000 calculations)
        self.assertLess(transform_time, 5.0, "TSR transform calculation is too slow")

    def test_benchmark_distance_calculation(self):
        """Benchmark distance calculation performance."""
        num_calculations = 100  # Reduced for faster testing
        test_transforms = [
            np.eye(4),
            self.T0_w,
            self.Tw_e,
            np.array([[1, 0, 0, 0.5], [0, 1, 0, 0.5], [0, 0, 1, 0.5], [0, 0, 0, 1]]),
        ]

        # Benchmark core distance calculation
        start_time = time.time()
        for _ in range(num_calculations):
            for transform in test_transforms:
                self.tsr.distance(transform)
        distance_time = time.time() - start_time

        print("Distance Calculation Benchmark:")
        print(f"  Core:   {distance_time:.4f}s ({num_calculations * len(test_transforms)} calculations)")

        # Should be reasonably fast (less than 10 seconds for 400 calculations)
        self.assertLess(distance_time, 10.0, "TSR distance calculation is too slow")

    def test_benchmark_containment_test(self):
        """Benchmark containment test performance."""
        num_tests = 10000
        test_transforms = [
            np.eye(4),  # Should be contained
            np.array(
                [
                    [1, 0, 0, 10.0],  # Should not be contained
                    [0, 1, 0, 10.0],
                    [0, 0, 1, 10.0],
                    [0, 0, 0, 1],
                ]
            ),
        ]

        # Benchmark core containment test
        start_time = time.time()
        for _ in range(num_tests):
            for transform in test_transforms:
                self.tsr.contains(transform)
        containment_time = time.time() - start_time

        print("Containment Test Benchmark:")
        print(f"  Core:   {containment_time:.4f}s ({num_tests * len(test_transforms)} tests)")

        # Should be reasonably fast (less than 5 seconds for 20000 tests)
        self.assertLess(containment_time, 5.0, "TSR containment test is too slow")

    def run_all_benchmarks(self):
        """Run all benchmarks and print summary."""
        print("=" * 50)
        print("TSR Performance Benchmarks")
        print("=" * 50)

        self.benchmark_tsr_creation()
        print()

        self.benchmark_sampling()
        print()

        self.benchmark_transform_calculation()
        print()

        self.benchmark_distance_calculation()
        print()

        self.benchmark_containment_test()
        print()

        print("=" * 50)


class ChainSolvePerformanceBenchmark(unittest.TestCase):
    """Chain inverse-membership benchmarks for the four #85 paths.

    The relevant planning metric is the warm-started solve (a nearby pose whose
    parent coordinates are retained), not whether an expensive cold global search
    can eventually recover every witness. We report median and tail latency for
    forward sampling, witness validation, warm solve, and cold solve.
    """

    def setUp(self):
        # A rotation-rich two-TSR chain with nonidentity frames and mixed
        # fixed/free coordinates -- the case where cold solves are expensive.
        rng = np.random.default_rng(20260920)
        parts = []
        for j in range(2):
            lo, hi, fr = np.zeros(6), np.zeros(6), rng.random(6) < 0.5
            c, h = rng.uniform(-0.4, 0.4, 6), rng.uniform(0.05, 0.4, 6)
            lo[fr], hi[fr] = c[fr] - h[fr], c[fr] + h[fr]
            T0_w = TSR.xyzrpy_to_trans(rng.uniform(-0.3, 0.3, 6)) if j == 0 else np.eye(4)
            Tw_e = TSR.xyzrpy_to_trans(rng.uniform(-0.15, 0.15, 6))
            parts.append(TSR(T0_w=T0_w, Tw_e=Tw_e, Bw=np.column_stack((lo, hi))))
        self.chain = TSRChain(TSRs=parts)
        self.rng = rng

    @staticmethod
    def _report(label, times):
        ms = np.array(times) * 1e3
        print(f"{label}: median {np.median(ms):.3f} ms, p95 {np.percentile(ms, 95):.3f} ms (n={len(ms)})")
        return np.median(ms)

    def test_benchmark_cold_process_first_warm_solve(self):
        """First warm solve in a FRESH process (#88).

        Steady-state warm solves are sub-millisecond, but the first solve in a new
        process must not silently pay the SciPy import: the valid-witness fast path
        returns before `import scipy.optimize`. Reported separately from
        steady-state median/p95 so the one-time cost is not hidden.
        """
        import subprocess
        import sys

        script = "\n".join(
            [
                "import time, numpy as np",
                "from tsr import TSR, TSRChain",
                "seg = TSR(Bw=np.array([[0.,1.],[0,0],[0,0],[0,0],[0,0],[0,0]]))",
                "chain = TSRChain(TSRs=[seg, seg])",
                "s = chain.sample_with_witness(rng=np.random.default_rng(0))",
                "t0 = time.perf_counter()",
                "r = chain.solve(s.pose, initial_guess=s.coordinates)",
                "dt = (time.perf_counter() - t0) * 1e3",
                "import sys as _sys",
                "print(f'{dt:.3f} {\"scipy.optimize\" in _sys.modules}')",
            ]
        )
        out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        dt_ms, scipy_loaded = out.stdout.split()
        print(f"cold-process first warm solve: {float(dt_ms):.3f} ms, scipy.optimize loaded: {scipy_loaded}")
        self.assertEqual(scipy_loaded, "False", "warm fast path imported scipy.optimize")

    def test_benchmark_chain_paths(self):
        n = 200
        chain = self.chain

        # 1. Forward sampling with witness (no optimizer).
        samples = []
        t = []
        for _ in range(n):
            s = time.perf_counter()
            samples.append(chain.sample_with_witness(rng=self.rng))
            t.append(time.perf_counter() - s)
        self._report("sample_with_witness", t)

        # 2. Witness validation (one composition, no SciPy).
        t = []
        for smp in samples:
            s = time.perf_counter()
            chain.validate_witness(smp.pose, smp.coordinates)
            t.append(time.perf_counter() - s)
        validate_median = self._report("validate_witness", t)

        # 3. Warm-started solve from the retained witness (fast path).
        t = []
        for smp in samples:
            s = time.perf_counter()
            chain.solve(smp.pose, initial_guess=smp.coordinates)
            t.append(time.perf_counter() - s)
        self._report("solve (warm)", t)

        # 4. Cold solve at the default budget (no initial guess).
        t = []
        for smp in samples[:40]:  # cold is heavy; fewer iterations
            s = time.perf_counter()
            chain.solve(smp.pose)
            t.append(time.perf_counter() - s)
        self._report("solve (cold)", t)

        # Sanity bounds only: witness validation is cheap; nothing is pathological.
        self.assertLess(validate_median, 5.0, "witness validation is too slow")


class GraspOracleBenchmark(unittest.TestCase):
    """Non-gating benchmark for the analytic grasp oracle (#93).

    Reports per-certification median/p95 and a projected #73 template/Hypothesis
    workload. No wall-clock assertions -- correctness lives in test_grasp_oracle.
    """

    def test_benchmark_certify(self):
        cases = [
            (
                Sphere(0.03),
                pose_from([0.06, 0, 0], [-1, 0, 0], [0, 1, 0]),
                dict(preshape=0.066, max_aperture=0.14, mode="surface"),
            ),
            (
                Cylinder(0.03, 0.12),
                pose_from([0.06, 0, 0.06], [-1, 0, 0], [0, 1, 0]),
                dict(preshape=0.066, max_aperture=0.14, mode="side"),
            ),
            (
                Torus(0.06, 0.02),
                pose_from([0, 0, 0.05], [0, 0, -1], [0, 1, 0]),
                dict(preshape=0.164, max_aperture=0.30, mode="span"),
            ),
        ]
        for prim, pose, kw in cases:  # warm up
            certify(prim, pose, finger_length=0.08, clearance=0.006, **kw)

        times = []
        for _ in range(2000):
            prim, pose, kw = cases[len(times) % len(cases)]
            s = time.perf_counter()
            certify(prim, pose, finger_length=0.08, clearance=0.006, **kw)
            times.append(time.perf_counter() - s)
        ms = np.array(times) * 1e3
        median, p95 = float(np.median(ms)), float(np.percentile(ms, 95))
        print(f"grasp oracle certify: median {median:.4f} ms, p95 {p95:.4f} ms (n={len(ms)})")

        # Projected #73 workload: ~4 primitives x ~9 modes x (extrema+mid+interior,
        # say 5 evals) x ~40 Hypothesis examples ~= 7200 certifications.
        projected = 4 * 9 * 5 * 40
        print(f"projected #73 (~{projected} certifications): ~{projected * median / 1e3:.2f} s median-rate")
        self.assertLess(median, 20.0, "oracle certification unexpectedly slow (guard only)")


if __name__ == "__main__":
    # Run benchmarks
    benchmark = PerformanceBenchmark()
    benchmark.setUp()
    benchmark.run_all_benchmarks()
