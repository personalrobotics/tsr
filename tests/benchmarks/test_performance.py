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


if __name__ == "__main__":
    # Run benchmarks
    benchmark = PerformanceBenchmark()
    benchmark.setUp()
    benchmark.run_all_benchmarks()
