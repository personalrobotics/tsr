#!/usr/bin/env python
"""
Performance benchmarks for TSR implementations.

These benchmarks ensure that the refactored implementation
doesn't introduce performance regressions.
"""

import time
import numpy as np
import unittest
from numpy import pi

# Import both implementations for comparison
from tsr.tsr import TSR as LegacyTSR
from tsr.core.tsr import TSR as CoreTSR


class PerformanceBenchmark(unittest.TestCase):
    """Performance benchmarks for TSR implementations."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Common test parameters
        self.T0_w = np.array([
            [1, 0, 0, 0.1],
            [0, 1, 0, 0.2],
            [0, 0, 1, 0.3],
            [0, 0, 0, 1]
        ])
        
        self.Tw_e = np.array([
            [0, 0, 1, 0.05],
            [1, 0, 0, 0],
            [0, 1, 0, 0.1],
            [0, 0, 0, 1]
        ])
        
        self.Bw = np.array([
            [-0.01, 0.01],  # x bounds
            [-0.01, 0.01],  # y bounds
            [-0.01, 0.01],  # z bounds
            [-pi/4, pi/4],   # roll bounds
            [-pi/4, pi/4],   # pitch bounds
            [-pi/2, pi/2]    # yaw bounds
        ])
        
        # Create TSR instances
        self.legacy_tsr = LegacyTSR(T0_w=self.T0_w, Tw_e=self.Tw_e, Bw=self.Bw)
        self.core_tsr = CoreTSR(T0_w=self.T0_w, Tw_e=self.Tw_e, Bw=self.Bw)
    
    def benchmark_tsr_creation(self):
        """Benchmark TSR creation performance."""
        num_iterations = 1000
        
        # Benchmark legacy creation
        start_time = time.time()
        for _ in range(num_iterations):
            LegacyTSR(T0_w=self.T0_w, Tw_e=self.Tw_e, Bw=self.Bw)
        legacy_time = time.time() - start_time
        
        # Benchmark core creation
        start_time = time.time()
        for _ in range(num_iterations):
            CoreTSR(T0_w=self.T0_w, Tw_e=self.Tw_e, Bw=self.Bw)
        core_time = time.time() - start_time
        
        print(f"TSR Creation Benchmark:")
        print(f"  Legacy: {legacy_time:.4f}s ({num_iterations} iterations)")
        print(f"  Core:   {core_time:.4f}s ({num_iterations} iterations)")
        print(f"  Ratio:  {core_time/legacy_time:.2f}x")
        
        # Core should not be significantly slower (within 20%)
        self.assertLess(core_time, legacy_time * 1.2, 
                       "Core implementation is significantly slower than legacy")
    
    def benchmark_sampling(self):
        """Benchmark sampling performance."""
        num_samples = 10000
        
        # Benchmark legacy sampling
        start_time = time.time()
        for _ in range(num_samples):
            self.legacy_tsr.sample_xyzrpy()
        legacy_time = time.time() - start_time
        
        # Benchmark core sampling
        start_time = time.time()
        for _ in range(num_samples):
            self.core_tsr.sample_xyzrpy()
        core_time = time.time() - start_time
        
        print(f"Sampling Benchmark:")
        print(f"  Legacy: {legacy_time:.4f}s ({num_samples} samples)")
        print(f"  Core:   {core_time:.4f}s ({num_samples} samples)")
        print(f"  Ratio:  {core_time/legacy_time:.2f}x")
        
        # Core should not be significantly slower (within 20%)
        self.assertLess(core_time, legacy_time * 1.2, 
                       "Core implementation is significantly slower than legacy")
    
    def benchmark_transform_calculation(self):
        """Benchmark transform calculation performance."""
        num_calculations = 10000
        test_inputs = [
            np.zeros(6),
            np.array([0.1, 0.2, 0.3, pi/4, pi/6, pi/3]),
            np.array([-0.1, -0.2, -0.3, -pi/4, -pi/6, -pi/3])
        ]
        
        # Benchmark legacy transform calculation
        start_time = time.time()
        for _ in range(num_calculations):
            for xyzrpy in test_inputs:
                self.legacy_tsr.to_transform(xyzrpy)
        legacy_time = time.time() - start_time
        
        # Benchmark core transform calculation
        start_time = time.time()
        for _ in range(num_calculations):
            for xyzrpy in test_inputs:
                self.core_tsr.to_transform(xyzrpy)
        core_time = time.time() - start_time
        
        print(f"Transform Calculation Benchmark:")
        print(f"  Legacy: {legacy_time:.4f}s ({num_calculations * len(test_inputs)} calculations)")
        print(f"  Core:   {core_time:.4f}s ({num_calculations * len(test_inputs)} calculations)")
        print(f"  Ratio:  {core_time/legacy_time:.2f}x")
        
        # Core should not be significantly slower (within 20%)
        self.assertLess(core_time, legacy_time * 1.2, 
                       "Core implementation is significantly slower than legacy")
    
    def benchmark_distance_calculation(self):
        """Benchmark distance calculation performance."""
        num_calculations = 10000
        test_transforms = [
            np.eye(4),
            self.T0_w,
            self.Tw_e,
            np.array([
                [1, 0, 0, 0.5],
                [0, 1, 0, 0.5],
                [0, 0, 1, 0.5],
                [0, 0, 0, 1]
            ])
        ]
        
        # Benchmark legacy distance calculation
        start_time = time.time()
        for _ in range(num_calculations):
            for transform in test_transforms:
                self.legacy_tsr.distance(transform)
        legacy_time = time.time() - start_time
        
        # Benchmark core distance calculation
        start_time = time.time()
        for _ in range(num_calculations):
            for transform in test_transforms:
                self.core_tsr.distance(transform)
        core_time = time.time() - start_time
        
        print(f"Distance Calculation Benchmark:")
        print(f"  Legacy: {legacy_time:.4f}s ({num_calculations * len(test_transforms)} calculations)")
        print(f"  Core:   {core_time:.4f}s ({num_calculations * len(test_transforms)} calculations)")
        print(f"  Ratio:  {core_time/legacy_time:.2f}x")
        
        # Core should not be significantly slower (within 20%)
        self.assertLess(core_time, legacy_time * 1.2, 
                       "Core implementation is significantly slower than legacy")
    
    def benchmark_containment_test(self):
        """Benchmark containment test performance."""
        num_tests = 10000
        test_transforms = [
            np.eye(4),  # Should be contained
            np.array([
                [1, 0, 0, 10.0],  # Should not be contained
                [0, 1, 0, 10.0],
                [0, 0, 1, 10.0],
                [0, 0, 0, 1]
            ])
        ]
        
        # Benchmark legacy containment test
        start_time = time.time()
        for _ in range(num_tests):
            for transform in test_transforms:
                self.legacy_tsr.contains(transform)
        legacy_time = time.time() - start_time
        
        # Benchmark core containment test
        start_time = time.time()
        for _ in range(num_tests):
            for transform in test_transforms:
                self.core_tsr.contains(transform)
        core_time = time.time() - start_time
        
        print(f"Containment Test Benchmark:")
        print(f"  Legacy: {legacy_time:.4f}s ({num_tests * len(test_transforms)} tests)")
        print(f"  Core:   {core_time:.4f}s ({num_tests * len(test_transforms)} tests)")
        print(f"  Ratio:  {core_time/legacy_time:.2f}x")
        
        # Core should not be significantly slower (within 20%)
        self.assertLess(core_time, legacy_time * 1.2, 
                       "Core implementation is significantly slower than legacy")
    
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


if __name__ == '__main__':
    # Run benchmarks
    benchmark = PerformanceBenchmark()
    benchmark.setUp()
    benchmark.run_all_benchmarks() 