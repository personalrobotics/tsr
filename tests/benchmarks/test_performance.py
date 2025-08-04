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

# Import core implementation for testing
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
        
        # Create TSR instance
        self.tsr = CoreTSR(T0_w=self.T0_w, Tw_e=self.Tw_e, Bw=self.Bw)
    
    def test_benchmark_tsr_creation(self):
        """Benchmark TSR creation performance."""
        num_iterations = 1000
        
        # Benchmark core creation
        start_time = time.time()
        for _ in range(num_iterations):
            CoreTSR(T0_w=self.T0_w, Tw_e=self.Tw_e, Bw=self.Bw)
        creation_time = time.time() - start_time
        
        print(f"TSR Creation Benchmark:")
        print(f"  Core:   {creation_time:.4f}s ({num_iterations} iterations)")
        
        # Should be reasonably fast (less than 1 second for 1000 iterations)
        self.assertLess(creation_time, 1.0, 
                       "TSR creation is too slow")
    
    def test_benchmark_sampling(self):
        """Benchmark sampling performance."""
        num_samples = 10000
        
        # Benchmark core sampling
        start_time = time.time()
        for _ in range(num_samples):
            self.tsr.sample_xyzrpy()
        sampling_time = time.time() - start_time
        
        print(f"Sampling Benchmark:")
        print(f"  Core:   {sampling_time:.4f}s ({num_samples} samples)")
        
        # Should be reasonably fast (less than 5 seconds for 10000 samples)
        self.assertLess(sampling_time, 5.0, 
                       "TSR sampling is too slow")
    
    def test_benchmark_transform_calculation(self):
        """Benchmark transform calculation performance."""
        num_calculations = 10000
        test_inputs = [
            np.zeros(6),
            np.array([0.1, 0.2, 0.3, pi/4, pi/6, pi/3]),
            np.array([-0.1, -0.2, -0.3, -pi/4, -pi/6, -pi/3])
        ]
        
        # Benchmark core transform calculation
        start_time = time.time()
        for _ in range(num_calculations):
            for xyzrpy in test_inputs:
                self.tsr.to_transform(xyzrpy)
        transform_time = time.time() - start_time
        
        print(f"Transform Calculation Benchmark:")
        print(f"  Core:   {transform_time:.4f}s ({num_calculations * len(test_inputs)} calculations)")
        
        # Should be reasonably fast (less than 5 seconds for 30000 calculations)
        self.assertLess(transform_time, 5.0, 
                       "TSR transform calculation is too slow")
    
    def test_benchmark_distance_calculation(self):
        """Benchmark distance calculation performance."""
        num_calculations = 100  # Reduced for faster testing
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
        
        # Benchmark core distance calculation
        start_time = time.time()
        for _ in range(num_calculations):
            for transform in test_transforms:
                self.tsr.distance(transform)
        distance_time = time.time() - start_time
        
        print(f"Distance Calculation Benchmark:")
        print(f"  Core:   {distance_time:.4f}s ({num_calculations * len(test_transforms)} calculations)")
        
        # Should be reasonably fast (less than 10 seconds for 400 calculations)
        self.assertLess(distance_time, 10.0, 
                       "TSR distance calculation is too slow")
    
    def test_benchmark_containment_test(self):
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
        
        # Benchmark core containment test
        start_time = time.time()
        for _ in range(num_tests):
            for transform in test_transforms:
                self.tsr.contains(transform)
        containment_time = time.time() - start_time
        
        print(f"Containment Test Benchmark:")
        print(f"  Core:   {containment_time:.4f}s ({num_tests * len(test_transforms)} tests)")
        
        # Should be reasonably fast (less than 5 seconds for 20000 tests)
        self.assertLess(containment_time, 5.0, 
                       "TSR containment test is too slow")
    
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