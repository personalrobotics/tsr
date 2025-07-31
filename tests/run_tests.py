#!/usr/bin/env python
"""
Comprehensive test runner for TSR library refactoring.

This script runs all tests to ensure the refactored implementation
is equivalent to the original and maintains performance.
"""

import sys
import os
import unittest
import time
import argparse
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def run_unit_tests():
    """Run all unit tests."""
    print("=" * 60)
    print("Running Unit Tests")
    print("=" * 60)
    
    # Discover and run all unit tests
    loader = unittest.TestLoader()
    start_dir = os.path.join(os.path.dirname(__file__), 'tsr')
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


def run_equivalence_tests():
    """Run equivalence tests between old and new implementations."""
    print("\n" + "=" * 60)
    print("Running Equivalence Tests")
    print("=" * 60)
    
    # Import and run equivalence tests
    from .tsr.test_equivalence import TestTSEquivalence
    
    suite = unittest.TestLoader().loadTestsFromTestCase(TestTSEquivalence)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


def run_wrapper_tests():
    """Run wrapper-specific tests."""
    print("\n" + "=" * 60)
    print("Running Wrapper Tests")
    print("=" * 60)
    
    # Import and run wrapper tests
    from .tsr.test_wrappers.test_openrave_wrapper import (
        TestOpenRAVEWrapper, TestOpenRAVETSRFunctions, TestOpenRAVECompatibility
    )
    
    test_classes = [TestOpenRAVEWrapper, TestOpenRAVETSRFunctions, TestOpenRAVECompatibility]
    
    all_successful = True
    for test_class in test_classes:
        suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        all_successful = all_successful and result.wasSuccessful()
    
    return all_successful


def run_performance_benchmarks():
    """Run performance benchmarks."""
    print("\n" + "=" * 60)
    print("Running Performance Benchmarks")
    print("=" * 60)
    
    # Import and run performance benchmarks
    from .benchmarks.test_performance import PerformanceBenchmark
    
    suite = unittest.TestLoader().loadTestsFromTestCase(PerformanceBenchmark)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


def run_regression_tests():
    """Run regression tests for existing functionality."""
    print("\n" + "=" * 60)
    print("Running Regression Tests")
    print("=" * 60)
    
    # Import and run existing tests
    from .tsr.test_tsr import TsrTest
    
    suite = unittest.TestLoader().loadTestsFromTestCase(TsrTest)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


def run_all_tests():
    """Run all tests and provide a comprehensive report."""
    print("TSR Library Refactoring Test Suite")
    print("=" * 60)
    
    start_time = time.time()
    
    # Run all test categories
    test_results = {}
    
    try:
        test_results['unit'] = run_unit_tests()
    except Exception as e:
        print(f"Unit tests failed with error: {e}")
        test_results['unit'] = False
    
    try:
        test_results['equivalence'] = run_equivalence_tests()
    except Exception as e:
        print(f"Equivalence tests failed with error: {e}")
        test_results['equivalence'] = False
    
    try:
        test_results['wrapper'] = run_wrapper_tests()
    except Exception as e:
        print(f"Wrapper tests failed with error: {e}")
        test_results['wrapper'] = False
    
    try:
        test_results['performance'] = run_performance_benchmarks()
    except Exception as e:
        print(f"Performance benchmarks failed with error: {e}")
        test_results['performance'] = False
    
    try:
        test_results['regression'] = run_regression_tests()
    except Exception as e:
        print(f"Regression tests failed with error: {e}")
        test_results['regression'] = False
    
    end_time = time.time()
    
    # Print summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    all_passed = True
    for test_type, passed in test_results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"{test_type.upper():15} : {status}")
        if not passed:
            all_passed = False
    
    print(f"\nTotal Time: {end_time - start_time:.2f} seconds")
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED! 🎉")
        print("The refactored implementation is equivalent to the original.")
    else:
        print("\n❌ SOME TESTS FAILED! ❌")
        print("Please review the failures above.")
    
    return all_passed


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Run TSR library tests')
    parser.add_argument('--unit', action='store_true', help='Run only unit tests')
    parser.add_argument('--equivalence', action='store_true', help='Run only equivalence tests')
    parser.add_argument('--wrapper', action='store_true', help='Run only wrapper tests')
    parser.add_argument('--performance', action='store_true', help='Run only performance benchmarks')
    parser.add_argument('--regression', action='store_true', help='Run only regression tests')
    
    args = parser.parse_args()
    
    # If specific test type is requested, run only that
    if args.unit:
        success = run_unit_tests()
    elif args.equivalence:
        success = run_equivalence_tests()
    elif args.wrapper:
        success = run_wrapper_tests()
    elif args.performance:
        success = run_performance_benchmarks()
    elif args.regression:
        success = run_regression_tests()
    else:
        # Run all tests
        success = run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main() 