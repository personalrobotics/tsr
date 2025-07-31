# TSR Library Testing Strategy

This directory contains comprehensive tests to ensure the TSR library refactoring maintains functionality and performance.

## Test Structure

```
tests/
├── README.md                    # This file
├── run_tests.py                 # Main test runner
├── tsr/
│   ├── test_tsr.py             # Original TSR tests
│   ├── test_equivalence.py     # Equivalence tests (old vs new)
│   └── test_wrappers/
│       └── test_openrave_wrapper.py  # OpenRAVE wrapper tests
├── fixtures/
│   └── mock_robot.py           # Mock robot for testing
└── benchmarks/
    └── test_performance.py     # Performance benchmarks
```

## Test Categories

### 1. Equivalence Tests (`test_equivalence.py`)
**Purpose**: Ensure the new TSR implementation produces exactly the same results as the old one.

**What it tests**:
- TSR creation and properties
- Sampling behavior (with same random seeds)
- Transform calculations
- Distance and containment tests
- Edge cases and validation

**Key principle**: Same inputs → Same outputs

### 2. Unit Tests
**Purpose**: Test individual components in isolation.

**What it tests**:
- Core TSR functionality
- Utility functions
- Wrapper implementations
- Error handling

### 3. Wrapper Tests (`test_wrappers/`)
**Purpose**: Test simulator-specific wrapper implementations.

**What it tests**:
- OpenRAVE adapter functionality
- Robot interface compatibility
- Object type detection
- Legacy compatibility

### 4. Performance Benchmarks (`benchmarks/`)
**Purpose**: Ensure no performance regression.

**What it tests**:
- TSR creation speed
- Sampling performance
- Transform calculation speed
- Distance calculation speed
- Containment test speed

**Acceptance criteria**: New implementation should not be more than 20% slower than old.

### 5. Regression Tests
**Purpose**: Ensure existing functionality still works.

**What it tests**:
- Original test cases
- Known use cases
- Backward compatibility

## Mock Robot Interface

The `fixtures/mock_robot.py` provides a mock implementation that mimics OpenRAVE behavior without requiring the actual simulator. This allows testing of:

- Robot manipulator management
- Object transforms
- Grasp scenarios
- End-effector positioning

## Running Tests

### Run All Tests
```bash
cd tests
python run_tests.py
```

### Run Specific Test Categories
```bash
# Unit tests only
python run_tests.py --unit

# Equivalence tests only
python run_tests.py --equivalence

# Wrapper tests only
python run_tests.py --wrapper

# Performance benchmarks only
python run_tests.py --performance

# Regression tests only
python run_tests.py --regression
```

### Run Individual Test Files
```bash
# Run equivalence tests
python -m unittest tests.tsr.test_equivalence

# Run performance benchmarks
python -m unittest tests.benchmarks.test_performance

# Run wrapper tests
python -m unittest tests.tsr.test_wrappers.test_openrave_wrapper
```

## Test Output

The test runner provides detailed output including:

1. **Test Results**: Pass/fail status for each test
2. **Performance Metrics**: Timing comparisons between old and new implementations
3. **Summary Report**: Overall test status and execution time
4. **Error Details**: Specific failure information for debugging

## Continuous Integration

These tests should be run:

1. **Before each commit**: Ensure no regressions
2. **After refactoring**: Validate equivalence
3. **Before releases**: Comprehensive validation
4. **In CI/CD pipeline**: Automated testing

## Adding New Tests

### For New Features
1. Add unit tests in `tests/tsr/`
2. Add equivalence tests if applicable
3. Add performance benchmarks if performance-critical
4. Update this documentation

### For New Wrappers (e.g., MuJoCo)
1. Create `tests/tsr/test_wrappers/test_mujoco_wrapper.py`
2. Add mock MuJoCo robot in `tests/fixtures/`
3. Update test runner to include new wrapper tests
4. Add MuJoCo-specific test cases

## Debugging Test Failures

### Equivalence Test Failures
1. Check if the failure is due to numerical precision differences
2. Verify that the same random seeds are being used
3. Ensure both implementations handle edge cases identically
4. Check for differences in floating-point arithmetic

### Performance Test Failures
1. Run benchmarks multiple times to account for system variance
2. Check if the performance regression is acceptable
3. Profile the code to identify bottlenecks
4. Consider if the performance trade-off is worth the benefits

### Wrapper Test Failures
1. Verify that the mock robot interface matches the real simulator
2. Check that the wrapper correctly implements the abstract interface
3. Ensure backward compatibility is maintained
4. Test with actual simulator if available

## Best Practices

1. **Reproducible Tests**: Use fixed random seeds for deterministic results
2. **Comprehensive Coverage**: Test edge cases and error conditions
3. **Performance Monitoring**: Track performance over time
4. **Documentation**: Keep tests well-documented and maintainable
5. **Isolation**: Tests should not depend on each other
6. **Mocking**: Use mocks to avoid simulator dependencies

## Future Enhancements

1. **MuJoCo Wrapper Tests**: Add when MuJoCo wrapper is implemented
2. **Integration Tests**: Test with actual simulators
3. **Memory Benchmarks**: Track memory usage
4. **Coverage Reports**: Ensure comprehensive code coverage
5. **Automated Testing**: Set up CI/CD pipeline 