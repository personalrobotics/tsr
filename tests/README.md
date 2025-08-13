# TSR Library Tests

This directory contains comprehensive tests for the TSR library, ensuring the core functionality works correctly and the refactoring maintains compatibility.

## Test Structure

```
tests/
├── __init__.py
├── run_tests.py              # Main test runner
├── README.md                 # This file
├── benchmarks/
│   ├── __init__.py
│   └── test_performance.py   # Performance benchmarks
└── tsr/
    ├── __init__.py
    ├── test_tsr.py           # Core TSR tests
    ├── test_tsr_chain.py     # TSR chain tests
    ├── test_serialization.py # Serialization tests
    └── test_utils.py         # Utility function tests
```

## Test Categories

### 1. Unit Tests (`test_tsr/`)
- Core TSR functionality
- TSR chain operations
- Serialization/deserialization
- Utility functions

### 2. Core Functionality Tests
- TSR creation and validation
- Sampling operations
- Distance calculations
- TSR chain composition

### 3. New Architecture Tests
- TSRTemplate functionality
- TSRLibraryRelational operations
- Schema validation
- Advanced sampling utilities

## Running Tests

### All Tests
```bash
python -m pytest tests/
```

### Specific Test Categories
```bash
# Core TSR tests
python -m pytest tests/tsr/test_tsr.py

# TSR chain tests
python -m pytest tests/tsr/test_tsr_chain.py

# Serialization tests
python -m pytest tests/tsr/test_serialization.py

# Performance benchmarks
python -m pytest tests/benchmarks/test_performance.py
```

### Using the Test Runner
```bash
# Run all tests with the custom runner
python tests/run_tests.py

# Run specific test categories
python -m pytest tests/tsr/test_tsr.py::TsrTest::test_tsr_creation
```

## Test Coverage

The test suite covers:

### Core TSR Functionality
- TSR creation with various parameters
- Sampling from TSRs
- Distance calculations
- Constraint checking
- Geometric operations

### TSR Chains
- Chain creation and composition
- Multiple TSR handling
- Start/goal/constraint flags
- Chain sampling

### New Architecture Components
- TSRTemplate creation and instantiation
- TSRLibraryRelational registration and querying
- Schema validation (TaskCategory, TaskType, EntityClass)
- Advanced sampling utilities

### Serialization
- JSON serialization/deserialization
- YAML serialization/deserialization
- Dictionary conversion
- Error handling

### Performance
- Sampling performance benchmarks
- Large TSR set handling
- Memory usage optimization

## Adding New Tests

### For New Core Features
1. Create test file in `tests/tsr/`
2. Follow naming convention: `test_<feature>.py`
3. Add comprehensive test cases
4. Update this README

### For New Architecture Components
1. Create appropriate test file
2. Test both success and failure cases
3. Include edge cases and error conditions
4. Add performance tests if applicable

### Test Guidelines
- Use descriptive test names
- Test both valid and invalid inputs
- Include edge cases
- Test error conditions
- Keep tests independent and isolated
- Use pure geometric operations (no simulator dependencies)

## Continuous Integration

The test suite is designed to run in CI environments:
- No external simulator dependencies
- Fast execution
- Comprehensive coverage
- Clear error reporting

## Future Enhancements

1. **Integration Tests**: Add tests for integration with specific robotics frameworks
2. **Property-Based Testing**: Add property-based tests using hypothesis
3. **Performance Regression Tests**: Automated performance regression detection
4. **Documentation Tests**: Ensure code examples in docs are tested 