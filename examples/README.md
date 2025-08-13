# TSR Library Examples

This directory contains comprehensive examples demonstrating the TSR library functionality. The examples are organized into focused, individual files for better understanding and navigation.

## Example Files

### Core Examples

- **`01_basic_tsr.py`** - Core TSR creation and usage
  - Creating TSRs for grasping objects
  - Sampling poses from TSRs
  - Checking pose validity
  - Computing distances to TSRs

- **`02_tsr_chains.py`** - Complex constraints with TSR chains
  - Composing multiple TSRs for complex tasks
  - Example: Opening a refrigerator door
  - Sampling from TSR chains

- **`03_tsr_templates.py`** - Reusable, scene-agnostic TSR definitions
  - Creating templates for different object types
  - Instantiating templates at specific poses
  - Reusing templates across different scenes

### Advanced Examples

- **`04_relational_library.py`** - Task-based TSR generation and querying
  - Registering TSR generators for entity/task combinations
  - Querying available TSRs for given scenarios
  - Discovering available tasks for entities

- **`05_sampling.py`** - Advanced sampling from multiple TSRs
  - Computing weights based on TSR volumes
  - Weighted random sampling
  - Sampling from TSR templates

- **`06_serialization.py`** - TSR persistence and data exchange
  - Dictionary, JSON, and YAML serialization
  - TSR chain serialization
  - Cross-format roundtrip testing

## Running Examples

### Run All Examples
```bash
# From the examples directory
python run_all_examples.py

# Or from the project root
python examples/run_all_examples.py
```

### Run Individual Examples
```bash
# Run specific examples
python 01_basic_tsr.py
python 02_tsr_chains.py
python 03_tsr_templates.py
python 04_relational_library.py
python 05_sampling.py
python 06_serialization.py
```

### Legacy Support
The original `comprehensive_examples.py` file still works and runs all examples via the master runner.

## Example Output

Each example demonstrates specific functionality:

- **Basic TSR**: Shows how to create and use fundamental TSR operations
- **TSR Chains**: Demonstrates complex constraint composition
- **TSR Templates**: Illustrates reusable, scene-agnostic TSR definitions
- **Relational Library**: Shows task-based TSR generation and discovery
- **Sampling**: Demonstrates advanced sampling techniques
- **Serialization**: Shows data persistence and exchange capabilities

## Learning Path

For new users, we recommend following this order:

1. **Start with `01_basic_tsr.py`** to understand core TSR concepts
2. **Move to `03_tsr_templates.py`** to learn about reusable definitions
3. **Try `04_relational_library.py`** for task-based approaches
4. **Explore `02_tsr_chains.py`** for complex constraints
5. **Learn `05_sampling.py`** for advanced sampling
6. **Finish with `06_serialization.py`** for data persistence

## Requirements

All examples require the TSR library to be installed:
```bash
uv pip install -e .
```

The examples use standard Python libraries:
- `numpy` - For numerical operations
- `tsr` - The TSR library itself
