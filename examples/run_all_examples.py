#!/usr/bin/env python
"""
Master Example Runner: Execute all TSR library examples.

This script runs all the individual example files in sequence,
demonstrating the complete TSR library functionality.
"""

import subprocess
import sys
import os


def run_example(example_file):
    """Run a single example file and return success status."""
    print(f"\n{'='*60}")
    print(f"Running: {example_file}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(
            [sys.executable, example_file],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        
        if result.returncode == 0:
            print(result.stdout)
            return True
        else:
            print(f"Error running {example_file}:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"Exception running {example_file}: {e}")
        return False


def main():
    """Run all TSR library examples."""
    print("TSR Library - Complete Example Suite")
    print("=" * 60)
    
    # List of example files in order
    examples = [
        "01_basic_tsr.py",
        "02_tsr_chains.py", 
        "03_tsr_templates.py",
        "04_relational_library.py",
        "05_sampling.py",
        "06_serialization.py",
        "07_template_file_management.py",
        "08_template_generators.py"
    ]
    
    success_count = 0
    total_count = len(examples)
    
    for example in examples:
        if run_example(example):
            success_count += 1
    
    print(f"\n{'='*60}")
    print(f"Example Suite Complete: {success_count}/{total_count} examples passed")
    print(f"{'='*60}")
    
    if success_count == total_count:
        print("✅ All examples completed successfully!")
        return 0
    else:
        print(f"❌ {total_count - success_count} examples failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
