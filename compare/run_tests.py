#!/usr/bin/env python3
"""Entry script for running trimesh tests with rmesh comparison.

Usage:
    python -m compare.run_tests [pytest-args]

Examples:
    # Run specific test file
    python -m compare.run_tests test/trimesh/tests/test_inertia.py

    # Run with verbose output
    python -m compare.run_tests test/trimesh/tests/test_bounds.py -v

    # Run specific test
    python -m compare.run_tests test/trimesh/tests/test_inertia.py::test_inertia_basic

    # Custom output file
    python -m compare.run_tests test/trimesh/tests/test_mesh.py \\
        --comparison-output=my_results.md
"""

import sys
from pathlib import Path

# Add the project root to path so we can import compare module
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Also add the trimesh test directory so imports work
trimesh_tests = project_root / "test" / "trimesh"
if trimesh_tests.exists():
    sys.path.insert(0, str(trimesh_tests))


def main():
    """Run pytest with comparison mode enabled."""
    import pytest

    # Build pytest arguments
    args = [
        "--compare-rmesh",  # Enable comparison mode
        "-p",
        "compare.conftest",  # Load our conftest plugin
    ]

    # Add any user-provided arguments
    args.extend(sys.argv[1:])

    # If no test path specified, default to a common test
    if not any(arg.endswith(".py") or "::" in arg for arg in args):
        default_test = project_root / "test" / "trimesh" / "tests" / "test_inertia.py"
        if default_test.exists():
            args.append(str(default_test))

    print(f"Running: pytest {' '.join(args)}")
    print()

    # Run pytest
    exit_code = pytest.main(args)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
