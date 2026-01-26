"""Results aggregation and formatting for comparison tests."""

import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ComparisonResult:
    """Result of comparing a single property/method between trimesh and rmesh."""

    name: str
    trimesh_value: Any = None
    rmesh_value: Any = None
    trimesh_time: float = 0.0
    rmesh_time: float = 0.0
    identical: bool | None = None
    error: str | None = None
    rmesh_implemented: bool = True

    @property
    def speedup(self) -> float | None:
        """Calculate speedup ratio (trimesh_time / rmesh_time)."""
        if self.trimesh_time > 0 and self.rmesh_time > 0:
            return self.trimesh_time / self.rmesh_time
        return None

    def to_row(self) -> list:
        """Convert to a table row."""
        speedup = self.speedup
        speedup_str = f"{speedup:.2f}x" if speedup else "N/A"

        if not self.rmesh_implemented:
            identical_str = "Not impl"
        elif self.error:
            identical_str = f"Error: {self.error[:20]}"
        elif self.identical is None:
            identical_str = "N/A"
        else:
            identical_str = "Yes" if self.identical else "No"

        return [
            self.name,
            f"{self.trimesh_time:.6f}" if self.trimesh_time > 0 else "N/A",
            f"{self.rmesh_time:.6f}" if self.rmesh_time > 0 else "N/A",
            speedup_str,
            identical_str,
        ]


def arrays_equal(
    a: np.ndarray, b: np.ndarray, rtol: float = 1e-7, atol: float = 1e-10
) -> bool:
    """Compare two numpy arrays with tolerance.

    Args:
        a: First array
        b: Second array
        rtol: Relative tolerance
        atol: Absolute tolerance

    Returns:
        True if arrays are equal within tolerance
    """
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False

    a = np.asarray(a)
    b = np.asarray(b)

    if a.shape != b.shape:
        # Shape mismatch - can't directly compare
        # This is common for vertices/faces due to processing differences
        return False

    # Handle empty arrays
    if a.size == 0 and b.size == 0:
        return True

    # For integer arrays, compare as integers
    if np.issubdtype(a.dtype, np.integer) and np.issubdtype(b.dtype, np.integer):
        return np.array_equal(a.astype(np.int64), b.astype(np.int64))

    # For float arrays, convert to same dtype and use allclose
    a_float = a.astype(np.float64)
    b_float = b.astype(np.float64)
    return np.allclose(a_float, b_float, rtol=rtol, atol=atol)


def values_equal(a: Any, b: Any, rtol: float = 1e-5, atol: float = 1e-8) -> bool:  # noqa: C901
    """Type-aware comparison of values within tolerance.

    Args:
        a: First value
        b: Second value
        rtol: Relative tolerance for floats
        atol: Absolute tolerance for floats

    Returns:
        True (Python bool) if values are equal within tolerance
    """
    # Handle None
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False

    # Handle numpy arrays
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return bool(arrays_equal(a, b, rtol, atol))

    # Handle floats
    if isinstance(a, (float, np.floating)) and isinstance(b, (float, np.floating)):
        if np.isnan(a) and np.isnan(b):
            return True
        return bool(np.isclose(a, b, rtol=rtol, atol=atol))

    # Handle booleans
    if isinstance(a, (bool, np.bool_)) and isinstance(b, (bool, np.bool_)):
        return bool(a) == bool(b)

    # Handle integers
    if isinstance(a, (int, np.integer)) and isinstance(b, (int, np.integer)):
        return int(a) == int(b)

    # Handle lists/tuples - convert to arrays if numeric
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        try:
            return bool(arrays_equal(np.array(a), np.array(b), rtol, atol))
        except (ValueError, TypeError):
            return a == b

    # Handle matrices (3x3 for inertia tensor)
    if hasattr(a, "shape") and hasattr(b, "shape"):
        return bool(arrays_equal(np.asarray(a), np.asarray(b), rtol, atol))

    # Default comparison
    return bool(a == b)


@dataclass
class TestSessionResults:
    """Aggregate results across a test session."""

    results: list[ComparisonResult] = field(default_factory=list)
    results_by_property: dict[str, list[ComparisonResult]] = field(
        default_factory=lambda: defaultdict(list)
    )

    def add_result(self, result: ComparisonResult) -> None:
        """Add a comparison result."""
        self.results.append(result)
        self.results_by_property[result.name].append(result)

    def clear(self) -> None:
        """Clear all results."""
        self.results.clear()
        self.results_by_property.clear()

    def summary(self) -> dict:
        """Get a summary of results."""
        total = len(self.results)
        identical = sum(1 for r in self.results if r.identical is True)
        not_identical = sum(1 for r in self.results if r.identical is False)
        not_implemented = sum(1 for r in self.results if not r.rmesh_implemented)
        errors = sum(1 for r in self.results if r.error is not None)

        speedups = [
            r.speedup for r in self.results if r.speedup is not None and r.speedup > 0
        ]

        return {
            "total": total,
            "identical": identical,
            "not_identical": not_identical,
            "not_implemented": not_implemented,
            "errors": errors,
            "speedup_avg": sum(speedups) / len(speedups) if speedups else None,
            "speedup_min": min(speedups) if speedups else None,
            "speedup_max": max(speedups) if speedups else None,
        }

    def aggregate_by_property(self) -> dict[str, dict]:
        """Aggregate results by property name."""
        aggregated = {}

        for name, results in self.results_by_property.items():
            identical_count = sum(1 for r in results if r.identical is True)
            total = len(results)

            trimesh_times = [r.trimesh_time for r in results if r.trimesh_time > 0]
            rmesh_times = [r.rmesh_time for r in results if r.rmesh_time > 0]
            speedups = [
                r.speedup for r in results if r.speedup is not None and r.speedup > 0
            ]

            aggregated[name] = {
                "total_calls": total,
                "identical_count": identical_count,
                "identical_pct": (identical_count / total * 100) if total > 0 else 0,
                "trimesh_avg": sum(trimesh_times) / len(trimesh_times)
                if trimesh_times
                else 0,
                "rmesh_avg": sum(rmesh_times) / len(rmesh_times) if rmesh_times else 0,
                "speedup_avg": sum(speedups) / len(speedups) if speedups else None,
                "speedup_min": min(speedups) if speedups else None,
                "speedup_max": max(speedups) if speedups else None,
                "implemented": any(r.rmesh_implemented for r in results),
            }

        return aggregated

    def to_json(self) -> str:
        """Export results as JSON."""
        data = {
            "summary": self.summary(),
            "by_property": self.aggregate_by_property(),
        }
        return json.dumps(data, indent=2)


# Global singleton for session results
_session_results: TestSessionResults | None = None


def get_session_results() -> TestSessionResults:
    """Get the global session results singleton."""
    global _session_results
    if _session_results is None:
        _session_results = TestSessionResults()
    return _session_results


def reset_session_results() -> None:
    """Reset the global session results."""
    global _session_results
    _session_results = TestSessionResults()


def generate_comparison_report() -> str:
    """Generate full comparison report with both tables.

    Returns:
        Markdown formatted report
    """
    from .api_coverage import get_api_coverage

    results = get_session_results()
    aggregated = results.aggregate_by_property()
    summary = results.summary()

    lines = [
        "# rmesh vs trimesh Comparison",
        "",
    ]

    # API Coverage section
    try:
        coverage = get_api_coverage()
        lines.extend(
            [
                "## API Coverage",
                "",
                f"**{len(coverage['both'])}/{coverage['trimesh_count']} "
                f"trimesh.Trimesh attributes ({coverage['coverage_pct']:.1f}%)**",
                "",
                "| Attribute | Status |",
                "|-----------|--------|",
            ]
        )

        for attr in sorted(coverage["both"]):
            lines.append(f"| `{attr}` | ✓ |")

        lines.extend(
            [
                "",
                f"<details><summary>Not implemented "
                f"({len(coverage['trimesh_only'])} attributes)</summary>",
                "",
            ]
        )
        for attr in sorted(coverage["trimesh_only"]):
            lines.append(f"- `{attr}`")
        lines.extend(["", "</details>", ""])

    except Exception as e:
        lines.extend([f"API coverage unavailable: {e}", ""])

    # Performance comparison section
    if aggregated:
        lines.extend(
            [
                "## Performance Comparison",
                "",
                "| Property | Calls | Equal | Speedup (avg) | Speedup (min/max) |",
                "|----------|-------|-------|---------------|-------------------|",
            ]
        )

        for name in sorted(aggregated.keys()):
            stats = aggregated[name]

            if not stats["implemented"]:
                continue

            equal_pct = f"{stats['identical_pct']:.1f}%"

            if stats["speedup_avg"]:
                speedup_avg = f"{stats['speedup_avg']:.1f}x"
                speedup_range = (
                    f"{stats['speedup_min']:.1f}x / {stats['speedup_max']:.1f}x"
                )
            else:
                speedup_avg = "-"
                speedup_range = "-"

            lines.append(
                f"| `{name}` | {stats['total_calls']} | {equal_pct} "
                f"| {speedup_avg} | {speedup_range} |"
            )

        # Summary
        lines.extend(
            [
                "",
                "## Summary",
                "",
                f"- **Total comparisons:** {summary['total']}",
                f"- **Identical results:** {summary['identical']} "
                f"({summary['identical'] / summary['total'] * 100:.1f}%)"
                if summary["total"] > 0
                else "- **Identical results:** 0",
            ]
        )

        if summary["speedup_avg"]:
            avg = summary["speedup_avg"]
            min_s = summary["speedup_min"]
            max_s = summary["speedup_max"]
            lines.append(
                f"- **Speedup:** {avg:.1f}x avg ({min_s:.1f}x min, {max_s:.1f}x max)"
            )
    else:
        lines.extend(
            [
                "## Performance Comparison",
                "",
                "No comparison results recorded.",
            ]
        )

    return "\n".join(lines)


def save_results(
    output_path: str = "comparison.md", json_path: str | None = None
) -> None:
    """Save comparison report to file.

    Args:
        output_path: Path for markdown output (default: comparison.md)
        json_path: Optional path for JSON output
    """
    from pathlib import Path

    content = generate_comparison_report()

    if output_path:
        Path(output_path).write_text(content)
        print(f"Comparison report written to: {output_path}")

    if json_path:
        results = get_session_results()
        Path(json_path).write_text(results.to_json())
        print(f"Comparison JSON written to: {json_path}")
