"""Results aggregation and formatting for comparison tests."""

import json
from collections import defaultdict
from dataclasses import dataclass, field

from .comparison import ComparisonResult


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
