"""Results aggregation and reporting for rmesh/trimesh comparison.

Uses SQLite for thread-safe, persistent storage of comparison results.
"""

import json
import sqlite3
from dataclasses import dataclass
from importlib.metadata import version as _pkg_version
from typing import Any

import numpy as np


@dataclass
class ComparisonResult:
    """Result of comparing a single attribute between trimesh and rmesh."""

    name: str
    trimesh_time: float = 0.0
    rmesh_time: float = 0.0
    identical: bool | None = None
    error: str | None = None

    @property
    def speedup(self) -> float | None:
        """Speedup ratio (trimesh_time / rmesh_time)."""
        if self.trimesh_time > 0 and self.rmesh_time > 0:
            return self.trimesh_time / self.rmesh_time
        return None

    def to_row(self) -> list:
        """Format as a table row."""
        speedup = self.speedup
        speedup_str = f"{speedup:.2f}x" if speedup else "N/A"

        if self.error:
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


# ---------------------------------------------------------------------------
# Value comparison helpers
# ---------------------------------------------------------------------------


def arrays_equal(
    a: np.ndarray,
    b: np.ndarray,
    rtol: float = 1e-7,
    atol: float = 1e-10,
) -> bool:
    """Compare two numpy arrays with tolerance."""
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False

    a = np.asarray(a)
    b = np.asarray(b)

    if a.shape != b.shape:
        return False

    if a.size == 0 and b.size == 0:
        return True

    if np.issubdtype(a.dtype, np.integer) and np.issubdtype(
        b.dtype, np.integer
    ):
        return np.array_equal(a.astype(np.int64), b.astype(np.int64))

    return np.allclose(
        a.astype(np.float64), b.astype(np.float64), rtol=rtol, atol=atol
    )


def values_equal(  # noqa: C901
    a: Any,
    b: Any,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> bool:
    """Type-aware comparison of values within tolerance."""
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False

    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return bool(arrays_equal(a, b, rtol, atol))

    if isinstance(a, (float, np.floating)) and isinstance(
        b, (float, np.floating)
    ):
        if np.isnan(a) and np.isnan(b):
            return True
        return bool(np.isclose(a, b, rtol=rtol, atol=atol))

    if isinstance(a, (bool, np.bool_)) and isinstance(b, (bool, np.bool_)):
        return bool(a) == bool(b)

    if isinstance(a, (int, np.integer)) and isinstance(b, (int, np.integer)):
        return int(a) == int(b)

    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        try:
            return bool(arrays_equal(np.array(a), np.array(b), rtol, atol))
        except (ValueError, TypeError):
            return a == b

    if hasattr(a, "shape") and hasattr(b, "shape"):
        return bool(arrays_equal(np.asarray(a), np.asarray(b), rtol, atol))

    return bool(a == b)


# ---------------------------------------------------------------------------
# Session results (SQLite-backed)
# ---------------------------------------------------------------------------

def _rmesh_version() -> str:
    """Get the installed rmesh version, or 'unknown' if unavailable."""
    try:
        return _pkg_version("rmesh")
    except Exception:
        return "unknown"


_CREATE_TABLE = """\
CREATE TABLE IF NOT EXISTS comparisons (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT    NOT NULL,
    trimesh_time  REAL    NOT NULL DEFAULT 0.0,
    rmesh_time    REAL    NOT NULL DEFAULT 0.0,
    identical     INTEGER,
    error         TEXT,
    rmesh_version TEXT    NOT NULL DEFAULT 'unknown',
    created_at    TEXT    NOT NULL DEFAULT (datetime('now'))
);
"""

_CREATE_INDEX = """\
CREATE INDEX IF NOT EXISTS idx_comparisons_name ON comparisons(name);
"""


class TestSessionResults:
    """Aggregate results across a test session, backed by SQLite."""

    def __init__(self, db_path: str = ":memory:") -> None:
        self._db_path = db_path
        self._version = _rmesh_version()
        self._conn = sqlite3.connect(
            db_path, check_same_thread=False
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._ensure_table()

    def _ensure_table(self) -> None:
        self._conn.execute(_CREATE_TABLE)
        self._conn.execute(_CREATE_INDEX)
        self._conn.commit()

    def add_result(self, result: ComparisonResult) -> None:
        self._conn.execute(
            "INSERT INTO comparisons (name, trimesh_time, rmesh_time, identical, error, rmesh_version) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                result.name,
                result.trimesh_time,
                result.rmesh_time,
                (
                    None
                    if result.identical is None
                    else (1 if result.identical else 0)
                ),
                result.error,
                self._version,
            ),
        )
        self._conn.commit()

    def summary(self) -> dict:
        row = self._conn.execute(
            """\
            SELECT
                COUNT(*)                                             AS total,
                SUM(CASE WHEN identical = 1 THEN 1 ELSE 0 END)      AS identical,
                SUM(CASE WHEN identical = 0 THEN 1 ELSE 0 END)      AS not_identical,
                SUM(CASE WHEN error IS NOT NULL THEN 1 ELSE 0 END)   AS errors,
                AVG(CASE
                    WHEN trimesh_time > 0 AND rmesh_time > 0
                    THEN trimesh_time / rmesh_time END)              AS speedup_avg,
                MIN(CASE
                    WHEN trimesh_time > 0 AND rmesh_time > 0
                    THEN trimesh_time / rmesh_time END)              AS speedup_min,
                MAX(CASE
                    WHEN trimesh_time > 0 AND rmesh_time > 0
                    THEN trimesh_time / rmesh_time END)              AS speedup_max
            FROM comparisons
            """
        ).fetchone()

        return {
            "total": row[0],
            "identical": row[1] or 0,
            "not_identical": row[2] or 0,
            "errors": row[3] or 0,
            "speedup_avg": row[4],
            "speedup_min": row[5],
            "speedup_max": row[6],
        }

    def aggregate_by_property(self) -> dict[str, dict]:
        rows = self._conn.execute(
            """\
            SELECT
                name,
                COUNT(*)                                             AS total_calls,
                SUM(CASE WHEN identical = 1 THEN 1 ELSE 0 END)      AS identical_count,
                AVG(CASE WHEN trimesh_time > 0 THEN trimesh_time END) AS trimesh_avg,
                AVG(CASE WHEN rmesh_time > 0 THEN rmesh_time END)    AS rmesh_avg,
                AVG(CASE
                    WHEN trimesh_time > 0 AND rmesh_time > 0
                    THEN trimesh_time / rmesh_time END)              AS speedup_avg,
                MIN(CASE
                    WHEN trimesh_time > 0 AND rmesh_time > 0
                    THEN trimesh_time / rmesh_time END)              AS speedup_min,
                MAX(CASE
                    WHEN trimesh_time > 0 AND rmesh_time > 0
                    THEN trimesh_time / rmesh_time END)              AS speedup_max
            FROM comparisons
            GROUP BY name
            """
        ).fetchall()

        aggregated = {}
        for r in rows:
            name = r[0]
            total = r[1]
            identical_count = r[2] or 0
            aggregated[name] = {
                "total_calls": total,
                "identical_count": identical_count,
                "identical_pct": (
                    (identical_count / total * 100) if total > 0 else 0
                ),
                "trimesh_avg": r[3] or 0,
                "rmesh_avg": r[4] or 0,
                "speedup_avg": r[5],
                "speedup_min": r[6],
                "speedup_max": r[7],
            }
        return aggregated

    def to_json(self) -> str:
        return json.dumps(
            {
                "summary": self.summary(),
                "by_property": self.aggregate_by_property(),
            },
            indent=2,
        )

    def clear(self) -> None:
        self._conn.execute("DELETE FROM comparisons")
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------

_session_results: TestSessionResults | None = None


def get_session_results(db_path: str = ":memory:") -> TestSessionResults:
    global _session_results
    if _session_results is None:
        _session_results = TestSessionResults(db_path=db_path)
    return _session_results


def reset_session_results(db_path: str = ":memory:") -> None:
    global _session_results
    if _session_results is not None:
        _session_results.close()
    _session_results = TestSessionResults(db_path=db_path)


# ---------------------------------------------------------------------------
# Report generation (includes inline API coverage)
# ---------------------------------------------------------------------------


def _md_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    """Render a markdown table with constant-width, padded columns."""
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    sep = "|".join("-" * (w + 2) for w in widths)
    hdr = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |"
    lines = [hdr, f"|{sep}|"]
    for row in rows:
        line = "| " + " | ".join(
            c.ljust(w) for c, w in zip(row, widths)
        ) + " |"
        lines.append(line)
    return lines


def _get_api_coverage() -> dict | None:
    """Compute API coverage between rmesh.Trimesh and trimesh.Trimesh."""
    try:
        import trimesh

        import rmesh

        t = trimesh.Trimesh(
            vertices=np.zeros((3, 3)), faces=np.array([[0, 1, 2]])
        )
        r = rmesh.Trimesh(
            np.zeros((3, 3)), np.array([[0, 1, 2]], dtype=np.int64)
        )

        t_attrs = {a for a in dir(t) if not a.startswith("_")}
        r_attrs = {a for a in dir(r) if not a.startswith("_")}

        both = sorted(t_attrs & r_attrs)
        trimesh_only = sorted(t_attrs - r_attrs)

        return {
            "both": both,
            "trimesh_only": trimesh_only,
            "trimesh_count": len(t_attrs),
            "coverage_pct": (
                len(both) / len(t_attrs) * 100 if t_attrs else 0
            ),
        }
    except Exception:
        return None


def generate_comparison_report() -> str:
    """Generate a full markdown comparison report."""
    results = get_session_results()
    aggregated = results.aggregate_by_property()
    summary = results.summary()

    lines = ["# rmesh vs trimesh Comparison", ""]

    # API Coverage section
    coverage = _get_api_coverage()
    if coverage is not None:
        lines.extend(
            [
                "## API Coverage",
                "",
                f"**{len(coverage['both'])}/{coverage['trimesh_count']} "
                f"trimesh.Trimesh attributes "
                f"({coverage['coverage_pct']:.1f}%)**",
                "",
            ]
        )
        lines.extend(_md_table(
            ["Attribute", "Status"],
            [[f"`{attr}`", "✓"] for attr in coverage["both"]],
        ))

        lines.extend(
            [
                "",
                f"<details><summary>Not implemented "
                f"({len(coverage['trimesh_only'])} attributes)</summary>",
                "",
            ]
        )
        for attr in coverage["trimesh_only"]:
            lines.append(f"- `{attr}`")
        lines.extend(["", "</details>", ""])

    # Performance comparison section
    if aggregated:
        lines.extend(["## Performance Comparison", ""])

        perf_rows = []
        for name in sorted(aggregated):
            stats = aggregated[name]
            equal_pct = f"{stats['identical_pct']:.1f}%"
            if stats["speedup_avg"]:
                speedup_avg = f"{stats['speedup_avg']:.1f}x"
                speedup_range = (
                    f"{stats['speedup_min']:.1f}x / "
                    f"{stats['speedup_max']:.1f}x"
                )
            else:
                speedup_avg = "-"
                speedup_range = "-"
            perf_rows.append([
                f"`{name}`",
                str(stats["total_calls"]),
                equal_pct,
                speedup_avg,
                speedup_range,
            ])

        lines.extend(_md_table(
            ["Property", "Calls", "Equal", "Speedup (avg)", "Speedup (min/max)"],
            perf_rows,
        ))

        # Summary
        lines.extend(
            [
                "",
                "## Summary",
                "",
                f"- **Total comparisons:** {summary['total']}",
            ]
        )
        if summary["total"] > 0:
            pct = summary["identical"] / summary["total"] * 100
            lines.append(
                f"- **Identical results:** {summary['identical']} "
                f"({pct:.1f}%)"
            )
        else:
            lines.append("- **Identical results:** 0")

        if summary["speedup_avg"]:
            lines.append(
                f"- **Speedup:** {summary['speedup_avg']:.1f}x avg "
                f"({summary['speedup_min']:.1f}x min, "
                f"{summary['speedup_max']:.1f}x max)"
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
    output_path: str = "comparison.md",
    json_path: str | None = None,
) -> None:
    """Save comparison report to file(s)."""
    from pathlib import Path

    content = generate_comparison_report()

    if output_path:
        Path(output_path).write_text(content)
        print(f"Comparison report written to: {output_path}")

    if json_path:
        results = get_session_results()
        Path(json_path).write_text(results.to_json())
        print(f"Comparison JSON written to: {json_path}")
