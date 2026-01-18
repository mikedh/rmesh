"""Generate API coverage table comparing rmesh.Trimesh vs trimesh.Trimesh."""

import numpy as np


def get_api_coverage() -> dict:
    """Get API coverage data comparing rmesh and trimesh.

    Returns:
        Dict with keys: both, rmesh_only, trimesh_only, coverage_pct
    """
    import rmesh
    import trimesh

    # Create minimal instances
    t = trimesh.Trimesh(vertices=np.zeros((3, 3)), faces=np.array([[0, 1, 2]]))
    r = rmesh.Trimesh(np.zeros((3, 3)), np.array([[0, 1, 2]], dtype=np.int64))

    # Get public attrs (no dunder, no private)
    t_attrs = {a for a in dir(t) if not a.startswith("_")}
    r_attrs = {a for a in dir(r) if not a.startswith("_")}

    both = sorted(t_attrs & r_attrs)
    rmesh_only = sorted(r_attrs - t_attrs)
    trimesh_only = sorted(t_attrs - r_attrs)

    return {
        "both": both,
        "rmesh_only": rmesh_only,
        "trimesh_only": trimesh_only,
        "rmesh_count": len(r_attrs),
        "trimesh_count": len(t_attrs),
        "coverage_pct": len(both) / len(t_attrs) * 100 if t_attrs else 0,
    }


def generate_coverage_table() -> str:
    """Generate markdown table of API coverage.

    Returns:
        Markdown formatted string
    """
    data = get_api_coverage()

    lines = [
        "# rmesh vs trimesh API Coverage",
        "",
        f"**rmesh implements {len(data['both'])}/{data['trimesh_count']} "
        f"trimesh.Trimesh attributes ({data['coverage_pct']:.1f}%)**",
        "",
        "## Implemented (matching trimesh API)",
        "",
        "| Attribute | Type |",
        "|-----------|------|",
    ]

    # Categorize the shared attributes
    import trimesh

    t = trimesh.Trimesh(vertices=np.zeros((3, 3)), faces=np.array([[0, 1, 2]]))

    for attr in data["both"]:
        obj = getattr(t, attr, None)
        if callable(obj):
            attr_type = "method"
        elif isinstance(obj, property) or not callable(obj):
            attr_type = "property"
        else:
            attr_type = "?"
        lines.append(f"| `{attr}` | {attr_type} |")

    lines.extend(
        [
            "",
            "## rmesh-only (not in trimesh)",
            "",
            "| Attribute | Description |",
            "|-----------|-------------|",
        ]
    )

    rmesh_descriptions = {
        "cleanup": "Merge vertices, remove degenerate faces",
        "face_colors": "Per-face RGBA colors",
        "from_arrays": "Create from vertices/faces arrays",
        "material_diffuse": "Get material diffuse color by index",
        "material_has_texture": "Check if material has texture",
        "material_name": "Get material name by index",
        "simplify": "Mesh decimation",
        "uv": "UV texture coordinates",
    }

    for attr in data["rmesh_only"]:
        desc = rmesh_descriptions.get(attr, "")
        lines.append(f"| `{attr}` | {desc} |")

    lines.extend(
        [
            "",
            "## Not yet implemented (trimesh-only)",
            "",
            "<details>",
            "<summary>Click to expand (135 attributes)</summary>",
            "",
            "| Attribute |",
            "|-----------|",
        ]
    )

    for attr in data["trimesh_only"]:
        lines.append(f"| `{attr}` |")

    lines.extend(
        [
            "",
            "</details>",
        ]
    )

    return "\n".join(lines)


def save_coverage_table(path: str = "api_coverage.md") -> None:
    """Generate and save API coverage table.

    Args:
        path: Output file path
    """
    from pathlib import Path

    content = generate_coverage_table()
    Path(path).write_text(content)
    print(f"API coverage table written to: {path}")


if __name__ == "__main__":
    print(generate_coverage_table())
