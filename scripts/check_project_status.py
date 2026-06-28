#!/usr/bin/env python3
"""Check thesis project setup status.

This script is intentionally lightweight and uses only the Python standard
library so it can run before the thesis programming environment is finalized.
"""

from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "reports" / "project_status.md"

REQUIRED_FILES = [
    "README.md",
    "AGENTS.md",
    "docs/dashboard.md",
    "docs/current_status.md",
    "docs/thesis_plan.md",
    "docs/chapters/01_introduction.md",
    "docs/code_task_spec.md",
    "docs/risk_register.md",
    "docs/artifact_index.md",
    "docs/professor_update_template.md",
    "references/literature_notes.md",
    "src/README.md",
]

PLACEHOLDER_MARKERS = [
    "待填",
    "待確認",
    "尚未",
    "[citation needed]",
]

NEXT_INPUT = [
    "我的論文大方向是：",
    "程式需要做的是：",
    "教授最近要求我完成的是：",
]


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def count_markers(text: str) -> dict[str, int]:
    return {marker: text.count(marker) for marker in PLACEHOLDER_MARKERS}


def build_status() -> tuple[list[str], dict[str, int], list[tuple[str, int]]]:
    missing = []
    marker_totals = {marker: 0 for marker in PLACEHOLDER_MARKERS}
    files_with_markers: list[tuple[str, int]] = []

    for rel_path in REQUIRED_FILES:
        path = ROOT / rel_path
        if not path.exists():
            missing.append(rel_path)
            continue

        text = read_text(path)
        counts = count_markers(text)
        total = sum(counts.values())
        if total:
            files_with_markers.append((rel_path, total))
            for marker, count in counts.items():
                marker_totals[marker] += count

    return missing, marker_totals, files_with_markers


def render_text(missing: list[str], marker_totals: dict[str, int], files_with_markers: list[tuple[str, int]]) -> str:
    lines = [
        "Thesis project status",
        "=====================",
        f"Root: {ROOT}",
        "",
    ]
    if missing:
        lines.append("Missing required files:")
        for rel_path in missing:
            lines.append(f"- {rel_path}")
    else:
        lines.append("Required files: OK")

    lines.extend(["", "Placeholder markers:"])
    for marker, count in marker_totals.items():
        lines.append(f"- {marker}: {count}")

    if files_with_markers:
        lines.extend(["", "Files with placeholder markers:"])
        for rel_path, total in sorted(files_with_markers, key=lambda item: item[1], reverse=True):
            lines.append(f"- {rel_path}: {total}")

    lines.extend(["", "Next input needed:"])
    for item in NEXT_INPUT:
        lines.append(f"- {item}")

    lines.append("")
    if missing:
        lines.append("Status: setup incomplete")
    else:
        lines.append("Status: setup files present; thesis content still needs user/professor input")

    return "\n".join(lines) + "\n"


def render_markdown(missing: list[str], marker_totals: dict[str, int], files_with_markers: list[tuple[str, int]]) -> str:
    lines = [
        "# Project Status Report",
        "",
        f"- Root: `{ROOT}`",
        f"- Required files: {'missing files found' if missing else 'OK'}",
        "",
        "## Missing Required Files",
        "",
    ]
    if missing:
        lines.extend(f"- `{rel_path}`" for rel_path in missing)
    else:
        lines.append("- None")

    lines.extend(["", "## Placeholder Markers", "", "| Marker | Count |", "| --- | ---: |"])
    for marker, count in marker_totals.items():
        lines.append(f"| `{marker}` | {count} |")

    lines.extend(["", "## Files With Placeholder Markers", "", "| File | Count |", "| --- | ---: |"])
    if files_with_markers:
        for rel_path, total in sorted(files_with_markers, key=lambda item: item[1], reverse=True):
            lines.append(f"| `{rel_path}` | {total} |")
    else:
        lines.append("| None | 0 |")

    lines.extend(["", "## Next Input Needed", ""])
    lines.extend(f"- {item}" for item in NEXT_INPUT)

    lines.extend([
        "",
        "## Status",
        "",
        "Setup incomplete." if missing else "Setup files present; thesis content still needs user/professor input.",
        "",
    ])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check thesis project setup status.")
    parser.add_argument(
        "--write-report",
        action="store_true",
        help=f"Write a Markdown report to {REPORT_PATH.relative_to(ROOT)}.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    missing, marker_totals, files_with_markers = build_status()

    print(render_text(missing, marker_totals, files_with_markers), end="")

    if args.write_report:
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(render_markdown(missing, marker_totals, files_with_markers), encoding="utf-8")
        print()
        print(f"Wrote report: {REPORT_PATH.relative_to(ROOT)}")

    if missing:
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
