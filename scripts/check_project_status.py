#!/usr/bin/env python3
"""Check thesis project setup status.

This script is intentionally lightweight and uses only the Python standard
library so it can run before the thesis programming environment is finalized.
"""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

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


def main() -> int:
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

    print("Thesis project status")
    print("=====================")
    print(f"Root: {ROOT}")
    print()

    if missing:
        print("Missing required files:")
        for rel_path in missing:
            print(f"- {rel_path}")
    else:
        print("Required files: OK")

    print()
    print("Placeholder markers:")
    for marker, count in marker_totals.items():
        print(f"- {marker}: {count}")

    if files_with_markers:
        print()
        print("Files with placeholder markers:")
        for rel_path, total in sorted(files_with_markers, key=lambda item: item[1], reverse=True):
            print(f"- {rel_path}: {total}")

    print()
    print("Next input needed:")
    for item in NEXT_INPUT:
        print(f"- {item}")

    print()
    if missing:
        print("Status: setup incomplete")
        return 1

    print("Status: setup files present; thesis content still needs user/professor input")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
