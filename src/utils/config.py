from __future__ import annotations

from pathlib import Path


def load_config(path: str | Path) -> dict:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - exercised only without deps
        raise RuntimeError("PyYAML is required. Install dependencies with `pip install -r requirements.txt`.") from exc

    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data
