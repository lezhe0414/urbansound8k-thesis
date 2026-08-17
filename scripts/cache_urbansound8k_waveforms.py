from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.urbansound8k_waveform import load_resampled_waveform


def build_cache(
    raw_dir: Path,
    out_dir: Path,
    sample_rate: int,
    clip_duration_seconds: float,
    limit: int | None = None,
) -> Path:
    metadata_path = raw_dir / "metadata" / "UrbanSound8K.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing UrbanSound8K metadata: {metadata_path}")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite a non-empty waveform cache: {out_dir}. "
            "Choose a new output directory."
        )

    metadata = pd.read_csv(metadata_path)
    if limit is not None:
        metadata = metadata.head(int(limit))
    target_samples = int(round(sample_rate * clip_duration_seconds))
    rows: list[dict] = []
    out_dir.mkdir(parents=True, exist_ok=True)

    for row in tqdm(metadata.to_dict("records"), desc="Caching 32 kHz waveforms"):
        fold = int(row["fold"])
        source_file = str(row["slice_file_name"])
        source_path = raw_dir / "audio" / f"fold{fold}" / source_file
        waveform = load_resampled_waveform(source_path, sample_rate, target_samples)
        fold_dir = out_dir / f"fold{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        output_path = fold_dir / f"{Path(source_file).stem}.npy"
        np.save(output_path, waveform, allow_pickle=False)
        rows.append(
            {
                "slice_file_name": source_file,
                "fold": fold,
                "classID": int(row["classID"]),
                "class": str(row["class"]),
                "path": str(output_path.relative_to(out_dir)),
            }
        )

    pd.DataFrame(rows).to_csv(out_dir / "metadata.csv", index=False)
    manifest = {
        "source": str(raw_dir),
        "sample_rate": int(sample_rate),
        "clip_duration_seconds": float(clip_duration_seconds),
        "target_samples": target_samples,
        "dtype": "float32",
        "items": len(rows),
        "feature_extraction": "waveform-only cache; EfficientAT log-Mel frontend remains online",
    }
    (out_dir / "cache_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a separate 32 kHz waveform cache for EfficientAT.")
    parser.add_argument("--raw-dir", default="data/raw/UrbanSound8K_soundata")
    parser.add_argument("--out-dir", default="data/processed/urbansound8k_waveforms_32k_5s")
    parser.add_argument("--sample-rate", type=int, default=32_000)
    parser.add_argument("--clip-duration-seconds", type=float, default=5.0)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    path = build_cache(
        raw_dir=Path(args.raw_dir),
        out_dir=Path(args.out_dir),
        sample_rate=args.sample_rate,
        clip_duration_seconds=args.clip_duration_seconds,
        limit=args.limit,
    )
    print(f"Wrote waveform cache to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
