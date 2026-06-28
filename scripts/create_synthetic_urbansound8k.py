#!/usr/bin/env python3
"""Create a tiny UrbanSound8K-like dataset for smoke testing."""

from __future__ import annotations

import argparse
import csv
import math
import wave
from pathlib import Path

import numpy as np


CLASSES = [
    "air_conditioner",
    "car_horn",
    "children_playing",
    "dog_bark",
    "drilling",
    "engine_idling",
    "gun_shot",
    "jackhammer",
    "siren",
    "street_music",
]


def write_wav(path: Path, frequency: float, sample_rate: int, duration: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = int(sample_rate * duration)
    t = np.arange(samples) / sample_rate
    signal = 0.35 * np.sin(2 * math.pi * frequency * t)
    signal += 0.03 * np.random.default_rng(42).normal(size=samples)
    pcm = np.clip(signal, -1.0, 1.0)
    pcm = (pcm * 32767).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


def create_dataset(root: Path, sample_rate: int, duration: float, samples_per_fold: int) -> None:
    metadata_dir = root / "metadata"
    audio_dir = root / "audio"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for fold in range(1, 11):
        for index in range(samples_per_fold):
            class_id = (fold + index) % len(CLASSES)
            class_name = CLASSES[class_id]
            file_name = f"synthetic-{fold}-{index}-{class_name}.wav"
            frequency = 160.0 + class_id * 70.0 + fold
            write_wav(audio_dir / f"fold{fold}" / file_name, frequency, sample_rate, duration)
            rows.append(
                {
                    "slice_file_name": file_name,
                    "fsID": 0,
                    "start": 0.0,
                    "end": duration,
                    "salience": 1,
                    "fold": fold,
                    "classID": class_id,
                    "class": class_name,
                }
            )

    with (metadata_dir / "UrbanSound8K.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote synthetic UrbanSound8K-like data to {root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a tiny synthetic UrbanSound8K-like dataset.")
    parser.add_argument("--out-dir", default="data/raw/UrbanSound8K_synthetic")
    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument("--samples-per-fold", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    create_dataset(Path(args.out_dir), args.sample_rate, args.duration, args.samples_per_fold)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
