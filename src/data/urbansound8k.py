from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd


SplitName = Literal["train", "val", "test"]
FeatureRepresentation = Literal["mel", "mel_delta_delta"]


def temporal_delta(values: np.ndarray, width: int = 2) -> np.ndarray:
    """Compute regression deltas along time without changing the cached Mel data."""
    if values.ndim != 2:
        raise ValueError(f"Expected a 2D Mel-spectrogram, got shape {values.shape}.")
    if width < 1:
        raise ValueError("Delta width must be at least 1.")

    time_steps = values.shape[-1]
    padded = np.pad(values, ((0, 0), (width, width)), mode="edge")
    delta = np.zeros_like(values, dtype=np.float32)
    denominator = 2.0 * sum(offset * offset for offset in range(1, width + 1))
    for offset in range(1, width + 1):
        future = padded[:, width + offset : width + offset + time_steps]
        past = padded[:, width - offset : width - offset + time_steps]
        delta += offset * (future - past)
    return (delta / denominator).astype(np.float32, copy=False)


def build_feature_channels(mel: np.ndarray, representation: FeatureRepresentation) -> np.ndarray:
    """Create model input channels from one normalized Mel-spectrogram."""
    if representation == "mel":
        return mel[np.newaxis, ...].astype(np.float32, copy=False)
    if representation == "mel_delta_delta":
        delta = temporal_delta(mel)
        delta_delta = temporal_delta(delta)
        return np.stack((mel, delta, delta_delta), axis=0).astype(np.float32, copy=False)
    raise ValueError(f"Unsupported feature representation: {representation}")


@dataclass(frozen=True)
class UrbanSound8KItem:
    path: Path
    class_id: int
    class_name: str
    fold: int
    source_file: str


class UrbanSound8KMelDataset:
    """PyTorch-compatible dataset for cached UrbanSound8K Mel-spectrograms."""

    def __init__(
        self,
        processed_dir: str | Path,
        split: SplitName,
        test_fold: int,
        val_fold: int | None = None,
        max_samples: int | None = None,
        preload: bool = False,
        feature_representation: FeatureRepresentation = "mel",
    ) -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - exercised only without deps
            raise RuntimeError("PyTorch is required. Install dependencies with `pip install -r requirements.txt`.") from exc

        self.processed_dir = Path(processed_dir)
        self.split = split
        self.test_fold = int(test_fold)
        self.val_fold = int(val_fold) if val_fold is not None else self._default_val_fold(self.test_fold)
        if feature_representation not in {"mel", "mel_delta_delta"}:
            raise ValueError(f"Unsupported feature representation: {feature_representation}")
        self.feature_representation = feature_representation
        self._torch = torch

        metadata_path = self.processed_dir / "metadata.csv"
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Missing processed metadata: {metadata_path}. "
                "Run `python3 -m src.preprocess --raw-dir data/raw/UrbanSound8K --out-dir data/processed/urbansound8k_mels` first."
            )

        metadata = pd.read_csv(metadata_path)
        self.items = self._select_items(metadata)
        if max_samples is not None:
            self.items = self.items[: int(max_samples)]
        if not self.items:
            raise ValueError(f"No items found for split={split}, test_fold={test_fold}, val_fold={self.val_fold}.")
        self._cache = self._preload_items() if preload else None

    @staticmethod
    def _default_val_fold(test_fold: int) -> int:
        return 1 if test_fold == 10 else test_fold + 1

    def _select_items(self, metadata: pd.DataFrame) -> list[UrbanSound8KItem]:
        required = {"path", "classID", "class", "fold", "slice_file_name"}
        missing = required.difference(metadata.columns)
        if missing:
            raise ValueError(f"Processed metadata is missing columns: {sorted(missing)}")

        if self.split == "test":
            selected = metadata[metadata["fold"] == self.test_fold]
        elif self.split == "val":
            selected = metadata[metadata["fold"] == self.val_fold]
        else:
            selected = metadata[(metadata["fold"] != self.test_fold) & (metadata["fold"] != self.val_fold)]

        items: list[UrbanSound8KItem] = []
        for row in selected.to_dict("records"):
            path = Path(row["path"])
            if not path.is_absolute():
                path = self.processed_dir / path
            items.append(
                UrbanSound8KItem(
                    path=path,
                    class_id=int(row["classID"]),
                    class_name=str(row["class"]),
                    fold=int(row["fold"]),
                    source_file=str(row["slice_file_name"]),
                )
            )
        return items

    def __len__(self) -> int:
        return len(self.items)

    def _preload_items(self) -> list:
        cache = []
        for item in self.items:
            cache.append((self._load_mel(item.path), int(item.class_id)))
        return cache

    @staticmethod
    def _load_mel(path: Path) -> np.ndarray:
        with np.load(path) as payload:
            return payload["mel"].astype(np.float32)

    def __getitem__(self, index: int):
        item = self.items[index]
        if self._cache is None:
            mel = self._load_mel(item.path)
            class_id = item.class_id
        else:
            mel, class_id = self._cache[index]
        channels = build_feature_channels(mel, self.feature_representation)
        x = self._torch.from_numpy(channels)
        y = self._torch.tensor(class_id, dtype=self._torch.long)
        return x, y
