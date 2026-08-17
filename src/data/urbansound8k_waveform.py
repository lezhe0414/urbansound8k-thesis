from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd


SplitName = Literal["train", "val", "test"]


@dataclass(frozen=True)
class UrbanSound8KWaveformItem:
    raw_path: Path
    cache_path: Path | None
    class_id: int
    class_name: str
    fold: int
    source_file: str


def pad_or_truncate(waveform: np.ndarray, target_samples: int) -> np.ndarray:
    """Match EfficientAT's downstream fixed-length waveform handling."""
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.size >= target_samples:
        return waveform[:target_samples]
    return np.pad(waveform, (0, target_samples - waveform.size)).astype(np.float32, copy=False)


def load_resampled_waveform(path: Path, sample_rate: int, target_samples: int) -> np.ndarray:
    try:
        import librosa
    except ImportError as exc:  # pragma: no cover - dependency availability controls branch
        raise RuntimeError("librosa is required. Install dependencies with `pip install -r requirements.txt`.") from exc

    waveform, _ = librosa.load(path, sr=sample_rate, mono=True, dtype=np.float32)
    return pad_or_truncate(waveform, target_samples)


class UrbanSound8KWaveformDataset:
    """Raw-waveform UrbanSound8K dataset for models with their own audio frontend."""

    def __init__(
        self,
        raw_dir: str | Path,
        split: SplitName,
        test_fold: int,
        val_fold: int,
        sample_rate: int = 32_000,
        clip_duration_seconds: float = 5.0,
        waveform_cache_dir: str | Path | None = None,
        require_cache: bool = False,
        max_samples: int | None = None,
    ) -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - dependency availability controls branch
            raise RuntimeError("PyTorch is required. Install dependencies with `pip install -r requirements.txt`.") from exc

        self.raw_dir = Path(raw_dir)
        self.split = split
        self.test_fold = int(test_fold)
        self.val_fold = int(val_fold)
        self.sample_rate = int(sample_rate)
        self.target_samples = int(round(float(clip_duration_seconds) * self.sample_rate))
        self.waveform_cache_dir = Path(waveform_cache_dir) if waveform_cache_dir else None
        self.require_cache = bool(require_cache)
        self._torch = torch

        if self.test_fold == self.val_fold:
            raise ValueError("test_fold and val_fold must be different.")
        if self.target_samples <= 0:
            raise ValueError("clip_duration_seconds must produce at least one sample.")

        metadata_path = self.raw_dir / "metadata" / "UrbanSound8K.csv"
        audio_dir = self.raw_dir / "audio"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing UrbanSound8K metadata file: {metadata_path}")
        if not audio_dir.exists():
            raise FileNotFoundError(f"Missing UrbanSound8K audio directory: {audio_dir}")

        metadata = pd.read_csv(metadata_path)
        required = {"slice_file_name", "fold", "classID", "class"}
        missing = required.difference(metadata.columns)
        if missing:
            raise ValueError(f"UrbanSound8K metadata is missing columns: {sorted(missing)}")

        if split == "test":
            selected = metadata[metadata["fold"] == self.test_fold]
        elif split == "val":
            selected = metadata[metadata["fold"] == self.val_fold]
        elif split == "train":
            selected = metadata[(metadata["fold"] != self.test_fold) & (metadata["fold"] != self.val_fold)]
        else:
            raise ValueError(f"Unsupported split: {split}")

        if max_samples is not None:
            selected = selected.head(int(max_samples))

        self.items: list[UrbanSound8KWaveformItem] = []
        for row in selected.to_dict("records"):
            fold = int(row["fold"])
            source_file = str(row["slice_file_name"])
            cache_path = None
            if self.waveform_cache_dir is not None:
                cache_path = self.waveform_cache_dir / f"fold{fold}" / f"{Path(source_file).stem}.npy"
            item = UrbanSound8KWaveformItem(
                raw_path=audio_dir / f"fold{fold}" / source_file,
                cache_path=cache_path,
                class_id=int(row["classID"]),
                class_name=str(row["class"]),
                fold=fold,
                source_file=source_file,
            )
            if not item.raw_path.exists():
                raise FileNotFoundError(f"Missing audio file referenced by metadata: {item.raw_path}")
            if self.require_cache and (item.cache_path is None or not item.cache_path.exists()):
                raise FileNotFoundError(f"Missing required waveform cache entry: {item.cache_path}")
            self.items.append(item)

        if not self.items:
            raise ValueError(f"No items found for split={split}, test_fold={test_fold}, val_fold={val_fold}.")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        item = self.items[index]
        if item.cache_path is not None and item.cache_path.exists():
            waveform = np.load(item.cache_path, allow_pickle=False).astype(np.float32, copy=False)
            waveform = pad_or_truncate(waveform, self.target_samples)
        else:
            waveform = load_resampled_waveform(item.raw_path, self.sample_rate, self.target_samples)
        x = self._torch.from_numpy(waveform)
        y = self._torch.tensor(item.class_id, dtype=self._torch.long)
        return x, y
